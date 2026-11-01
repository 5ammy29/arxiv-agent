import hashlib
import json
import logging
from typing import Any, Protocol
from uuid import UUID

from project.application.papers.processing_lock import (
    ActivePaperProcessingLease,
    PaperProcessingLockManager,
)
from project.infrastructure.database.vector_indexes import (
    delete_paper_vector_index,
    get_paper_vector_index,
    save_paper_vector_index,
)
from project.infrastructure.storage.vector_indexes import (
    delete_vector_index,
    download_vector_index,
    upload_vector_index,
)
from project.processing.chunker import TextChunk


logger = logging.getLogger(__name__)


class VectorStoreProtocol(Protocol):
    model_name: str
    model: Any

    def reset(self) -> None: ...
    def add_chunks(self, chunks: list[TextChunk]) -> None: ...
    def serialize_index(self) -> bytes: ...
    def load_index(
        self, index_bytes: bytes, chunks: list[TextChunk], chunk_ids: list[int]
    ) -> None: ...


def _is_missing_storage_object(error: Exception) -> bool:
    return isinstance(error, FileNotFoundError) or any(
        getattr(error, attribute, None) == 404
        for attribute in ("status", "status_code", "code")
    )


def _fallback_content_fingerprint(chunks: list[TextChunk]) -> str:
    """Use only for direct callers outside the paper-preparation workflow."""
    payload = [
        {"chunk_id": chunk.chunk_id, "page": chunk.page, "text": chunk.text}
        for chunk in chunks
    ]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class VectorIndexCache:
    def __init__(
        self,
        *,
        lock_manager: PaperProcessingLockManager | None = None,
    ) -> None:
        self.lock_manager = lock_manager or PaperProcessingLockManager()

    def _discard_metadata_and_object(
        self,
        paper_id: UUID,
        active_lease: ActivePaperProcessingLease,
    ) -> None:
        active_lease.assert_active()
        storage_path = delete_paper_vector_index(
            paper_id,
            lease=active_lease.lease,
        )
        if isinstance(storage_path, str) and storage_path:
            # Readers may still be downloading the immutable object. Cleanup
            # uses a grace period and database references before deleting it.
            logger.info(
                "vector_index_cleanup_deferred paper_id=%s path=%s",
                paper_id,
                storage_path,
            )

    def get_or_build_index(
        self,
        paper_id: str | UUID,
        chunks: list[TextChunk],
        vector_store: VectorStoreProtocol,
        *,
        content_fingerprint: str | None = None,
        chunking_fingerprint: str | None = None,
    ) -> bool:
        paper_uuid = UUID(str(paper_id))
        with self.lock_manager.acquire(paper_uuid) as active_lease:
            return self._get_or_build_index(
                paper_uuid,
                chunks,
                vector_store,
                active_lease=active_lease,
                content_fingerprint=content_fingerprint,
                chunking_fingerprint=chunking_fingerprint,
            )

    def _get_or_build_index(
        self,
        paper_id: UUID,
        chunks: list[TextChunk],
        vector_store: VectorStoreProtocol,
        *,
        active_lease: ActivePaperProcessingLease,
        content_fingerprint: str | None = None,
        chunking_fingerprint: str | None = None,
    ) -> bool:
        if not chunks:
            raise ValueError("Cannot cache an index without chunks.")

        paper_uuid = paper_id
        current_chunk_ids = [chunk.chunk_id for chunk in chunks]
        if len(set(current_chunk_ids)) != len(current_chunk_ids):
            raise ValueError("Current chunks contain duplicate chunk IDs.")

        if content_fingerprint is None:
            content_fingerprint = _fallback_content_fingerprint(chunks)

        if chunking_fingerprint is None:
            chunking_fingerprint = "direct-chunks-v1"

        if not content_fingerprint.strip() or not chunking_fingerprint.strip():
            raise ValueError("Cache fingerprints must not be empty.")

        embedding_dimension = vector_store.model.get_sentence_embedding_dimension()
        if embedding_dimension is None or embedding_dimension <= 0:
            raise ValueError("Embedding model has an invalid dimension.")

        metadata = get_paper_vector_index(paper_uuid)
        store_is_clean = False
        invalidate_cache = False

        if metadata is not None:
            cached_chunk_ids = self._parse_chunk_ids(metadata.get("chunk_ids"))
            cache_is_valid = (
                metadata.get("embedding_model") == vector_store.model_name
                and metadata.get("embedding_dimension") == embedding_dimension
                and metadata.get("chunk_count") == len(chunks)
                and cached_chunk_ids is not None
                and len(cached_chunk_ids) == len(chunks)
                and len(set(cached_chunk_ids)) == len(cached_chunk_ids)
                and set(cached_chunk_ids) == set(current_chunk_ids)
                and metadata.get("content_fingerprint") == content_fingerprint
                and metadata.get("chunking_fingerprint") == chunking_fingerprint
                and bool(metadata.get("index_storage_path"))
            )

            if cache_is_valid:
                logger.debug("vector_index_cache_validated paper_id=%s valid=true", paper_uuid)
                try:
                    index_bytes = download_vector_index(metadata["index_storage_path"])
                except Exception as error:
                    if not _is_missing_storage_object(error):
                        raise
                    logger.warning("Cached FAISS object for paper %s is missing.", paper_uuid)
                    logger.info(
                        "vector_index_cache_miss paper_id=%s reason=storage_missing",
                        paper_uuid,
                    )
                    invalidate_cache = True
                else:
                    vector_store.reset()
                    store_is_clean = True
                    try:
                        vector_store.load_index(
                            index_bytes=index_bytes,
                            chunks=chunks,
                            chunk_ids=cached_chunk_ids,
                        )
                    except Exception as error:
                        logger.warning(
                            "Cached FAISS index for paper %s is invalid; rebuilding it. Reason: %s",
                            paper_uuid,
                            error,
                        )
                        logger.info(
                            "vector_index_cache_hit paper_id=%s path=%s",
                            paper_uuid,
                            metadata["index_storage_path"],
                        )
                        vector_store.reset()
                        store_is_clean = True
                        invalidate_cache = True
                    else:
                        logger.info("Restored cached FAISS index for paper %s.", paper_uuid)
                        return True
            else:
                logger.info(
                    "vector_index_cache_miss paper_id=%s reason=invalid_metadata",
                    paper_uuid,
                )
                invalidate_cache = True
        else:
            logger.info(
                "vector_index_cache_miss paper_id=%s reason=no_metadata",
                paper_uuid,
            )

        # Remove obsolete metadata before publishing a new immutable object.
        # The old object is left for grace-period cleanup so in-flight readers
        # are never disrupted.
        if invalidate_cache:
            self._discard_metadata_and_object(paper_uuid, active_lease)

        if not store_is_clean:
            vector_store.reset()
        active_lease.assert_active()
        vector_store.add_chunks(chunks)
        index_bytes = vector_store.serialize_index()
        active_lease.assert_active()
        storage_path = upload_vector_index(index_bytes, paper_uuid)
        logger.info("vector_index_uploaded paper_id=%s path=%s", paper_uuid, storage_path)

        try:
            active_lease.assert_active()
            save_paper_vector_index(
                paper_id=paper_uuid,
                index_storage_path=storage_path,
                embedding_model=vector_store.model_name,
                embedding_dimension=embedding_dimension,
                chunk_ids=current_chunk_ids,
                content_fingerprint=content_fingerprint,
                chunking_fingerprint=chunking_fingerprint,
                lease=active_lease.lease,
            )
        except Exception:
            logger.exception(
                "vector_index_publication_failed paper_id=%s path=%s",
                paper_uuid,
                storage_path,
            )
            try:
                delete_vector_index(storage_path)
            except Exception as cleanup_error:
                if not _is_missing_storage_object(cleanup_error):
                    logger.exception(
                        "Failed to clean up FAISS object after metadata persistence failure."
                    )
            raise
        logger.info("Built and persisted FAISS index for paper %s.", paper_uuid)
        return False

    @staticmethod
    def _parse_chunk_ids(value: Any) -> list[int] | None:
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                return None
        if not isinstance(value, list) or any(type(value_) is not int for value_ in value):
            return None
        return value
