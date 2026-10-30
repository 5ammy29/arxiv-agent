import json
import logging
from typing import Any, Protocol
from uuid import UUID

from project.infrastructure.database.vector_indexes import (
    get_paper_vector_index,
    save_paper_vector_index,
)
from project.infrastructure.storage.vector_indexes import (
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
        self,
        index_bytes: bytes,
        chunks: list[TextChunk],
        chunk_ids: list[int],
    ) -> None: ...


class VectorIndexCache:
    def get_or_build_index(
        self,
        paper_id: str | UUID,
        chunks: list[TextChunk],
        vector_store: VectorStoreProtocol,
    ) -> bool:
        if not chunks:
            raise ValueError("Cannot cache an index without chunks.")

        paper_uuid = UUID(str(paper_id))
        current_chunk_ids = [chunk.chunk_id for chunk in chunks]

        if len(set(current_chunk_ids)) != len(current_chunk_ids):
            raise ValueError("Current chunks contain duplicate chunk IDs.")

        embedding_dimension = (
            vector_store.model.get_sentence_embedding_dimension()
        )

        if embedding_dimension is None or embedding_dimension <= 0:
            raise ValueError("Embedding model has an invalid dimension.")

        metadata = get_paper_vector_index(paper_uuid)
        store_is_clean = False

        if metadata is not None:
            cached_chunk_ids = self._parse_chunk_ids(
                metadata.get("chunk_ids")
            )

            cache_is_valid = (
                metadata.get("embedding_model") == vector_store.model_name
                and metadata.get("embedding_dimension")
                == embedding_dimension
                and metadata.get("chunk_count") == len(chunks)
                and cached_chunk_ids is not None
                and len(cached_chunk_ids) == len(chunks)
                and len(set(cached_chunk_ids)) == len(cached_chunk_ids)
                and set(cached_chunk_ids) == set(current_chunk_ids)
                and bool(metadata.get("index_storage_path"))
            )

            if cache_is_valid:
                index_bytes = download_vector_index(
                    metadata["index_storage_path"]
                )

                vector_store.reset()
                store_is_clean = True

                try:
                    vector_store.load_index(
                        index_bytes=index_bytes,
                        chunks=chunks,
                        chunk_ids=cached_chunk_ids,
                    )
                except Exception as exc:
                    logger.warning(
                        "Cached FAISS index for paper %s is invalid; "
                        "rebuilding it. Reason: %s",
                        paper_uuid,
                        exc,
                    )

                    # Clear any partially restored index.
                    vector_store.reset()
                    store_is_clean = True
                else:
                    logger.info(
                        "Restored cached FAISS index for paper %s.",
                        paper_uuid,
                    )
                    return True
            else:
                logger.info(
                    "Cached FAISS metadata for paper %s is outdated; "
                    "rebuilding the index.",
                    paper_uuid,
                )

        # Reset only if the store has not already been cleared.
        if not store_is_clean:
            vector_store.reset()

        vector_store.add_chunks(chunks)

        index_bytes = vector_store.serialize_index()
        storage_path = upload_vector_index(index_bytes, paper_uuid)

        save_paper_vector_index(
            paper_id=paper_uuid,
            index_storage_path=storage_path,
            embedding_model=vector_store.model_name,
            embedding_dimension=embedding_dimension,
            chunk_ids=current_chunk_ids,
        )

        logger.info(
            "Built and persisted FAISS index for paper %s.",
            paper_uuid,
        )

        return False

    @staticmethod
    def _parse_chunk_ids(value: Any) -> list[int] | None:
        """Normalize JSONB chunk IDs into a list of integers."""
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                return None

        if not isinstance(value, list):
            return None

        if any(type(chunk_id) is not int for chunk_id in value):
            return None

        return value
