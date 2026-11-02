from contextlib import contextmanager
from threading import Event
from unittest.mock import Mock, patch
from uuid import UUID

import pytest

from project.application.retrieval.vector_index_cache import (
    VectorIndexCache,
    _fallback_content_fingerprint,
)
from project.application.papers.processing_lock import ActivePaperProcessingLease
from project.infrastructure.database.paper_locks import PaperProcessingLease
from project.processing.chunker import TextChunk


PAPER_ID = UUID("12345678-1234-5678-1234-567812345678")
MODEL_NAME = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIMENSION = 768
STORAGE_PATH = f"vector_indexes/{PAPER_ID}/test.faiss"
INDEX_BYTES = b"serialized-faiss-index"


@pytest.fixture
def chunks():
    return [
        TextChunk(chunk_id=0, page=1, text="First paper chunk."),
        TextChunk(chunk_id=1, page=1, text="Second paper chunk."),
    ]


@pytest.fixture
def vector_store():
    store = Mock()
    store.model_name = MODEL_NAME
    store.model.get_sentence_embedding_dimension.return_value = (
        EMBEDDING_DIMENSION
    )
    store.serialize_index.return_value = INDEX_BYTES
    return store


@pytest.fixture
def cache():
    class FakeLockManager:
        @contextmanager
        def acquire(self, paper_id):
            yield ActivePaperProcessingLease(
                PaperProcessingLease(paper_id, UUID(int=1), 1), Event()
            )

    return VectorIndexCache(lock_manager=FakeLockManager())


@pytest.fixture
def valid_metadata(chunks):
    return {
        "paper_id": PAPER_ID,
        "index_storage_path": STORAGE_PATH,
        "embedding_model": MODEL_NAME,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "chunk_ids": [0, 1],
        "chunk_count": 2,
        "content_fingerprint": _fallback_content_fingerprint(chunks),
        "chunking_fingerprint": "direct-chunks-v1",
    }


@pytest.fixture(autouse=True)
def discard_invalid_cache_metadata():
    with (
        patch(
            "project.application.retrieval.vector_index_cache.delete_paper_vector_index"
        ),
        patch(
            "project.application.retrieval.vector_index_cache.delete_vector_index"
        ),
    ):
        yield


def test_cache_miss_builds_and_persists_index(
    cache, chunks, vector_store
):
    with (
        patch(
            "project.application.retrieval.vector_index_cache."
            "get_paper_vector_index",
            return_value=None,
        ),
        patch(
            "project.application.retrieval.vector_index_cache."
            "upload_vector_index",
            return_value=STORAGE_PATH,
        ) as upload,
        patch(
            "project.application.retrieval.vector_index_cache."
            "save_paper_vector_index"
        ) as save,
    ):
        restored = cache.get_or_build_index(
            str(PAPER_ID), chunks, vector_store
        )

    assert restored is False
    vector_store.reset.assert_called_once()
    vector_store.add_chunks.assert_called_once_with(chunks)
    vector_store.serialize_index.assert_called_once_with()
    upload.assert_called_once_with(INDEX_BYTES, PAPER_ID)
    assert save.call_count == 1
    assert {
        key: value
        for key, value in save.call_args.kwargs.items()
        if key != "lease"
    } == {
        "paper_id": PAPER_ID,
        "index_storage_path": STORAGE_PATH,
        "embedding_model": MODEL_NAME,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "chunk_ids": [0, 1],
        "content_fingerprint": _fallback_content_fingerprint(chunks),
        "chunking_fingerprint": "direct-chunks-v1",
    }
    assert save.call_args.kwargs["lease"].paper_id == PAPER_ID


def test_cache_miss_is_observable(cache, chunks, vector_store, caplog):
    caplog.set_level("INFO")
    with (
        patch(
            "project.application.retrieval.vector_index_cache.get_paper_vector_index",
            return_value=None,
        ),
        patch(
            "project.application.retrieval.vector_index_cache.upload_vector_index",
            return_value=STORAGE_PATH,
        ),
        patch(
            "project.application.retrieval.vector_index_cache.save_paper_vector_index"
        ),
    ):
        cache.get_or_build_index(PAPER_ID, chunks, vector_store)

    assert any(
        "vector_index_cache_miss" in record.message
        and "no_metadata" in record.message
        for record in caplog.records
    )


def test_valid_cache_hit_restores_index_without_reembedding(
    cache, chunks, vector_store, valid_metadata
):
    with (
        patch(
            "project.application.retrieval.vector_index_cache."
            "get_paper_vector_index",
            return_value=valid_metadata,
        ),
        patch(
            "project.application.retrieval.vector_index_cache."
            "download_vector_index",
            return_value=INDEX_BYTES,
        ) as download,
        patch(
            "project.application.retrieval.vector_index_cache."
            "upload_vector_index"
        ) as upload,
        patch(
            "project.application.retrieval.vector_index_cache."
            "save_paper_vector_index"
        ) as save,
    ):
        restored = cache.get_or_build_index(
            str(PAPER_ID), chunks, vector_store
        )

    assert restored is True
    download.assert_called_once_with(STORAGE_PATH)
    vector_store.load_index.assert_called_once_with(
        index_bytes=INDEX_BYTES,
        chunks=chunks,
        chunk_ids=[0, 1],
    )
    vector_store.add_chunks.assert_not_called()
    vector_store.serialize_index.assert_not_called()
    upload.assert_not_called()
    save.assert_not_called()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("embedding_model", "some-other-model"),
        ("embedding_dimension", 384),
        ("chunk_count", 3),
        ("chunk_ids", [0, 99]),
        ("content_fingerprint", None),
        ("chunking_fingerprint", None),
        ("content_fingerprint", "stale-content-fingerprint"),
        ("chunking_fingerprint", "stale-chunking-fingerprint"),
    ],
)
def test_invalid_metadata_rebuilds_index(
    cache, chunks, vector_store, valid_metadata, field, value
):
    metadata = {**valid_metadata, field: value}

    with (
        patch(
            "project.application.retrieval.vector_index_cache."
            "get_paper_vector_index",
            return_value=metadata,
        ),
        patch(
            "project.application.retrieval.vector_index_cache."
            "upload_vector_index",
            return_value=STORAGE_PATH,
        ),
        patch(
            "project.application.retrieval.vector_index_cache."
            "save_paper_vector_index"
        ) as save,
    ):
        restored = cache.get_or_build_index(
            str(PAPER_ID), chunks, vector_store
        )

    assert restored is False
    vector_store.add_chunks.assert_called_once_with(chunks)
    vector_store.serialize_index.assert_called_once_with()
    vector_store.load_index.assert_not_called()

    save.assert_called_once()

    saved_metadata = {
        key: value
        for key, value in save.call_args.kwargs.items()
        if key != "lease"
    }

    assert saved_metadata["content_fingerprint"] == (
        _fallback_content_fingerprint(chunks)
    )
    assert saved_metadata["chunking_fingerprint"] == "direct-chunks-v1"


def test_corrupt_index_rebuilds_and_updates_metadata(
    cache, chunks, vector_store, valid_metadata
):
    vector_store.load_index.side_effect = ValueError(
        "Corrupt serialized index"
    )

    with (
        patch(
            "project.application.retrieval.vector_index_cache."
            "get_paper_vector_index",
            return_value=valid_metadata,
        ),
        patch(
            "project.application.retrieval.vector_index_cache."
            "download_vector_index",
            return_value=INDEX_BYTES,
        ),
        patch(
            "project.application.retrieval.vector_index_cache."
            "upload_vector_index",
            return_value=STORAGE_PATH,
        ) as upload,
        patch(
            "project.application.retrieval.vector_index_cache."
            "save_paper_vector_index"
        ) as save,
    ):
        restored = cache.get_or_build_index(
            str(PAPER_ID), chunks, vector_store
        )

    assert restored is False
    assert vector_store.reset.call_count == 2
    vector_store.add_chunks.assert_called_once_with(chunks)
    upload.assert_called_once_with(INDEX_BYTES, PAPER_ID)
    save.assert_called_once()


def test_database_lookup_failure_propagates(
    cache, chunks, vector_store
):
    with patch(
        "project.application.retrieval.vector_index_cache."
        "get_paper_vector_index",
        side_effect=RuntimeError("Database unavailable"),
    ):
        with pytest.raises(RuntimeError, match="Database unavailable"):
            cache.get_or_build_index(
                str(PAPER_ID), chunks, vector_store
            )

    vector_store.add_chunks.assert_not_called()


def test_storage_download_failure_propagates(
    cache, chunks, vector_store, valid_metadata
):
    with (
        patch(
            "project.application.retrieval.vector_index_cache."
            "get_paper_vector_index",
            return_value=valid_metadata,
        ),
        patch(
            "project.application.retrieval.vector_index_cache."
            "download_vector_index",
            side_effect=RuntimeError("Storage unavailable"),
        ),
    ):
        with pytest.raises(RuntimeError, match="Storage unavailable"):
            cache.get_or_build_index(
                str(PAPER_ID), chunks, vector_store
            )

    vector_store.add_chunks.assert_not_called()


def test_missing_storage_object_rebuilds_index(
    cache, chunks, vector_store, valid_metadata
):
    class MissingObjectError(Exception):
        status_code = 404

    with (
        patch(
            "project.application.retrieval.vector_index_cache."
            "get_paper_vector_index",
            return_value=valid_metadata,
        ),
        patch(
            "project.application.retrieval.vector_index_cache."
            "download_vector_index",
            side_effect=MissingObjectError(),
        ),
        patch(
            "project.application.retrieval.vector_index_cache."
            "upload_vector_index",
            return_value=STORAGE_PATH,
        ),
        patch(
            "project.application.retrieval.vector_index_cache."
            "save_paper_vector_index",
        ) as save,
    ):
        assert cache.get_or_build_index(PAPER_ID, chunks, vector_store) is False

    vector_store.add_chunks.assert_called_once_with(chunks)
    save.assert_called_once()


@pytest.mark.parametrize("paper_id", ["not-a-uuid", ""])
def test_invalid_paper_id_is_rejected(
    cache, chunks, vector_store, paper_id
):
    with pytest.raises(ValueError):
        cache.get_or_build_index(paper_id, chunks, vector_store)


def test_empty_chunks_are_rejected(cache, vector_store):
    with pytest.raises(ValueError, match="without chunks"):
        cache.get_or_build_index(PAPER_ID, [], vector_store)


def test_duplicate_chunk_ids_are_rejected(cache, vector_store):
    duplicate_chunks = [
        TextChunk(chunk_id=1, page=1, text="First."),
        TextChunk(chunk_id=1, page=2, text="Second."),
    ]

    with pytest.raises(ValueError, match="duplicate chunk IDs"):
        cache.get_or_build_index(
            PAPER_ID, duplicate_chunks, vector_store
        )


def test_metadata_persistence_failure_cleans_up_uploaded_index(
    cache, chunks, vector_store
):
    with (
        patch(
            "project.application.retrieval.vector_index_cache.get_paper_vector_index",
            return_value=None,
        ),
        patch(
            "project.application.retrieval.vector_index_cache.upload_vector_index",
            return_value=STORAGE_PATH,
        ),
        patch(
            "project.application.retrieval.vector_index_cache.save_paper_vector_index",
            side_effect=RuntimeError("Database unavailable"),
        ),
        patch(
            "project.application.retrieval.vector_index_cache.delete_vector_index"
        ) as delete,
    ):
        with pytest.raises(RuntimeError, match="Database unavailable"):
            cache.get_or_build_index(PAPER_ID, chunks, vector_store)

    delete.assert_called_once_with(STORAGE_PATH)


def test_embedding_failure_does_not_upload_or_publish_index(
    cache, chunks, vector_store
):
    vector_store.add_chunks.side_effect = RuntimeError("embedding unavailable")

    with (
        patch(
            "project.application.retrieval.vector_index_cache.get_paper_vector_index",
            return_value=None,
        ),
        patch(
            "project.application.retrieval.vector_index_cache.upload_vector_index"
        ) as upload,
        patch(
            "project.application.retrieval.vector_index_cache.save_paper_vector_index"
        ) as save,
    ):
        with pytest.raises(RuntimeError, match="embedding unavailable"):
            cache.get_or_build_index(PAPER_ID, chunks, vector_store)

    upload.assert_not_called()
    save.assert_not_called()


def test_index_storage_upload_failure_does_not_publish_metadata(
    cache, chunks, vector_store
):
    with (
        patch(
            "project.application.retrieval.vector_index_cache.get_paper_vector_index",
            return_value=None,
        ),
        patch(
            "project.application.retrieval.vector_index_cache.upload_vector_index",
            side_effect=RuntimeError("storage unavailable"),
        ),
        patch(
            "project.application.retrieval.vector_index_cache.save_paper_vector_index"
        ) as save,
    ):
        with pytest.raises(RuntimeError, match="storage unavailable"):
            cache.get_or_build_index(PAPER_ID, chunks, vector_store)

    save.assert_not_called()
