from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock
from unittest.mock import patch
from uuid import uuid4

import psycopg
import pytest

from project.application.papers.processing_lock import (
    PaperProcessingLockError,
    PaperProcessingLockManager,
)
from project.application.retrieval.vector_index_cache import VectorIndexCache
from project.infrastructure.database.paper_locks import (
    PaperProcessingLeaseLostError,
    acquire_paper_processing_lock,
    release_paper_processing_lock,
    renew_paper_processing_lock,
)
from project.infrastructure.database.papers import save_paper_artifact
from project.infrastructure.database.vector_indexes import (
    PaperContentChangedError,
    get_paper_vector_index,
    save_paper_vector_index,
)
from project.processing.chunker import TextChunk


pytestmark = pytest.mark.integration

CONTENT_FINGERPRINT = "content-v1"
CHUNKING_FINGERPRINT = "chunking-v1"


def _expire_lease(database_url: str, paper_id) -> None:
    with psycopg.connect(database_url) as connection:
        connection.execute(
            """
            UPDATE paper_processing_locks
            SET lease_expires_at = acquired_at + INTERVAL '1 microsecond'
            WHERE paper_id = %s
            """,
            (paper_id,),
        )


def _publish(paper_id, lease, storage_path, content=CONTENT_FINGERPRINT):
    return save_paper_vector_index(
        paper_id=paper_id,
        index_storage_path=storage_path,
        embedding_model="integration-model",
        embedding_dimension=3,
        chunk_ids=[0],
        content_fingerprint=content,
        chunking_fingerprint=CHUNKING_FINGERPRINT,
        lease=lease,
    )


def test_stale_worker_cannot_overwrite_newer_index_metadata(
    integration_database_url,
    integration_paper,
):
    paper_id, _ = integration_paper()

    first = acquire_paper_processing_lock(paper_id, uuid4(), 60)
    assert first is not None

    save_paper_artifact(
        paper_id,
        f"papers/test/{uuid4().hex}.pdf",
        CONTENT_FINGERPRINT,
        CHUNKING_FINGERPRINT,
        lease=first,
    )

    first_path = f"vector_indexes/{paper_id}/{uuid4().hex}.faiss"

    _expire_lease(integration_database_url, paper_id)

    second = acquire_paper_processing_lock(paper_id, uuid4(), 60)
    assert second is not None

    second_path = f"vector_indexes/{paper_id}/{uuid4().hex}.faiss"
    _publish(paper_id, second, second_path)

    with pytest.raises(PaperProcessingLeaseLostError):
        _publish(paper_id, first, first_path)

    metadata = get_paper_vector_index(paper_id)

    assert metadata is not None
    assert metadata["index_storage_path"] == second_path
    assert first_path != second_path


def test_content_change_during_index_build_rejects_stale_publication(
    integration_database_url,
    integration_paper,
):
    paper_id, _ = integration_paper()

    lease = acquire_paper_processing_lock(paper_id, uuid4(), 60)
    assert lease is not None

    save_paper_artifact(
        paper_id,
        f"papers/test/{uuid4().hex}.pdf",
        CONTENT_FINGERPRINT,
        CHUNKING_FINGERPRINT,
        lease=lease,
    )

    with psycopg.connect(integration_database_url) as connection:
        connection.execute(
            """
            UPDATE paper_artifacts
            SET content_fingerprint = 'content-v2'
            WHERE paper_id = %s
            """,
            (paper_id,),
        )

    with pytest.raises(PaperContentChangedError):
        _publish(
            paper_id,
            lease,
            f"vector_indexes/{paper_id}/{uuid4().hex}.faiss",
        )


def test_only_current_lease_can_publish_metadata_for_unique_paths(
    integration_paper,
):
    paper_id, _ = integration_paper()

    lease = acquire_paper_processing_lock(paper_id, uuid4(), 60)
    assert lease is not None

    save_paper_artifact(
        paper_id,
        f"papers/test/{uuid4().hex}.pdf",
        CONTENT_FINGERPRINT,
        CHUNKING_FINGERPRINT,
        lease=lease,
    )

    storage_path = f"vector_indexes/{paper_id}/{uuid4().hex}.faiss"

    _publish(paper_id, lease, storage_path)

    metadata = get_paper_vector_index(paper_id)

    assert metadata is not None
    assert metadata["index_storage_path"] == storage_path
    assert metadata["content_fingerprint"] == CONTENT_FINGERPRINT


def test_two_workers_build_same_paper_index_only_once(
    integration_database_url,
    integration_paper,
):
    paper_id, _ = integration_paper()

    content_fingerprint = "concurrent-content-v1"
    chunking_fingerprint = "concurrent-chunking-v1"

    chunks = [
        TextChunk(
            chunk_id=0,
            page=1,
            text="Transformers use attention mechanisms.",
        ),
        TextChunk(
            chunk_id=1,
            page=2,
            text="FAISS provides vector similarity search.",
        ),
    ]

    seed_manager = PaperProcessingLockManager()

    with seed_manager.acquire(paper_id) as active_lease:
        save_paper_artifact(
            paper_id,
            f"papers/test/{uuid4().hex}.pdf",
            content_fingerprint,
            chunking_fingerprint,
            lease=active_lease.lease,
        )

    build_started = Event()
    release_build = Event()
    second_worker_contended = Event()

    state_lock = Lock()
    build_count = 0
    stored_objects = {}

    storage_path = f"vector_indexes/{paper_id}/concurrent.faiss"

    class FakeEmbeddingModel:
        def get_sentence_embedding_dimension(self):
            return 3

    class FakeVectorStore:
        def __init__(self):
            self.model_name = "integration-model"
            self.model = FakeEmbeddingModel()
            self.loaded_index = None

        def reset(self):
            self.loaded_index = None

        def add_chunks(self, requested_chunks):
            nonlocal build_count

            with state_lock:
                build_count += 1
                current_build = build_count

            if current_build == 1:
                build_started.set()
                assert release_build.wait(timeout=10)

        def serialize_index(self):
            return b"fake-faiss-index"

        def load_index(self, index_bytes, chunks, chunk_ids):
            assert index_bytes == b"fake-faiss-index"
            self.loaded_index = list(chunk_ids)

    def upload(index_bytes, requested_paper_id):
        assert requested_paper_id == paper_id

        with state_lock:
            stored_objects[storage_path] = index_bytes

        return storage_path

    def download(path):
        with state_lock:
            return stored_objects[path]

    lock_manager = PaperProcessingLockManager(
        wait_timeout=10,
        lease_duration=10,
        heartbeat_interval=2,
        retry_interval=0.01,
    )

    cache = VectorIndexCache(lock_manager=lock_manager)

    original_acquire = acquire_paper_processing_lock

    def track_contention(*args, **kwargs):
        result = original_acquire(*args, **kwargs)

        if result is None:
            second_worker_contended.set()

        return result

    def run_worker():
        store = FakeVectorStore()

        restored = cache.get_or_build_index(
            paper_id,
            chunks,
            store,
            content_fingerprint=content_fingerprint,
            chunking_fingerprint=chunking_fingerprint,
        )

        return restored, store

    try:
        with (
            patch(
                "project.application.retrieval.vector_index_cache.upload_vector_index",
                side_effect=upload,
            ),
            patch(
                "project.application.retrieval.vector_index_cache.download_vector_index",
                side_effect=download,
            ),
            patch(
                "project.application.papers.processing_lock.acquire_paper_processing_lock",
                side_effect=track_contention,
            ),
            ThreadPoolExecutor(max_workers=2) as executor,
        ):
            worker_a = executor.submit(run_worker)
            worker_b = None

            try:
                assert build_started.wait(timeout=5)

                worker_b = executor.submit(run_worker)

                assert second_worker_contended.wait(timeout=5)

            finally:
                release_build.set()

            result_a = worker_a.result(timeout=10)
            assert worker_b is not None
            result_b = worker_b.result(timeout=10)

    finally:
        release_build.set()

    results = [result_a, result_b]

    restored_results = [result[0] for result in results]
    stores = [result[1] for result in results]

    assert sorted(restored_results) == [False, True]
    assert build_count == 1
    assert len(stored_objects) == 1

    restored_stores = [
        store
        for restored, store in zip(restored_results, stores)
        if restored
    ]

    assert len(restored_stores) == 1
    assert restored_stores[0].loaded_index == [0, 1]

    metadata = get_paper_vector_index(paper_id)

    assert metadata is not None
    assert metadata["index_storage_path"] == storage_path
    assert metadata["content_fingerprint"] == content_fingerprint
    assert metadata["chunking_fingerprint"] == chunking_fingerprint
    assert metadata["chunk_count"] == len(chunks)


def test_worker_losing_lease_during_index_build_cannot_upload_or_publish(
    integration_database_url,
    integration_paper,
):
    paper_id, _ = integration_paper()

    content_fingerprint = "lease-loss-content-v1"
    chunking_fingerprint = "lease-loss-chunking-v1"

    chunks = [
        TextChunk(
            chunk_id=0,
            page=1,
            text="Transformers use attention mechanisms.",
        )
    ]

    seed_manager = PaperProcessingLockManager()

    with seed_manager.acquire(paper_id) as active_lease:
        save_paper_artifact(
            paper_id,
            f"papers/test/{uuid4().hex}.pdf",
            content_fingerprint,
            chunking_fingerprint,
            lease=active_lease.lease,
        )

    build_started = Event()
    release_build = Event()
    renewal_rejected = Event()

    class FakeEmbeddingModel:
        def get_sentence_embedding_dimension(self):
            return 3

    class FakeVectorStore:
        def __init__(self):
            self.model_name = "integration-model"
            self.model = FakeEmbeddingModel()

        def reset(self):
            pass

        def add_chunks(self, requested_chunks):
            build_started.set()
            assert release_build.wait(timeout=10)

        def serialize_index(self):
            return b"fake-faiss-index"

        def load_index(self, index_bytes, chunks, chunk_ids):
            raise AssertionError("No cached index should be loaded.")

    lock_manager = PaperProcessingLockManager(
        wait_timeout=10,
        lease_duration=10,
        heartbeat_interval=1,
        retry_interval=0.01,
    )

    cache = VectorIndexCache(lock_manager=lock_manager)
    store = FakeVectorStore()

    original_renew = renew_paper_processing_lock

    def track_renewal(lease, lease_duration_seconds=300):
        renewed = original_renew(lease, lease_duration_seconds)

        if not renewed:
            renewal_rejected.set()

        return renewed

    replacement_lease = None

    try:
        with (
            patch(
                "project.application.papers.processing_lock.renew_paper_processing_lock",
                side_effect=track_renewal,
            ),
            patch(
                "project.application.retrieval.vector_index_cache.upload_vector_index"
            ) as mock_upload,
        ):
            with ThreadPoolExecutor(max_workers=1) as executor:
                worker = executor.submit(
                    cache.get_or_build_index,
                    paper_id,
                    chunks,
                    store,
                    content_fingerprint=content_fingerprint,
                    chunking_fingerprint=chunking_fingerprint,
                )

                try:
                    assert build_started.wait(timeout=5)

                    _expire_lease(integration_database_url, paper_id)

                    replacement_lease = acquire_paper_processing_lock(
                        paper_id,
                        uuid4(),
                        60,
                    )
                    assert replacement_lease is not None

                    assert renewal_rejected.wait(timeout=5)

                finally:
                    release_build.set()

                with pytest.raises(PaperProcessingLockError):
                    worker.result(timeout=10)

            mock_upload.assert_not_called()

        assert get_paper_vector_index(paper_id) is None

    finally:
        release_build.set()

        if replacement_lease is not None:
            release_paper_processing_lock(replacement_lease)
