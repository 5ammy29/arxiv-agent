from unittest.mock import patch
from uuid import uuid4

import faiss
import numpy as np
import pytest

from project.application.retrieval.vector_index_cache import (
    VectorIndexCache,
    _fallback_content_fingerprint,
)
from project.processing.chunker import TextChunk


class FakeEmbeddingModel:

    def get_sentence_embedding_dimension(self):
        return 3


class FakeVectorStore:

    def __init__(self):
        self.model_name = "test-embedding-model"
        self.model = FakeEmbeddingModel()
        self.index = faiss.IndexFlatIP(3)
        self.chunks = []

    def reset(self):
        self.index = faiss.IndexFlatIP(3)
        self.chunks = []

    def add_chunks(self, chunks):
        self.chunks = list(chunks)

        embeddings = np.array(
            [
                self._embed(chunk.text)
                for chunk in chunks
            ],
            dtype=np.float32,
        )

        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

    def serialize_index(self):
        return faiss.serialize_index(self.index).tobytes()

    def load_index(self, index_bytes, chunks, chunk_ids):
        index_array = np.frombuffer(index_bytes, dtype=np.uint8)
        self.index = faiss.deserialize_index(index_array)

        chunk_map = {chunk.chunk_id: chunk for chunk in chunks}
        self.chunks = [chunk_map[chunk_id] for chunk_id in chunk_ids]

        if self.index.ntotal != len(self.chunks):
            raise ValueError("Index size does not match chunk count.")

    def search(self, query, top_k=5):
        query_embedding = np.array(
            [self._embed(query)],
            dtype=np.float32,
        )
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(
            query_embedding,
            min(top_k, self.index.ntotal),
        )

        return [
            (self.chunks[index], float(score))
            for score, index in zip(scores[0], indices[0])
            if index >= 0
        ]

    @staticmethod
    def _embed(text):
        value = sum(ord(char) for char in text)
        return [
            float(value % 97 + 1),
            float(value % 53 + 1),
            float(value % 29 + 1),
        ]


@pytest.fixture
def chunks():
    return [
        TextChunk(chunk_id=0, page=1, text="Transformers use attention."),
        TextChunk(chunk_id=1, page=2, text="FAISS supports vector similarity search."),
        TextChunk(chunk_id=2, page=3, text="Embeddings represent text as vectors."),
    ]


def make_metadata(paper_id, chunks, storage_path="vector_indexes/test.faiss"):
    return {
        "paper_id": str(paper_id),
        "index_storage_path": storage_path,
        "embedding_model": "test-embedding-model",
        "embedding_dimension": 3,
        "chunk_ids": [chunk.chunk_id for chunk in chunks],
        "chunk_count": len(chunks),
        "content_fingerprint": _fallback_content_fingerprint(chunks),
        "chunking_fingerprint": "direct-chunks-v1",
    }


@pytest.fixture(autouse=True)
def discard_invalid_cache_metadata():
    with (
        patch("project.application.retrieval.vector_index_cache.delete_paper_vector_index"),
        patch("project.application.retrieval.vector_index_cache.delete_vector_index"),
    ):
        yield


def test_cache_miss_persists_index_and_metadata(chunks):
    paper_id = uuid4()
    vector_store = FakeVectorStore()
    stored_objects = {}
    saved_metadata = {}

    def upload(index_bytes, requested_paper_id):
        path = f"vector_indexes/{requested_paper_id}.faiss"
        stored_objects[path] = index_bytes
        return path

    def save_metadata(**kwargs):
        saved_metadata.update(kwargs)

    with (
        patch(
            "project.application.retrieval.vector_index_cache.get_paper_vector_index",
            return_value=None,
        ),
        patch(
            "project.application.retrieval.vector_index_cache.upload_vector_index",
            side_effect=upload,
        ),
        patch(
            "project.application.retrieval.vector_index_cache.save_paper_vector_index",
            side_effect=save_metadata,
        ),
    ):
        restored = VectorIndexCache().get_or_build_index(
            paper_id,
            chunks,
            vector_store,
        )

    assert restored is False
    assert vector_store.index.ntotal == len(chunks)
    assert len(stored_objects) == 1
    assert saved_metadata["paper_id"] == paper_id
    assert saved_metadata["chunk_ids"] == [0, 1, 2]
    assert len(saved_metadata["chunk_ids"]) == len(chunks)


def test_valid_cache_restores_index_without_rebuilding(chunks):
    paper_id = uuid4()
    original_store = FakeVectorStore()
    original_store.add_chunks(chunks)
    serialized_index = original_store.serialize_index()

    metadata = make_metadata(paper_id, chunks)
    restored_store = FakeVectorStore()

    with (
        patch(
            "project.application.retrieval.vector_index_cache.get_paper_vector_index",
            return_value=metadata,
        ),
        patch(
            "project.application.retrieval.vector_index_cache.download_vector_index",
            return_value=serialized_index,
        ),
        patch(
            "project.application.retrieval.vector_index_cache.upload_vector_index",
        ) as upload,
        patch(
            "project.application.retrieval.vector_index_cache.save_paper_vector_index",
        ) as save_metadata,
    ):
        restored = VectorIndexCache().get_or_build_index(
            paper_id,
            chunks,
            restored_store,
        )

    assert restored is True
    assert restored_store.index.ntotal == len(chunks)
    assert [chunk.chunk_id for chunk in restored_store.chunks] == [0, 1, 2]
    upload.assert_not_called()
    save_metadata.assert_not_called()

    results = restored_store.search("Transformers use attention.")
    assert results
    assert results[0][0].chunk_id == 0


def test_missing_metadata_rebuilds_index(chunks):
    paper_id = uuid4()
    vector_store = FakeVectorStore()

    with (
        patch(
            "project.application.retrieval.vector_index_cache.get_paper_vector_index",
            return_value=None,
        ),
        patch(
            "project.application.retrieval.vector_index_cache.upload_vector_index",
            return_value=f"vector_indexes/{paper_id}.faiss",
        ) as upload,
        patch(
            "project.application.retrieval.vector_index_cache.save_paper_vector_index",
        ) as save_metadata,
    ):
        restored = VectorIndexCache().get_or_build_index(
            paper_id,
            chunks,
            vector_store,
        )

    assert restored is False
    assert vector_store.index.ntotal == len(chunks)
    upload.assert_called_once()
    save_metadata.assert_called_once()


def test_corrupt_cached_index_is_rebuilt(chunks):
    paper_id = uuid4()
    vector_store = FakeVectorStore()
    metadata = make_metadata(paper_id, chunks)

    with (
        patch(
            "project.application.retrieval.vector_index_cache.get_paper_vector_index",
            return_value=metadata,
        ),
        patch(
            "project.application.retrieval.vector_index_cache.download_vector_index",
            return_value=b"not a valid FAISS index",
        ),
        patch(
            "project.application.retrieval.vector_index_cache.upload_vector_index",
            return_value=f"vector_indexes/{paper_id}.faiss",
        ) as upload,
        patch(
            "project.application.retrieval.vector_index_cache.save_paper_vector_index",
        ) as save_metadata,
    ):
        restored = VectorIndexCache().get_or_build_index(
            paper_id,
            chunks,
            vector_store,
        )

    assert restored is False
    assert vector_store.index.ntotal == len(chunks)
    upload.assert_called_once()
    save_metadata.assert_called_once()
