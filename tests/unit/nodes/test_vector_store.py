import faiss
import numpy as np
import pytest

from project.nodes.vector_store import VectorStore
from project.processing.chunker import TextChunk


@pytest.fixture
def vector_store(monkeypatch):
    """Create a VectorStore without downloading a real embedding model."""

    class FakeModel:
        def get_sentence_embedding_dimension(self):
            return 3

    monkeypatch.setattr(
        "project.nodes.vector_store.SentenceTransformer",
        lambda model_name: FakeModel(),
    )
    return VectorStore()


@pytest.fixture
def chunks():
    return [
        TextChunk(chunk_id=0, page=1, text="First chunk"),
        TextChunk(chunk_id=1, page=1, text="Second chunk"),
    ]


def make_index():
    """Create a small FAISS index for testing."""
    index = faiss.IndexFlatIP(3)
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    index.add(embeddings)
    return index


def test_serialize_index_returns_bytes(vector_store):
    vector_store.index = make_index()

    result = vector_store.serialize_index()

    assert isinstance(result, bytes)
    assert result


def test_serialize_empty_index_raises_error(vector_store):
    with pytest.raises(ValueError, match="empty vector index"):
        vector_store.serialize_index()


def test_load_index_restores_index_and_chunk_mapping(vector_store, chunks):
    serialized_index = faiss.serialize_index(make_index()).tobytes()
    chunk_ids = [1, 0]

    vector_store.load_index(serialized_index, chunks, chunk_ids)

    assert vector_store.index is not None
    assert vector_store.index.ntotal == 2
    assert [chunk.chunk_id for chunk in vector_store.chunks] == [1, 0]


@pytest.mark.parametrize(
    ("index_bytes", "chunks", "chunk_ids", "error"),
    [
        (b"", [], [], "empty"),
        (b"some bytes", [], [], "Chunk IDs must not be empty"),
    ],
)
def test_load_index_rejects_invalid_input(
    vector_store, index_bytes, chunks, chunk_ids, error
):
    with pytest.raises(ValueError, match=error):
        vector_store.load_index(index_bytes, chunks, chunk_ids)


def test_load_index_rejects_duplicate_chunk_ids(vector_store, chunks):
    serialized_index = faiss.serialize_index(make_index()).tobytes()

    with pytest.raises(ValueError, match="unique"):
        vector_store.load_index(serialized_index, chunks, [0, 0])


def test_load_index_rejects_mismatched_chunk_ids(vector_store, chunks):
    serialized_index = faiss.serialize_index(make_index()).tobytes()

    with pytest.raises(ValueError, match="do not match"):
        vector_store.load_index(serialized_index, chunks, [0, 2])


def test_load_index_rejects_index_size_mismatch(vector_store, chunks):
    one_vector_index = faiss.IndexFlatIP(3)
    one_vector_index.add(
        np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    )
    serialized_index = faiss.serialize_index(one_vector_index).tobytes()

    with pytest.raises(ValueError, match="index size"):
        vector_store.load_index(serialized_index, chunks, [0, 1])


def test_load_index_rejects_embedding_dimension_mismatch(vector_store, chunks):
    wrong_dimension_index = faiss.IndexFlatIP(2)
    wrong_dimension_index.add(
        np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )
    )
    serialized_index = faiss.serialize_index(wrong_dimension_index).tobytes()

    with pytest.raises(ValueError, match="dimension"):
        vector_store.load_index(serialized_index, chunks, [0, 1])
