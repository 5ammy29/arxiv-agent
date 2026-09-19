from project.nodes.chunker import TextChunk
from project.nodes.vector_store import VectorStore

def create_chunks():
    chunks = []

    chunks.append(TextChunk(chunk_id=0, page=1, text="Transformers use self attention to process sequences."))

    chunks.append(TextChunk(chunk_id=1, page=2, text="The experiment uses a dataset of medical images."))

    chunks.append(TextChunk(chunk_id=2, page=3, text="The model is trained using the Adam optimizer."))

    return chunks

def test_empty_store_returns_no_results():
    store = VectorStore()

    results = store.search("What model is used?")

    assert results == []

def test_add_chunks():
    store = VectorStore()

    chunks = create_chunks()

    store.add_chunks(chunks)

    assert store.index is not None
    assert store.index.ntotal == 3
    assert len(store.chunks) == 3

def test_search_returns_chunks():
    store = VectorStore()

    chunks = create_chunks()

    store.add_chunks(chunks)

    results = store.search("What optimization method is used?", top_k=2)

    assert len(results) == 2

    for result in results:
        assert isinstance(result, TextChunk)


def test_search_returns_most_relevant_chunk():
    store = VectorStore()

    chunks = create_chunks()

    store.add_chunks(chunks)

    results = store.search("What optimizer is used for training?", top_k=1)

    assert len(results) == 1
    assert results[0].chunk_id == 2


def test_top_k_limits_results():
    store = VectorStore()

    chunks = create_chunks()

    store.add_chunks(chunks)

    results = store.search("What does the paper discuss?", top_k=2)

    assert len(results) == 2


def test_top_k_cannot_be_zero():
    store = VectorStore()

    chunks = create_chunks()

    store.add_chunks(chunks)

    try:
        store.search("What is the paper about?", top_k=0)
        assert False
    except ValueError:
        pass


def test_top_k_cannot_be_negative():
    store = VectorStore()

    chunks = create_chunks()

    store.add_chunks(chunks)

    try:
        store.search("What is the paper about?", top_k=-1)
        assert False
    except ValueError:
        pass


def test_top_k_larger_than_chunk_count():
    store = VectorStore()

    chunks = create_chunks()

    store.add_chunks(chunks)

    results = store.search("What is the paper about?", top_k=10)

    assert len(results) == 3


def test_empty_chunks_are_ignored():
    store = VectorStore()

    store.add_chunks([])

    assert store.index is None
    assert store.chunks == []
