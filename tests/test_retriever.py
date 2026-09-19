from project.nodes.chunker import TextChunk
from project.nodes.retriever import Retriever
from project.nodes.vector_store import SearchResult

class FakeVectorStore:
    def __init__(self):
        self.query = None
        self.top_k = None

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        self.query = query
        self.top_k = top_k

        chunks = []

        chunks.append(TextChunk(chunk_id=0, page=1, text="The model uses a Transformer architecture."))
        chunks.append(TextChunk(chunk_id=1, page=2, text="The dataset contains 50,000 samples."))

        results = []

        results.append(SearchResult(chunk=chunks[0], score=0.91))
        results.append(SearchResult(chunk=chunks[1], score=0.72))

        return results

def test_retrieve_returns_results():
    vector_store = FakeVectorStore()
    retriever = Retriever(vector_store)

    results = retriever.retrieve("What architecture does the model use?")

    assert len(results) == 2
    assert results[0].chunk.chunk_id == 0
    assert results[1].chunk.chunk_id == 1

def test_retrieve_preserves_scores():
    vector_store = FakeVectorStore()
    retriever = Retriever(vector_store)

    results = retriever.retrieve("What architecture does the model use?")

    assert results[0].score == 0.91
    assert results[1].score == 0.72

def test_retrieve_processes_query():
    vector_store = FakeVectorStore()
    retriever = Retriever(vector_store)

    retriever.retrieve("   What   architecture does the model use?   ")

    assert vector_store.query == "What architecture does the model use?"

def test_retrieve_passes_top_k():
    vector_store = FakeVectorStore()
    retriever = Retriever(vector_store)

    retriever.retrieve("What architecture does the model use?", top_k=3)

    assert vector_store.top_k == 3

def test_retrieve_rejects_empty_query():
    vector_store = FakeVectorStore()
    retriever = Retriever(vector_store)

    try:
        retriever.retrieve("   ")
        assert False
    except ValueError:
        pass

def test_retrieve_filters_by_similarity_threshold():
    vector_store = FakeVectorStore()
    retriever = Retriever(vector_store)

    results = retriever.retrieve("What architecture does the model use?", similarity_threshold=0.8)

    assert len(results) == 1
    assert results[0].score == 0.91

def test_retrieve_includes_result_at_similarity_threshold():
    vector_store = FakeVectorStore()
    retriever = Retriever(vector_store)

    results = retriever.retrieve("What architecture does the model use?", similarity_threshold=0.91)

    assert len(results) == 1
    assert results[0].score == 0.91

def test_retrieve_returns_no_results_below_threshold():
    vector_store = FakeVectorStore()
    retriever = Retriever(vector_store)

    results = retriever.retrieve("What architecture does the model use?", similarity_threshold=0.95)

    assert results == []

def test_retrieve_rejects_invalid_similarity_threshold():
    vector_store = FakeVectorStore()
    retriever = Retriever(vector_store)

    try:
        retriever.retrieve("What architecture does the model use?", similarity_threshold=1.1)
        assert False
    except ValueError:
        pass

def test_retrieve_rejects_negative_invalid_similarity_threshold():
    vector_store = FakeVectorStore()
    retriever = Retriever(vector_store)

    try:
        retriever.retrieve("What architecture does the model use?", similarity_threshold=-1.1)
        assert False
    except ValueError:
        pass
