from project.nodes.chunker import TextChunk
from project.nodes.retriever import Retriever


class FakeVectorStore:
    def __init__(self):
        self.query = None
        self.top_k = None

    def search(self, query: str, top_k: int = 5) -> list[TextChunk]:
        self.query = query
        self.top_k = top_k

        chunks = []

        chunks.append(TextChunk(chunk_id=0, page=1, text="The model uses a Transformer architecture."))
        chunks.append(TextChunk(chunk_id=1, page=2, text="The dataset contains 50,000 samples."))

        return chunks


def test_retrieve_returns_chunks():
    vector_store = FakeVectorStore()
    retriever = Retriever(vector_store)

    results = retriever.retrieve("What architecture does the model use?")

    assert len(results) == 2
    assert results[0].chunk_id == 0
    assert results[1].chunk_id == 1


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
