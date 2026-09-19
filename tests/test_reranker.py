from project.nodes.chunker import TextChunk
from project.nodes.reranker import Reranker
from project.nodes.vector_store import SearchResult

def test_reranker_returns_results():
    chunk = TextChunk(chunk_id=0, page=1, text="Transformer architecture.")

    results = []
    results.append(SearchResult(chunk=chunk, score=0.9))

    reranker = Reranker()

    reranked = reranker.rerank("What architecture is used?", results)

    assert reranked == results

def test_reranker_returns_empty_results():
    reranker = Reranker()

    results = reranker.rerank("What is the paper about?", [])

    assert results == []
