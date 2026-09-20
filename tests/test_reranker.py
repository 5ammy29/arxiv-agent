from project.nodes.chunker import TextChunk
from project.nodes.reranker import Reranker
from project.nodes.vector_store import SearchResult

class FakeCrossEncoder:
    def predict(self, pairs):
        scores = []

        for query, text in pairs:
            if "Transformer" in text:
                scores.append(0.2)
            else:
                scores.append(0.9)

        return scores

def create_results():
    chunks = [
        TextChunk(
            chunk_id=0,
            page=1,
            text="The model uses a Transformer architecture.",
        ),
        TextChunk(
            chunk_id=1,
            page=2,
            text="The experiment uses a medical image dataset.",
        ),
    ]

    return [
        SearchResult(chunk=chunks[0], score=0.91),
        SearchResult(chunk=chunks[1], score=0.72),
    ]

def test_reranker_returns_empty_results():
    reranker = Reranker(model=FakeCrossEncoder())

    results = reranker.rerank("What is the model?", [])

    assert results == []

def test_reranker_ranks_results():
    reranker = Reranker(model=FakeCrossEncoder())
    results = create_results()

    reranked = reranker.rerank("What is discussed?", results)

    assert len(reranked) == 2
    assert reranked[0].chunk.chunk_id == 1
    assert reranked[1].chunk.chunk_id == 0

def test_reranker_preserves_retrieval_scores():
    reranker = Reranker(model=FakeCrossEncoder())
    results = create_results()

    reranked = reranker.rerank("What is discussed?", results)

    assert reranked[0].score == 0.72
    assert reranked[1].score == 0.91

def test_reranker_adds_rerank_scores():
    reranker = Reranker(model=FakeCrossEncoder())
    results = create_results()

    reranked = reranker.rerank("What is discussed?", results)

    assert reranked[0].rerank_score == 0.9
    assert reranked[1].rerank_score == 0.2