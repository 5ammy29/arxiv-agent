from sentence_transformers import CrossEncoder
from project.nodes.vector_store import SearchResult

class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", model=None):
        if model is None:
            model = CrossEncoder(model_name)

        self.model = model

    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        if not results:
            return []

        pairs = []

        for result in results:
            pairs.append((query, result.chunk.text))

        scores = self.model.predict(pairs)

        reranked_results = []

        for result, score in zip(results, scores):
            reranked_results.append(SearchResult(chunk=result.chunk, score=result.score, rerank_score=float(score)))

        reranked_results.sort(key=lambda result: result.rerank_score, reverse=True)

        return reranked_results