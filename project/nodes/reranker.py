from project.nodes.vector_store import SearchResult

class Reranker:
    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        return results
