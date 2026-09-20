from project.nodes.query import QueryProcessor
from project.nodes.reranker import Reranker
from project.nodes.vector_store import SearchResult, VectorStore

class Retriever:
    def __init__(
        self,
        vector_store: VectorStore,
        query_processor: QueryProcessor | None = None,
        reranker: Reranker | None = None,
    ):
        self.vector_store = vector_store
        self.query_processor = query_processor or QueryProcessor()
        self.reranker = reranker

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float | None = None,
    ) -> list[SearchResult]:
        processed_query = self.query_processor.process(query)

        if similarity_threshold is not None:
            if similarity_threshold < -1.0 or similarity_threshold > 1.0:
                raise ValueError(
                    "similarity_threshold must be between -1.0 and 1.0"
                )

        results = self.vector_store.search(processed_query, top_k=top_k)

        if similarity_threshold is not None:
            results = [
                result
                for result in results
                if result.score >= similarity_threshold
            ]

        if self.reranker is not None:
            results = self.reranker.rerank(processed_query, results)

        return results