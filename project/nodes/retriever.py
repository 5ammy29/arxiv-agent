from project.nodes.chunker import TextChunk
from project.nodes.query import QueryProcessor
from project.nodes.vector_store import VectorStore


class Retriever:
    def __init__(self, vector_store: VectorStore, query_processor: QueryProcessor | None = None):
        self.vector_store = vector_store

        if query_processor is None:
            query_processor = QueryProcessor()

        self.query_processor = query_processor

    def retrieve(self, query: str, top_k: int = 5) -> list[TextChunk]:
        processed_query = self.query_processor.process(query)
        return self.vector_store.search(processed_query, top_k=top_k)
