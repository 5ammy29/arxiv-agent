import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from project.nodes.chunker import TextChunk

class VectorStore:
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks: list[TextChunk] = []

    def add_chunks(self, chunks: list[TextChunk]) -> None:
        if not chunks:
            return

        texts = []

        for chunk in chunks:
            texts.append(chunk.text)

        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

        embeddings = embeddings.astype(np.float32)

        if self.index is None:
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)

        self.index.add(embeddings)

        for chunk in chunks:
            self.chunks.append(chunk)

    def search(self, query: str, top_k: int = 5) -> list[TextChunk]:
        if self.index is None or not self.chunks:
            return []

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        query_embedding = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)

        query_embedding = query_embedding.astype(np.float32)

        top_k = min(top_k, len(self.chunks))

        distances, indices = self.index.search(query_embedding, top_k)

        results = []

        for index in indices[0]:
            if index != -1:
                results.append(self.chunks[index])

        return results
