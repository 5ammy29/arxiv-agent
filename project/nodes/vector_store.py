from dataclasses import dataclass

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from project.processing.chunker import TextChunk


@dataclass
class SearchResult:
    chunk: TextChunk
    score: float
    rerank_score: float | None = None
    paper_id: str | None = None
    arxiv_id: str | None = None
    paper_title: str | None = None


class VectorStore:
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks: list[TextChunk] = []

    def reset(self) -> None:
        self.index = None
        self.chunks = []

    def serialize_index(self) -> bytes:
        if self.index is None:
            raise ValueError("Cannot serialize an empty vector index.")

        return faiss.serialize_index(self.index).tobytes()

    def load_index(
        self,
        index_bytes: bytes,
        chunks: list[TextChunk],
        chunk_ids: list[int],
    ) -> None:
        if not index_bytes:
            raise ValueError("Vector index data must not be empty.")

        if len(chunk_ids) == 0:
            raise ValueError("Chunk IDs must not be empty.")

        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("Chunk IDs must be unique.")

        chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}

        if len(chunks_by_id) != len(chunks):
            raise ValueError("Current chunks contain duplicate chunk IDs.")

        if set(chunk_ids) != set(chunks_by_id):
            raise ValueError(
                "Cached chunk IDs do not match the current paper chunks."
            )

        serialized_array = np.frombuffer(index_bytes, dtype=np.uint8)
        restored_index = faiss.deserialize_index(serialized_array)

        if restored_index.ntotal != len(chunk_ids):
            raise ValueError(
                "Cached index size does not match the stored chunk mapping."
            )

        expected_dimension = self.model.get_sentence_embedding_dimension()

        if restored_index.d != expected_dimension:
            raise ValueError(
                "Cached index dimension does not match the embedding model."
            )

        self.index = restored_index
        self.chunks = [chunks_by_id[chunk_id] for chunk_id in chunk_ids]

    def add_chunks(self, chunks: list[TextChunk]) -> None:
        if not chunks:
            return

        texts = []

        for chunk in chunks:
            texts.append(chunk.text)

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        embeddings = embeddings.astype(np.float32)

        if self.index is None:
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)

        self.index.add(embeddings)

        for chunk in chunks:
            self.chunks.append(chunk)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if self.index is None or not self.chunks:
            return []

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        query_embedding = query_embedding.astype(np.float32)

        top_k = min(top_k, len(self.chunks))
        distances, indices = self.index.search(query_embedding, top_k)

        results = []

        for score, index in zip(distances[0], indices[0]):
            if index != -1:
                results.append(
                    SearchResult(
                        chunk=self.chunks[index],
                        score=float(score),
                    )
                )

        return results
