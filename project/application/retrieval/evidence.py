"""Serializable evidence records used at the graph/checkpoint boundary."""

from typing import Any

from project.nodes.vector_store import SearchResult
from project.processing.chunker import TextChunk


def chunk_from_record(chunk: TextChunk | dict[str, Any]) -> TextChunk:
    if isinstance(chunk, TextChunk):
        return chunk
    return TextChunk(
        chunk_id=int(chunk["chunk_id"]),
        page=int(chunk["page"]),
        text=str(chunk["text"]),
    )


def chunk_to_record(chunk: TextChunk) -> dict[str, Any]:
    return {"chunk_id": chunk.chunk_id, "page": chunk.page, "text": chunk.text}


def result_to_record(result: SearchResult) -> dict[str, Any]:
    return {
        "chunk": chunk_to_record(result.chunk),
        "score": result.score,
        "rerank_score": result.rerank_score,
        "paper_id": result.paper_id,
        "arxiv_id": result.arxiv_id,
        "paper_title": result.paper_title,
    }


def result_from_record(result: SearchResult | dict[str, Any]) -> SearchResult:
    if isinstance(result, SearchResult):
        return result
    return SearchResult(
        chunk=chunk_from_record(result["chunk"]),
        score=float(result["score"]),
        rerank_score=(
            float(result["rerank_score"])
            if result.get("rerank_score") is not None
            else None
        ),
        paper_id=result.get("paper_id"),
        arxiv_id=result.get("arxiv_id"),
        paper_title=result.get("paper_title"),
    )
