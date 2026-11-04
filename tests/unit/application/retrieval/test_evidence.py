from project.application.retrieval.evidence import (
    chunk_from_record,
    chunk_to_record,
    result_from_record,
    result_to_record,
)
from project.nodes.vector_store import SearchResult
from project.processing.chunker import TextChunk


def test_chunk_serialization_preserves_metadata():
    chunk = TextChunk(
        chunk_id=42,
        page=7,
        text="Evidence from the paper.",
    )

    record = chunk_to_record(chunk)
    restored = chunk_from_record(record)

    assert record == {
        "chunk_id": 42,
        "page": 7,
        "text": "Evidence from the paper.",
    }
    assert restored == chunk


def test_search_result_serialization_preserves_all_evidence_metadata():
    result = SearchResult(
        chunk=TextChunk(
            chunk_id=42,
            page=7,
            text="Evidence from the paper.",
        ),
        score=0.91,
        rerank_score=0.95,
        paper_id="paper-uuid-123",
        arxiv_id="1706.03762",
        paper_title="Attention Is All You Need",
    )

    record = result_to_record(result)
    restored = result_from_record(record)

    assert restored == result
    assert restored.paper_id == "paper-uuid-123"
    assert restored.arxiv_id == "1706.03762"
    assert restored.paper_title == "Attention Is All You Need"
    assert restored.chunk.chunk_id == 42
    assert restored.chunk.page == 7


def test_result_from_record_supports_missing_rerank_score():
    record = {
        "chunk": {
            "chunk_id": 3,
            "page": 2,
            "text": "Some evidence.",
        },
        "score": 0.8,
        "paper_id": "paper-uuid-456",
        "arxiv_id": "1810.04805",
        "paper_title": "BERT",
    }

    restored = result_from_record(record)

    assert restored.rerank_score is None
    assert restored.paper_id == "paper-uuid-456"
    assert restored.arxiv_id == "1810.04805"
    assert restored.paper_title == "BERT"
    assert restored.chunk.chunk_id == 3
    assert restored.chunk.page == 2
