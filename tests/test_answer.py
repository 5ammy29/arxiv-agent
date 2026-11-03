from datetime import datetime

from project.nodes.answer import AnswerBuilder
from project.nodes.arxiv import Paper
from project.processing.chunker import TextChunk
from project.nodes.vector_store import SearchResult


def test_answer_builder_includes_answer_source_and_evidence():
    paper = Paper(
        arxiv_id="1234.5678",
        title="Test Paper",
        authors=["Test Author"],
        abstract="Test abstract",
        published=datetime.now(),
        updated=datetime.now(),
        url="https://arxiv.org/abs/1234.5678",
        pdf_url="https://arxiv.org/pdf/1234.5678",
        categories=["cs.AI"],
    )

    results = [
        SearchResult(
            chunk=TextChunk(chunk_id=0, page=3, text="Evidence one"),
            score=0.9,
        ),
        SearchResult(
            chunk=TextChunk(chunk_id=1, page=3, text="Evidence two"),
            score=0.8,
        ),
        SearchResult(
            chunk=TextChunk(chunk_id=2, page=7, text="Evidence three"),
            score=0.7,
        ),
    ]

    builder = AnswerBuilder()
    result = builder.build("Final answer.", paper, results)

    assert "## Answer" in result
    assert "Final answer." in result
    assert "## Evidence" in result
    assert "- Page 3" in result
    assert "- Page 7" in result
    assert result.count("- Page 3") == 1
    assert "## Source" in result
    assert "Test Paper" in result
    assert "arXiv: 1234.5678" in result


def test_answer_builder_handles_no_results():
    paper = Paper(
        arxiv_id="1234.5678",
        title="Test Paper",
        authors=["Test Author"],
        abstract="Test abstract",
        published=datetime.now(),
        updated=datetime.now(),
        url="https://arxiv.org/abs/1234.5678",
        pdf_url="https://arxiv.org/pdf/1234.5678",
        categories=["cs.AI"],
    )

    builder = AnswerBuilder()
    result = builder.build("Final answer.", paper, [])

    assert "## Answer" in result
    assert "Final answer." in result
    assert "## Evidence" in result
    assert "## Source" in result
    assert "Test Paper" in result


def test_answer_builder_lists_only_papers_with_evidence():
    selected = Paper(
        arxiv_id="selectedv1", title="Selected", authors=[], abstract="",
        published=datetime.now(), updated=datetime.now(), url="", pdf_url="", categories=[],
    )
    results = [
        SearchResult(TextChunk(4, 2, "first"), 0.9, paper_id="a", arxiv_id="firstv2", paper_title="First"),
        SearchResult(TextChunk(8, 6, "second"), 0.8, paper_id="b", arxiv_id="secondv1", paper_title="Second"),
    ]

    result = AnswerBuilder().build("Comparison.", selected, results)

    assert "First — arXiv: firstv2; pages 2" in result
    assert "Second — arXiv: secondv1; pages 6" in result
    assert "Selected — arXiv" not in result


def test_answer_builder_supports_multiple_papers_without_selected_paper():
    results = [
        SearchResult(
            chunk=TextChunk(chunk_id=1, page=3, text="Evidence from paper A"),
            score=0.9,
            paper_id="paper-a",
            arxiv_id="1706.03762",
            paper_title="Attention Is All You Need",
        ),
        SearchResult(
            chunk=TextChunk(chunk_id=2, page=5, text="Evidence from paper B"),
            score=0.8,
            paper_id="paper-b",
            arxiv_id="1810.04805",
            paper_title="BERT",
        ),
    ]

    result = AnswerBuilder().build(
        "Both papers use Transformer architecture in different ways.",
        None,
        results,
    )

    assert "Attention Is All You Need" in result
    assert "arXiv: 1706.03762" in result
    assert "BERT" in result
    assert "arXiv: 1810.04805" in result
    assert "## Evidence" in result
    assert "## Source" in result

    assert "Page 3 — Attention Is All You Need (1706.03762), chunks 1" in result
    assert "Page 5 — BERT (1810.04805), chunks 2" in result