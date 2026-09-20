from datetime import datetime
from project.nodes.answer import AnswerBuilder
from project.nodes.arxiv import Paper
from project.nodes.chunker import TextChunk
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
        SearchResult(chunk=TextChunk(chunk_id=0, page=3, text="Evidence one"), score=0.9),
        SearchResult(chunk=TextChunk(chunk_id=1, page=3, text="Evidence two"), score=0.8),
        SearchResult(chunk=TextChunk(chunk_id=2, page=7, text="Evidence three"), score=0.7),
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