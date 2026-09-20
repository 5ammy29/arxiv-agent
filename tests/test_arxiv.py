import pytest
from datetime import datetime
from unittest.mock import MagicMock
from project.nodes.arxiv import ArxivClient, ArxivError, Paper

def create_mock_result():
    result = MagicMock()

    result.get_short_id.return_value = "2401.12345"
    result.title = "Test Paper"
    result.authors = [MagicMock(name="Alice"), MagicMock(name="Bob")]
    result.summary = "This is a test abstract."
    result.published = datetime(2024, 1, 15)
    result.updated = datetime(2024, 1, 20)
    result.entry_id = "https://arxiv.org/abs/2401.12345"
    result.pdf_url = "https://arxiv.org/pdf/2401.12345"
    result.categories = ["cs.AI", "cs.LG"]

    result.authors[0].name = "Alice"
    result.authors[1].name = "Bob"

    return result

def test_search_by_topic_returns_papers():
    client = ArxivClient()
    mock_result = create_mock_result()

    client.client.results = MagicMock(return_value=[mock_result])

    papers = client.search_by_topic("machine learning")

    assert len(papers) == 1
    assert isinstance(papers[0], Paper)
    assert papers[0].arxiv_id == "2401.12345"
    assert papers[0].title == "Test Paper"

def test_search_by_topic_rejects_empty_query():
    client = ArxivClient()

    with pytest.raises(ValueError, match="search query cannot be empty"):
        client.search_by_topic("   ")

def test_search_by_topic_respects_max_results():
    client = ArxivClient()
    mock_results = [
        create_mock_result(),
        create_mock_result(),
        create_mock_result(),
    ]

    client.client.results = MagicMock(return_value=mock_results)

    papers = client.search_by_topic("machine learning", max_results=3)

    assert len(papers) == 3

def test_search_by_id_returns_paper():
    client = ArxivClient()
    mock_result = create_mock_result()

    client.client.results = MagicMock(return_value=[mock_result])

    paper = client.search_by_id("2401.12345")

    assert isinstance(paper, Paper)
    assert paper.arxiv_id == "2401.12345"
    assert paper.title == "Test Paper"

def test_search_by_id_returns_none_when_not_found():
    client = ArxivClient()

    client.client.results = MagicMock(return_value=[])

    paper = client.search_by_id("9999.99999")

    assert paper is None

def test_search_by_id_accepts_abs_url():
    client = ArxivClient()
    mock_result = create_mock_result()

    client.client.results = MagicMock(return_value=[mock_result])

    paper = client.search_by_id("https://arxiv.org/abs/2401.12345")

    assert paper is not None
    assert paper.arxiv_id == "2401.12345"

def test_search_by_id_accepts_pdf_url():
    client = ArxivClient()
    mock_result = create_mock_result()

    client.client.results = MagicMock(return_value=[mock_result])

    paper = client.search_by_id("https://arxiv.org/pdf/2401.12345.pdf")

    assert paper is not None
    assert paper.arxiv_id == "2401.12345"

def test_extract_arxiv_id():
    assert ArxivClient._extract_arxiv_id("2401.12345") == "2401.12345"
    assert (
        ArxivClient._extract_arxiv_id("https://arxiv.org/abs/2401.12345")
        == "2401.12345"
    )
    assert (
        ArxivClient._extract_arxiv_id("https://arxiv.org/pdf/2401.12345.pdf")
        == "2401.12345"
    )

def test_normalize_result():
    client = ArxivClient()
    mock_result = create_mock_result()

    paper = client._normalize_result(mock_result)

    assert paper.arxiv_id == "2401.12345"
    assert paper.title == "Test Paper"
    assert paper.authors == ["Alice", "Bob"]
    assert paper.abstract == "This is a test abstract."
    assert paper.published == datetime(2024, 1, 15)
    assert paper.updated == datetime(2024, 1, 20)
    assert paper.url == "https://arxiv.org/abs/2401.12345"
    assert paper.pdf_url == "https://arxiv.org/pdf/2401.12345"
    assert paper.categories == ["cs.AI", "cs.LG"]

def test_search_by_topic_raises_arxiv_error():
    class FakeClient:
        def results(self, search):
            raise RuntimeError("network failure")

    client = ArxivClient()
    client.client = FakeClient()

    try:
        client.search_by_topic("transformers")
    except ArxivError as error:
        assert str(error) == "Failed to search arXiv"
    else:
        raise AssertionError("Expected ArxivError")