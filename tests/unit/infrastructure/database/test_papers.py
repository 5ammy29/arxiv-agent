import pytest
from unittest.mock import MagicMock, call, patch
from uuid import uuid4

from project.infrastructure.database.papers import (
    parse_arxiv_id,
    register_paper,
    get_paper,
    update_paper_status,
    add_paper_to_session,
    get_session_papers,
    set_active_paper,
    remove_paper_from_session,
    save_paper_artifact,
    save_paper_content,
)


def test_parse_versioned_arxiv_id():
    assert parse_arxiv_id("1706.03762v7") == ("1706.03762", 7)


def test_parse_arxiv_abs_url():
    assert parse_arxiv_id(
        "https://arxiv.org/abs/1706.03762v7"
    ) == ("1706.03762", 7)


def test_parse_arxiv_pdf_url():
    assert parse_arxiv_id(
        "https://arxiv.org/pdf/1706.03762v7.pdf"
    ) == ("1706.03762", 7)


def test_parse_unversioned_arxiv_id():
    assert parse_arxiv_id("1706.03762") == ("1706.03762", None)


def test_parse_arxiv_id_with_whitespace():
    assert parse_arxiv_id(" 1706.03762v7 ") == ("1706.03762", 7)


def make_paper():
    return {
        "arxiv_id": "1706.03762v7",
        "title": "Attention Is All You Need",
        "authors": ["Author One", "Author Two"],
        "abstract": "A test abstract.",
        "published": "2017-06-12T00:00:00Z",
        "updated": "2017-06-12T00:00:00Z",
        "url": "https://arxiv.org/abs/1706.03762v7",
        "pdf_url": "https://arxiv.org/pdf/1706.03762v7",
        "categories": ["cs.CL", "cs.LG"],
    }


@patch("project.infrastructure.database.papers.get_connection")
def test_register_paper(mock_get_connection):
    paper_id = uuid4()
    row = {
        "id": paper_id,
        "arxiv_id": "1706.03762",
        "version": 7,
        "title": "Attention Is All You Need",
    }

    connection = MagicMock()
    connection.execute.return_value.fetchone.return_value = row
    mock_get_connection.return_value.__enter__.return_value = connection

    result = register_paper(make_paper())

    assert result == row

    query, params = connection.execute.call_args.args
    assert "INSERT INTO papers" in query
    assert "ON CONFLICT (arxiv_id, version)" in query
    assert params[0:3] == (
        "1706.03762",
        7,
        "Attention Is All You Need",
    )


@patch("project.infrastructure.database.papers.get_connection")
def test_register_paper_requires_version(mock_get_connection):
    paper = make_paper()
    paper["arxiv_id"] = "1706.03762"

    with pytest.raises(ValueError, match="versioned arXiv ID"):
        register_paper(paper)

    mock_get_connection.assert_not_called()


@patch("project.infrastructure.database.papers.get_connection")
def test_get_paper(mock_get_connection):
    row = {"arxiv_id": "1706.03762", "version": 7}

    connection = MagicMock()
    connection.execute.return_value.fetchone.return_value = row
    mock_get_connection.return_value.__enter__.return_value = connection

    result = get_paper("1706.03762v7")

    assert result == row

    query, params = connection.execute.call_args.args
    assert "SELECT *" in query
    assert "FROM papers" in query
    assert params == ("1706.03762", 7)


@patch("project.infrastructure.database.papers.get_connection")
def test_get_nonexistent_paper(mock_get_connection):
    connection = MagicMock()
    connection.execute.return_value.fetchone.return_value = None
    mock_get_connection.return_value.__enter__.return_value = connection

    assert get_paper("1706.03762v7") is None


@patch("project.infrastructure.database.papers.get_connection")
def test_update_paper_status(mock_get_connection):
    row = {
        "arxiv_id": "1706.03762",
        "version": 7,
        "processing_status": "ready",
    }

    connection = MagicMock()
    connection.execute.return_value.fetchone.return_value = row
    mock_get_connection.return_value.__enter__.return_value = connection

    result = update_paper_status("1706.03762v7", "ready")

    assert result == row

    query, params = connection.execute.call_args.args
    assert "UPDATE papers" in query
    assert params == ("ready", "1706.03762", 7)


def test_update_paper_status_rejects_invalid_status():
    with pytest.raises(ValueError, match="Invalid processing status"):
        update_paper_status("1706.03762v7", "completed")


@patch("project.infrastructure.database.papers.get_connection")
def test_add_paper_to_session(mock_get_connection):
    session_id = uuid4()
    paper_id = uuid4()

    connection = MagicMock()
    mock_get_connection.return_value.__enter__.return_value = connection

    add_paper_to_session(session_id, paper_id)

    queries = [
        call.args[0]
        for call in connection.execute.call_args_list
    ]

    assert any("INSERT INTO session_papers" in query for query in queries)


@patch("project.infrastructure.database.papers.get_connection")
def test_add_active_paper_to_session(mock_get_connection):
    session_id = uuid4()
    paper_id = uuid4()

    connection = MagicMock()
    mock_get_connection.return_value.__enter__.return_value = connection

    add_paper_to_session(session_id, paper_id, is_active=True)

    queries = [
        call.args[0]
        for call in connection.execute.call_args_list
    ]

    assert any("FOR UPDATE" in query for query in queries)
    assert any("SET is_active = FALSE" in query for query in queries)
    assert any("INSERT INTO session_papers" in query for query in queries)


@patch("project.infrastructure.database.papers.get_connection")
def test_get_session_papers(mock_get_connection):
    session_id = uuid4()
    paper_id = uuid4()

    rows = [
        {
            "id": paper_id,
            "arxiv_id": "1706.03762",
            "version": 7,
            "is_active": True,
        }
    ]

    connection = MagicMock()
    connection.execute.return_value.fetchall.return_value = rows
    mock_get_connection.return_value.__enter__.return_value = connection

    result = get_session_papers(session_id)

    assert result == rows
    assert len(result) == 1
    assert result[0]["is_active"] is True

    query, params = connection.execute.call_args.args
    assert "JOIN papers AS p" in query
    assert "WHERE sp.session_id = %s" in query
    assert params == (session_id,)


@patch("project.infrastructure.database.papers.get_connection")
def test_set_active_paper(mock_get_connection):
    session_id = uuid4()
    paper_id = uuid4()

    connection = MagicMock()
    connection.execute.return_value.fetchone.return_value = (1,)
    mock_get_connection.return_value.__enter__.return_value = connection

    result = set_active_paper(session_id, paper_id)

    assert result is True

    queries = [
        call.args[0]
        for call in connection.execute.call_args_list
    ]

    assert any("FOR UPDATE" in query for query in queries)
    assert any("SET is_active = FALSE" in query for query in queries)
    assert any("SET is_active = TRUE" in query for query in queries)


@patch("project.infrastructure.database.papers.get_connection")
def test_set_active_paper_not_in_session(mock_get_connection):
    connection = MagicMock()
    connection.execute.return_value.fetchone.side_effect = [
        {"id": uuid4()},
        None,
    ]
    mock_get_connection.return_value.__enter__.return_value = connection

    result = set_active_paper(uuid4(), uuid4())

    assert result is False


@patch("project.infrastructure.database.papers.get_connection")
def test_remove_paper_from_session(mock_get_connection):
    session_id = uuid4()
    paper_id = uuid4()

    connection = MagicMock()
    connection.execute.return_value.fetchone.return_value = (paper_id,)
    mock_get_connection.return_value.__enter__.return_value = connection

    result = remove_paper_from_session(session_id, paper_id)

    assert result is True

    query, params = connection.execute.call_args.args
    assert "DELETE FROM session_papers" in query
    assert params == (session_id, paper_id)


@patch("project.infrastructure.database.papers.get_connection")
def test_remove_paper_not_in_session(mock_get_connection):
    connection = MagicMock()
    connection.execute.return_value.fetchone.side_effect = [
        {"id": uuid4()},
        None,
    ]
    mock_get_connection.return_value.__enter__.return_value = connection

    result = remove_paper_from_session(uuid4(), uuid4())

    assert result is False


@patch("project.infrastructure.database.papers.get_connection")
def test_save_paper_artifact(mock_get_connection):
    paper_id = uuid4()
    connection = MagicMock()
    mock_get_connection.return_value.__enter__.return_value = connection

    save_paper_artifact(paper_id, "papers/1706.03762v7.pdf")

    query, params = connection.execute.call_args.args

    assert "INSERT INTO paper_artifacts" in query
    assert "ON CONFLICT (paper_id)" in query
    assert params == (paper_id, "papers/1706.03762v7.pdf")


@patch("project.infrastructure.database.papers.get_connection")
def test_save_paper_content(mock_get_connection):
    paper_id = uuid4()
    connection = MagicMock()
    mock_get_connection.return_value.__enter__.return_value = connection

    pages = [
        {"page_number": 1, "text": "First page."},
        {"page_number": 2, "text": "Second page."},
    ]
    chunks = [
        {"chunk_id": 0, "page_number": 1, "text": "First chunk."},
        {"chunk_id": 1, "page_number": 2, "text": "Second chunk."},
    ]

    save_paper_content(paper_id, pages, chunks)

    queries = [
        call.args[0]
        for call in connection.execute.call_args_list
    ]

    assert "DELETE FROM paper_pages" in queries[0]
    assert sum("INSERT INTO paper_pages" in query for query in queries) == 2
    assert sum("INSERT INTO paper_chunks" in query for query in queries) == 2

    assert connection.execute.call_args_list[1].args[1] == (
        paper_id, 1, "First page."
    )
    assert connection.execute.call_args_list[2].args[1] == (
        paper_id, 2, "Second page."
    )
    assert connection.execute.call_args_list[3].args[1] == (
        paper_id, 0, 1, "First chunk."
    )
    assert connection.execute.call_args_list[4].args[1] == (
        paper_id, 1, 2, "Second chunk."
    )
