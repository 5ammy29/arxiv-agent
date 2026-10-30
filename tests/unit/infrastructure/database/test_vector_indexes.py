import json
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from project.infrastructure.database.vector_indexes import (
    delete_paper_vector_index,
    get_paper_vector_index,
    save_paper_vector_index,
)


@patch("project.infrastructure.database.vector_indexes.get_connection")
def test_save_paper_vector_index(mock_get_connection):
    paper_id = uuid4()
    row = {
        "paper_id": paper_id,
        "index_storage_path": f"vector_indexes/{paper_id}.faiss",
        "embedding_model": "BAAI/bge-base-en-v1.5",
        "embedding_dimension": 768,
        "chunk_ids": [0, 1, 2],
        "chunk_count": 3,
    }

    connection = MagicMock()
    connection.execute.return_value.fetchone.return_value = row
    mock_get_connection.return_value.__enter__.return_value = connection

    result = save_paper_vector_index(
        paper_id=paper_id,
        index_storage_path=f"vector_indexes/{paper_id}.faiss",
        embedding_model="BAAI/bge-base-en-v1.5",
        embedding_dimension=768,
        chunk_ids=[0, 1, 2],
    )

    assert result == row

    query, params = connection.execute.call_args.args
    assert "INSERT INTO paper_vector_indexes" in query
    assert "ON CONFLICT (paper_id)" in query
    assert params[0] == paper_id
    assert params[1] == f"vector_indexes/{paper_id}.faiss"
    assert params[2] == "BAAI/bge-base-en-v1.5"
    assert params[3] == 768
    assert json.loads(params[4]) == [0, 1, 2]
    assert params[5] == 3


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"index_storage_path": ""}, "Index storage path"),
        ({"embedding_model": ""}, "Embedding model"),
        ({"embedding_dimension": 0}, "Embedding dimension"),
        ({"chunk_ids": []}, "Chunk IDs"),
    ],
)
def test_save_paper_vector_index_rejects_invalid_input(
    overrides,
    message,
):
    arguments = {
        "paper_id": uuid4(),
        "index_storage_path": "vector_indexes/test.faiss",
        "embedding_model": "BAAI/bge-base-en-v1.5",
        "embedding_dimension": 768,
        "chunk_ids": [0, 1],
    }
    arguments.update(overrides)

    with patch(
        "project.infrastructure.database.vector_indexes.get_connection"
    ) as mock_get_connection:
        with pytest.raises(ValueError, match=message):
            save_paper_vector_index(**arguments)

        mock_get_connection.assert_not_called()


@patch("project.infrastructure.database.vector_indexes.get_connection")
def test_get_paper_vector_index(mock_get_connection):
    paper_id = uuid4()
    row = {
        "paper_id": paper_id,
        "index_storage_path": f"vector_indexes/{paper_id}.faiss",
        "embedding_model": "BAAI/bge-base-en-v1.5",
        "embedding_dimension": 768,
        "chunk_ids": [0, 1],
        "chunk_count": 2,
    }

    connection = MagicMock()
    connection.execute.return_value.fetchone.return_value = row
    mock_get_connection.return_value.__enter__.return_value = connection

    result = get_paper_vector_index(paper_id)

    assert result == row

    query, params = connection.execute.call_args.args
    assert "SELECT *" in query
    assert "FROM paper_vector_indexes" in query
    assert "WHERE paper_id = %s" in query
    assert params == (paper_id,)


@patch("project.infrastructure.database.vector_indexes.get_connection")
def test_get_nonexistent_paper_vector_index(mock_get_connection):
    connection = MagicMock()
    connection.execute.return_value.fetchone.return_value = None
    mock_get_connection.return_value.__enter__.return_value = connection

    assert get_paper_vector_index(uuid4()) is None


@patch("project.infrastructure.database.vector_indexes.get_connection")
def test_delete_paper_vector_index(mock_get_connection):
    paper_id = uuid4()

    connection = MagicMock()
    connection.execute.return_value.fetchone.return_value = {
        "paper_id": paper_id
    }
    mock_get_connection.return_value.__enter__.return_value = connection

    assert delete_paper_vector_index(paper_id) is True

    query, params = connection.execute.call_args.args
    assert "DELETE FROM paper_vector_indexes" in query
    assert "RETURNING paper_id" in query
    assert params == (paper_id,)


@patch("project.infrastructure.database.vector_indexes.get_connection")
def test_delete_nonexistent_paper_vector_index(mock_get_connection):
    connection = MagicMock()
    connection.execute.return_value.fetchone.return_value = None
    mock_get_connection.return_value.__enter__.return_value = connection

    assert delete_paper_vector_index(uuid4()) is False
