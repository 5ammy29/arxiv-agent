import json
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from project.infrastructure.database.vector_indexes import (
    PaperContentChangedError,
    delete_paper_vector_index,
    get_paper_vector_index,
    save_paper_vector_index,
)
from project.infrastructure.database.paper_locks import PaperProcessingLease


def processing_lease(paper_id):
    return PaperProcessingLease(paper_id, uuid4(), 1)


@patch("project.infrastructure.database.vector_indexes.get_connection")
def test_save_paper_vector_index(mock_get_connection):
    paper_id = uuid4()
    row = {
        "paper_id": paper_id,
        "index_storage_path": f"vector_indexes/{paper_id}/test.faiss",
        "embedding_model": "BAAI/bge-base-en-v1.5",
        "embedding_dimension": 768,
        "chunk_ids": [0, 1, 2],
        "chunk_count": 3,
        "content_fingerprint": "content-fingerprint",
        "chunking_fingerprint": "chunking-fingerprint",
    }

    connection = MagicMock()
    connection.execute.return_value.fetchone.return_value = row
    mock_get_connection.return_value.__enter__.return_value = connection

    result = save_paper_vector_index(
        paper_id=paper_id,
        index_storage_path=f"vector_indexes/{paper_id}/test.faiss",
        embedding_model="BAAI/bge-base-en-v1.5",
        embedding_dimension=768,
        chunk_ids=[0, 1, 2],
        content_fingerprint="content-fingerprint",
        chunking_fingerprint="chunking-fingerprint",
        lease=processing_lease(paper_id),
    )

    assert result == row

    query, params = connection.execute.call_args.args
    assert "INSERT INTO paper_vector_indexes" in query
    assert "ON CONFLICT (paper_id)" in query
    assert params[0] == paper_id
    assert params[1] == f"vector_indexes/{paper_id}/test.faiss"
    assert params[2] == "BAAI/bge-base-en-v1.5"
    assert params[3] == 768
    assert json.loads(params[4]) == [0, 1, 2]
    assert params[5] == 3
    assert params[6] == "content-fingerprint"
    assert params[7] == "chunking-fingerprint"


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
    paper_id = uuid4()
    arguments = {
        "paper_id": paper_id,
        "index_storage_path": "vector_indexes/test-paper/test.faiss",
        "embedding_model": "BAAI/bge-base-en-v1.5",
        "embedding_dimension": 768,
        "chunk_ids": [0, 1],
        "content_fingerprint": "content-fingerprint",
        "chunking_fingerprint": "chunking-fingerprint",
        "lease": processing_lease(paper_id),
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
        "index_storage_path": f"vector_indexes/{paper_id}/test.faiss",
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
def test_save_vector_index_rejects_content_changed_while_waiting(mock_get_connection):
    paper_id = uuid4()
    connection = MagicMock()
    connection.execute.side_effect = [
        MagicMock(fetchone=MagicMock(return_value={"paper_id": paper_id})),
        MagicMock(
            fetchone=MagicMock(
                return_value={
                    "content_fingerprint": "new-content",
                    "chunking_fingerprint": "chunking-fingerprint",
                }
            )
        ),
    ]
    mock_get_connection.return_value.__enter__.return_value = connection

    with pytest.raises(PaperContentChangedError, match="stale paper content"):
        save_paper_vector_index(
            paper_id=paper_id,
            index_storage_path=f"vector_indexes/{paper_id}/test.faiss",
            embedding_model="model",
            embedding_dimension=3,
            chunk_ids=[0],
            content_fingerprint="old-content",
            chunking_fingerprint="chunking-fingerprint",
            lease=processing_lease(paper_id),
        )


@patch("project.infrastructure.database.vector_indexes.get_connection")
def test_delete_paper_vector_index(mock_get_connection):
    paper_id = uuid4()

    connection = MagicMock()
    connection.execute.return_value.fetchone.return_value = {
        "index_storage_path": f"vector_indexes/{paper_id}/test.faiss"
    }
    mock_get_connection.return_value.__enter__.return_value = connection

    assert delete_paper_vector_index(
        paper_id, lease=processing_lease(paper_id)
    ) == connection.execute.return_value.fetchone.return_value["index_storage_path"]

    query, params = connection.execute.call_args.args
    assert "DELETE FROM paper_vector_indexes" in query
    assert "RETURNING index_storage_path" in query
    assert params == (paper_id,)


@patch("project.infrastructure.database.vector_indexes.get_connection")
def test_delete_nonexistent_paper_vector_index(mock_get_connection):
    connection = MagicMock()
    mock_get_connection.return_value.__enter__.return_value = connection

    paper_id = uuid4()
    connection.execute.side_effect = [
        MagicMock(fetchone=MagicMock(return_value={"paper_id": paper_id})),
        MagicMock(fetchone=MagicMock(return_value=None)),
    ]
    assert delete_paper_vector_index(paper_id, lease=processing_lease(paper_id)) is None
