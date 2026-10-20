from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

from project.infrastructure.database.sessions import (
    create_session,
    get_session,
    list_sessions
)


USER_A = UUID("e18d69e9-a0f5-4817-8a5b-408d0ff7c62b")
USER_B = UUID("bed68ecc-e5a7-420f-b861-697d62a432a1")


def make_row(session_id, user_id, title="Test session"):
    return (
        session_id,
        user_id,
        title,
        "active",
        "created_at",
        "updated_at",
    )


@patch("project.infrastructure.database.sessions.uuid4")
@patch("project.infrastructure.database.sessions.get_connection")
def test_create_session(mock_get_connection, mock_uuid4):
    session_id = uuid4()
    row = make_row(session_id, USER_A)

    mock_uuid4.return_value = session_id

    connection = MagicMock()
    connection.execute.return_value.fetchone.return_value = row
    mock_get_connection.return_value.__enter__.return_value = connection

    result = create_session(USER_A, "Test session")

    assert result["id"] == session_id
    assert result["user_id"] == USER_A
    assert result["title"] == "Test session"
    assert result["status"] == "active"

    query, params = connection.execute.call_args.args
    assert "INSERT INTO public.sessions" in query
    assert params == (session_id, USER_A, "Test session")


@patch("project.infrastructure.database.sessions.get_connection")
def test_get_session_for_owner(mock_get_connection):
    session_id = uuid4()
    row = make_row(session_id, USER_A)

    connection = MagicMock()
    connection.execute.return_value.fetchone.return_value = row
    mock_get_connection.return_value.__enter__.return_value = connection

    result = get_session(session_id, USER_A)

    assert result is not None
    assert result["id"] == session_id
    assert result["user_id"] == USER_A

    query, params = connection.execute.call_args.args
    assert "WHERE id = %s AND user_id = %s" in query
    assert params == (session_id, USER_A)


@patch("project.infrastructure.database.sessions.get_connection")
def test_get_session_for_another_user(mock_get_connection):
    connection = MagicMock()
    connection.execute.return_value.fetchone.return_value = None
    mock_get_connection.return_value.__enter__.return_value = connection

    result = get_session(uuid4(), USER_B)

    assert result is None


@patch("project.infrastructure.database.sessions.get_connection")
def test_get_nonexistent_session(mock_get_connection):
    connection = MagicMock()
    connection.execute.return_value.fetchone.return_value = None
    mock_get_connection.return_value.__enter__.return_value = connection

    result = get_session(uuid4(), USER_A)

    assert result is None


@patch("project.infrastructure.database.sessions.get_connection")
def test_list_sessions(mock_get_connection):
    session_1 = make_row(uuid4(), USER_A, "Session 1")
    session_2 = make_row(uuid4(), USER_A, "Session 2")

    connection = MagicMock()
    connection.execute.return_value.fetchall.return_value = [
        session_1,
        session_2,
    ]
    mock_get_connection.return_value.__enter__.return_value = connection

    result = list_sessions(USER_A)

    assert len(result) == 2
    assert result[0]["title"] == "Session 1"
    assert result[1]["title"] == "Session 2"

    query, params = connection.execute.call_args.args
    assert "WHERE user_id = %s" in query
    assert "LIMIT %s OFFSET %s" in query
    assert params == (USER_A, 20, 0)


@patch("project.infrastructure.database.sessions.get_connection")
def test_list_sessions_with_pagination(mock_get_connection):
    connection = MagicMock()
    connection.execute.return_value.fetchall.return_value = []
    mock_get_connection.return_value.__enter__.return_value = connection

    result = list_sessions(USER_A, limit=10, offset=20)

    assert result == []

    query, params = connection.execute.call_args.args
    assert params == (USER_A, 10, 20)