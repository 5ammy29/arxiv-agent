from uuid import UUID, uuid4

from project.infrastructure.database.connection import get_connection


def create_session(user_id: UUID, title: str | None = None) -> dict:
    session_id = uuid4()

    with get_connection() as connection:
        row = connection.execute(
            """
            INSERT INTO public.sessions (id, user_id, title)
            VALUES (%s, %s, %s)
            RETURNING id, user_id, title, status, created_at, updated_at
            """,
            (session_id, user_id, title),
        ).fetchone()

    return {
        "id": row[0],
        "user_id": row[1],
        "title": row[2],
        "status": row[3],
        "created_at": row[4],
        "updated_at": row[5],
    }


def get_session(session_id: UUID, user_id: UUID) -> dict | None:
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT id, user_id, title, status, created_at, updated_at
            FROM public.sessions
            WHERE id = %s AND user_id = %s
            """,
            (session_id, user_id),
        ).fetchone()

    if row is None:
        return None

    return {
        "id": row[0],
        "user_id": row[1],
        "title": row[2],
        "status": row[3],
        "created_at": row[4],
        "updated_at": row[5],
    }


def list_sessions(user_id: UUID, limit: int = 20, offset: int = 0) -> list[dict]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT id, user_id, title, status, created_at, updated_at
            FROM public.sessions
            WHERE user_id = %s
            ORDER BY updated_at DESC
            LIMIT %s OFFSET %s
            """,
            (user_id, limit, offset),
        ).fetchall()

    sessions = []

    for row in rows:
        session = {
            "id": row[0],
            "user_id": row[1],
            "title": row[2],
            "status": row[3],
            "created_at": row[4],
            "updated_at": row[5],
        }

        sessions.append(session)

    return sessions

