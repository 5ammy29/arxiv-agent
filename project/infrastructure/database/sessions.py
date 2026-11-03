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
        "id": row["id"],
        "user_id": row["user_id"],
        "title": row["title"],
        "status": row["status"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
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
        "id": row["id"],
        "user_id": row["user_id"],
        "title": row["title"],
        "status": row["status"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
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
            "id": row["id"],
            "user_id": row["user_id"],
            "title": row["title"],
            "status": row["status"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

        sessions.append(session)

    return sessions

