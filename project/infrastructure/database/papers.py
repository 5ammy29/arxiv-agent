import json
import re
from typing import Any
from uuid import UUID

from project.infrastructure.database.connection import get_connection
from project.infrastructure.database.paper_locks import (
    PaperProcessingLease,
    assert_current_paper_processing_lease,
)


version_pattern = re.compile(r"^(?P<base>.+?)v(?P<version>\d+)$")


def _assert_lease_paper(lease: PaperProcessingLease, paper_id: UUID) -> None:
    if lease.paper_id != paper_id:
        raise ValueError("Processing lease does not belong to this paper.")


def parse_arxiv_id(id: str) -> tuple[str, int | None]:
    id = id.strip()

    if "/abs/" in id:
        id = id.split("/abs/", maxsplit=1)[1]
    elif "/pdf/" in id:
        id = id.split("/pdf/", maxsplit=1)[1]

    if id.endswith(".pdf"):
        id = id[:-4]

    match = version_pattern.fullmatch(id)

    if match:
        return match.group("base"), int(match.group("version"))

    return id, None


def register_paper(paper: dict[str, Any]) -> dict[str, Any]:
    arxiv_id, version = parse_arxiv_id(paper["arxiv_id"])

    if version is None:
        raise ValueError("Paper must have a versioned arXiv ID.")

    with get_connection() as connection:
        row = connection.execute(
            """
            INSERT INTO papers (
                arxiv_id, version, title, authors, abstract,
                published_at, arxiv_updated_at, url, pdf_url, categories
            )
            VALUES (
                %s, %s, %s, %s::jsonb, %s,
                %s, %s, %s, %s, %s::jsonb
            )
            ON CONFLICT (arxiv_id, version)
            DO UPDATE SET updated_at = papers.updated_at
            RETURNING *
            """,
            (
                arxiv_id,
                version,
                paper["title"],
                json.dumps(paper.get("authors", [])),
                paper["abstract"],
                paper["published"],
                paper["updated"],
                paper["url"],
                paper["pdf_url"],
                json.dumps(paper.get("categories", [])),
            ),
        ).fetchone()

    return dict(row)


def get_paper(arxiv_id: str) -> dict[str, Any] | None:
    base_id, version = parse_arxiv_id(arxiv_id)

    if version is None:
        raise ValueError("Paper must have a versioned arXiv ID.")

    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT *
            FROM papers
            WHERE arxiv_id = %s AND version = %s
            """,
            (base_id, version),
        ).fetchone()

    return dict(row) if row is not None else None


def update_paper_status(
    arxiv_id: str,
    status: str,
    *,
    lease: PaperProcessingLease,
) -> dict[str, Any] | None:
    base_id, version = parse_arxiv_id(arxiv_id)

    if version is None:
        raise ValueError("Paper must have a versioned arXiv ID.")

    allowed_statuses = {
        "pending",
        "downloading",
        "parsing",
        "ready",
        "failed",
    }

    if status not in allowed_statuses:
        raise ValueError(f"Invalid processing status: {status}")

    with get_connection() as connection:
        assert_current_paper_processing_lease(connection, lease)
        row = connection.execute(
            """
            UPDATE papers
            SET processing_status = %s,
                updated_at = NOW()
            WHERE arxiv_id = %s AND version = %s AND id = %s
            RETURNING *
            """,
            (status, base_id, version, lease.paper_id),
        ).fetchone()

    return dict(row) if row is not None else None


def add_paper_to_session(
    session_id: UUID,
    paper_id: UUID,
    is_active: bool = False,
) -> None:
    with get_connection() as connection:
        session = connection.execute(
            "SELECT id FROM sessions WHERE id = %s FOR UPDATE",
            (session_id,),
        ).fetchone()

        if session is None:
            raise ValueError("Session does not exist.")

        if is_active:
            connection.execute(
                """
                UPDATE session_papers
                SET is_active = FALSE
                WHERE session_id = %s AND is_active = TRUE
                """,
                (session_id,),
            )

        connection.execute(
            """
            INSERT INTO session_papers (session_id, paper_id, is_active)
            VALUES (%s, %s, %s)
            ON CONFLICT (session_id, paper_id)
            DO UPDATE SET is_active = EXCLUDED.is_active
            """,
            (session_id, paper_id, is_active),
        )


def get_session_papers(session_id: UUID) -> list[dict[str, Any]]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                p.*,
                sp.is_active,
                sp.added_at
            FROM session_papers AS sp
            JOIN papers AS p ON p.id = sp.paper_id
            WHERE sp.session_id = %s
            ORDER BY sp.added_at DESC
            """,
            (session_id,),
        ).fetchall()

    return [dict(row) for row in rows]


def set_active_paper(session_id: UUID, paper_id: UUID) -> bool:
    with get_connection() as connection:
        session = connection.execute(
            "SELECT id FROM sessions WHERE id = %s FOR UPDATE",
            (session_id,),
        ).fetchone()

        if session is None:
            raise ValueError("Session does not exist.")

        exists = connection.execute(
            """
            SELECT 1
            FROM session_papers
            WHERE session_id = %s AND paper_id = %s
            """,
            (session_id, paper_id),
        ).fetchone()

        if exists is None:
            return False

        connection.execute(
            """
            UPDATE session_papers
            SET is_active = FALSE
            WHERE session_id = %s AND is_active = TRUE
            """,
            (session_id,),
        )

        connection.execute(
            """
            UPDATE session_papers
            SET is_active = TRUE
            WHERE session_id = %s AND paper_id = %s
            """,
            (session_id, paper_id),
        )

    return True


def remove_paper_from_session(session_id: UUID, paper_id: UUID) -> bool:
    with get_connection() as connection:
        session = connection.execute(
            "SELECT id FROM sessions WHERE id = %s FOR UPDATE",
            (session_id,),
        ).fetchone()

        if session is None:
            raise ValueError("Session does not exist.")

        row = connection.execute(
            """
            DELETE FROM session_papers
            WHERE session_id = %s AND paper_id = %s
            RETURNING paper_id
            """,
            (session_id, paper_id),
        ).fetchone()

    return row is not None


def save_paper_artifact(
    paper_id: UUID,
    pdf_storage_path: str,
    content_fingerprint: str | None = None,
    chunking_fingerprint: str | None = None,
    *,
    lease: PaperProcessingLease,
) -> None:
    _assert_lease_paper(lease, paper_id)
    with get_connection() as connection:
        assert_current_paper_processing_lease(connection, lease)
        connection.execute(
            """
            INSERT INTO paper_artifacts (
                paper_id, pdf_storage_path, content_fingerprint,
                chunking_fingerprint
            )
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (paper_id)
            DO UPDATE SET
                pdf_storage_path = EXCLUDED.pdf_storage_path,
                content_fingerprint = EXCLUDED.content_fingerprint,
                chunking_fingerprint = EXCLUDED.chunking_fingerprint,
                updated_at = NOW()
            """,
            (
                paper_id,
                pdf_storage_path,
                content_fingerprint,
                chunking_fingerprint,
            ),
        )


def save_paper_content(
    paper_id: UUID,
    pages: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    content_fingerprint: str,
    chunking_fingerprint: str,
    *,
    lease: PaperProcessingLease,
) -> str | None:
    _assert_lease_paper(lease, paper_id)
    with get_connection() as connection:
        assert_current_paper_processing_lease(connection, lease)
        previous_index = connection.execute(
            """
            DELETE FROM paper_vector_indexes
            WHERE paper_id = %s
            RETURNING index_storage_path
            """,
            (paper_id,),
        ).fetchone()

        connection.execute(
            "DELETE FROM paper_pages WHERE paper_id = %s",
            (paper_id,),
        )

        for page in pages:
            connection.execute(
                """
                INSERT INTO paper_pages (paper_id, page_number, text)
                VALUES (%s, %s, %s)
                """,
                (paper_id, page["page_number"], page["text"]),
            )

        for chunk in chunks:
            connection.execute(
                """
                INSERT INTO paper_chunks (
                    paper_id, chunk_id, page_number, text
                )
                VALUES (%s, %s, %s, %s)
                """,
                (
                    paper_id,
                    chunk["chunk_id"],
                    chunk["page_number"],
                    chunk["text"],
                ),
            )

        result = connection.execute(
            """
            UPDATE paper_artifacts
            SET content_fingerprint = %s,
                chunking_fingerprint = %s,
                updated_at = NOW()
            WHERE paper_id = %s
            RETURNING paper_id
            """,
            (content_fingerprint, chunking_fingerprint, paper_id),
        )

        if result.fetchone() is None:
            raise RuntimeError(
                "Cannot save content: paper artifact does not exist."
            )

    return (
        previous_index["index_storage_path"]
        if previous_index is not None
        else None
    )


def invalidate_paper_content(
    paper_id: UUID,
    *,
    lease: PaperProcessingLease,
) -> str | None:
    _assert_lease_paper(lease, paper_id)
    with get_connection() as connection:
        assert_current_paper_processing_lease(connection, lease)
        previous_index = connection.execute(
            """
            DELETE FROM paper_vector_indexes
            WHERE paper_id = %s
            RETURNING index_storage_path
            """,
            (paper_id,),
        ).fetchone()

        connection.execute(
            "DELETE FROM paper_pages WHERE paper_id = %s",
            (paper_id,),
        )

        connection.execute(
            """
            UPDATE paper_artifacts
            SET content_fingerprint = NULL,
                chunking_fingerprint = NULL,
                updated_at = NOW()
            WHERE paper_id = %s
            """,
            (paper_id,),
        )

    return (
        previous_index["index_storage_path"]
        if previous_index is not None
        else None
    )


def get_paper_content(paper_id: UUID) -> dict[str, list[dict[str, Any]]]:
    with get_connection() as connection:
        pages = connection.execute(
            """
            SELECT page_number, text
            FROM paper_pages
            WHERE paper_id = %s
            ORDER BY page_number
            """,
            (paper_id,),
        ).fetchall()

        chunks = connection.execute(
            """
            SELECT chunk_id, page_number, text
            FROM paper_chunks
            WHERE paper_id = %s
            ORDER BY chunk_id
            """,
            (paper_id,),
        ).fetchall()

    return {
        "pages": [dict(page) for page in pages],
        "chunks": [dict(chunk) for chunk in chunks],
    }


def get_paper_artifact(paper_id: UUID) -> dict[str, Any] | None:
    with get_connection() as connection:
        artifact = connection.execute(
            """
            SELECT pdf_storage_path, content_fingerprint, chunking_fingerprint
            FROM paper_artifacts
            WHERE paper_id = %s
            """,
            (paper_id,),
        ).fetchone()

    return dict(artifact) if artifact else None


def replace_paper_pdf(
    paper_id: UUID,
    new_pdf_storage_path: str,
    *,
    lease: PaperProcessingLease,
) -> dict[str, str | None]:
    if not new_pdf_storage_path.strip():
        raise ValueError("PDF storage path must not be empty.")
    _assert_lease_paper(lease, paper_id)

    with get_connection() as connection:
        assert_current_paper_processing_lease(connection, lease)
        paper = connection.execute(
            """
            SELECT id
            FROM papers
            WHERE id = %s
            FOR UPDATE
            """,
            (paper_id,),
        ).fetchone()

        if paper is None:
            raise ValueError(f"Paper {paper_id} does not exist.")

        previous_artifact = connection.execute(
            """
            SELECT pdf_storage_path
            FROM paper_artifacts
            WHERE paper_id = %s
            FOR UPDATE
            """,
            (paper_id,),
        ).fetchone()

        previous_pdf_path = (
            previous_artifact["pdf_storage_path"]
            if previous_artifact is not None
            else None
        )

        previous_index = connection.execute(
            """
            DELETE FROM paper_vector_indexes
            WHERE paper_id = %s
            RETURNING index_storage_path
            """,
            (paper_id,),
        ).fetchone()

        connection.execute(
            """
            DELETE FROM paper_pages
            WHERE paper_id = %s
            """,
            (paper_id,),
        )

        connection.execute(
            """
            INSERT INTO paper_artifacts (
                paper_id,
                pdf_storage_path,
                content_fingerprint,
                chunking_fingerprint
            )
            VALUES (%s, %s, NULL, NULL)
            ON CONFLICT (paper_id)
            DO UPDATE SET
                pdf_storage_path = EXCLUDED.pdf_storage_path,
                content_fingerprint = NULL,
                chunking_fingerprint = NULL,
                updated_at = NOW()
            """,
            (paper_id, new_pdf_storage_path),
        )

    return {
        "old_pdf_storage_path": (
            previous_pdf_path
            if previous_pdf_path != new_pdf_storage_path
            else None
        ),
        "old_index_storage_path": (
            previous_index["index_storage_path"]
            if previous_index is not None
            else None
        ),
    }


def get_referenced_artifact_paths() -> set[str]:
    """Return the database-authoritative set of active storage objects."""
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT pdf_storage_path AS storage_path
            FROM paper_artifacts
            UNION
            SELECT index_storage_path AS storage_path
            FROM paper_vector_indexes
            """
        ).fetchall()
    return {row["storage_path"] for row in rows if row["storage_path"]}


def get_active_processing_artifact_prefixes() -> set[str]:
    """Return paths that cleanup must preserve while their lease is live."""
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                'papers/' || p.arxiv_id || 'v' || p.version || '/' AS pdf_prefix,
                'vector_indexes/' || p.id::text || '/' AS index_prefix
            FROM paper_processing_locks AS l
            JOIN papers AS p ON p.id = l.paper_id
            WHERE l.lease_expires_at > clock_timestamp()
            """
        ).fetchall()
    return {
        prefix
        for row in rows
        for prefix in (row["pdf_prefix"], row["index_prefix"])
        if prefix
    }
