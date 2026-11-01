from typing import Any
from uuid import UUID

from project.infrastructure.database.connection import get_connection
from project.infrastructure.database.paper_locks import (
    PaperProcessingLease,
    assert_current_paper_processing_lease,
)


class PaperContentChangedError(RuntimeError):
    """Raised when an index was built from no-longer-current paper content."""


def save_paper_vector_index(
    paper_id: UUID,
    index_storage_path: str,
    embedding_model: str,
    embedding_dimension: int,
    chunk_ids: list[int],
    content_fingerprint: str,
    chunking_fingerprint: str,
    *,
    lease: PaperProcessingLease,
) -> dict[str, Any]:
    """Persist metadata for a paper's FAISS index."""
    if not index_storage_path.strip():
        raise ValueError("Index storage path must not be empty.")

    if not embedding_model.strip():
        raise ValueError("Embedding model must not be empty.")

    if embedding_dimension <= 0:
        raise ValueError("Embedding dimension must be positive.")

    if not chunk_ids:
        raise ValueError("Chunk IDs must not be empty.")

    if not content_fingerprint.strip():
        raise ValueError("Content fingerprint must not be empty.")

    if not chunking_fingerprint.strip():
        raise ValueError("Chunking fingerprint must not be empty.")
    if lease.paper_id != paper_id:
        raise ValueError("Processing lease does not belong to this paper.")

    with get_connection() as connection:
        assert_current_paper_processing_lease(connection, lease)
        artifact = connection.execute(
            """
            SELECT content_fingerprint, chunking_fingerprint
            FROM paper_artifacts
            WHERE paper_id = %s
            FOR UPDATE
            """,
            (paper_id,),
        ).fetchone()
        if (
            artifact is None
            or artifact["content_fingerprint"] != content_fingerprint
            or artifact["chunking_fingerprint"] != chunking_fingerprint
        ):
            raise PaperContentChangedError(
                "Cannot publish a vector index for stale paper content."
            )
        row = connection.execute(
            """
            INSERT INTO paper_vector_indexes (
                paper_id,
                index_storage_path,
                embedding_model,
                embedding_dimension,
                chunk_ids,
                chunk_count,
                content_fingerprint,
                chunking_fingerprint
            )
            VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s, %s)
            ON CONFLICT (paper_id)
            DO UPDATE SET
                index_storage_path = EXCLUDED.index_storage_path,
                embedding_model = EXCLUDED.embedding_model,
                embedding_dimension = EXCLUDED.embedding_dimension,
                chunk_ids = EXCLUDED.chunk_ids,
                chunk_count = EXCLUDED.chunk_count,
                content_fingerprint = EXCLUDED.content_fingerprint,
                chunking_fingerprint = EXCLUDED.chunking_fingerprint,
                updated_at = NOW()
            RETURNING *
            """,
            (
                paper_id,
                index_storage_path,
                embedding_model,
                embedding_dimension,
                __import__("json").dumps(chunk_ids),
                len(chunk_ids),
                content_fingerprint,
                chunking_fingerprint,
            ),
        ).fetchone()

    return dict(row)


def get_paper_vector_index(paper_id: UUID) -> dict[str, Any] | None:
    """Retrieve cached FAISS index metadata for a paper."""
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT *
            FROM paper_vector_indexes
            WHERE paper_id = %s
            """,
            (paper_id,),
        ).fetchone()

    return dict(row) if row is not None else None


def delete_paper_vector_index(
    paper_id: UUID,
    *,
    lease: PaperProcessingLease,
) -> str | None:
    """Delete index metadata and return the orphaned storage path, if any."""
    if lease.paper_id != paper_id:
        raise ValueError("Processing lease does not belong to this paper.")
    with get_connection() as connection:
        assert_current_paper_processing_lease(connection, lease)
        row = connection.execute(
            """
            DELETE FROM paper_vector_indexes
            WHERE paper_id = %s
            RETURNING index_storage_path
            """,
            (paper_id,),
        ).fetchone()

    return row["index_storage_path"] if row is not None else None
