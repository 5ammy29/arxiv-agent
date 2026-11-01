from dataclasses import dataclass
from uuid import UUID

from project.infrastructure.database.connection import get_connection


def _validate_lease_duration(lease_duration_seconds: int) -> None:
    if (
        isinstance(lease_duration_seconds, bool)
        or not isinstance(lease_duration_seconds, int)
        or lease_duration_seconds <= 0
    ):
        raise ValueError("Lease duration must be a positive integer.")


class PaperProcessingLeaseLostError(RuntimeError):
    """Raised when a database write is attempted by a stale worker."""


@dataclass(frozen=True)
class PaperProcessingLease:
    """Opaque capability required for mutations made during processing."""

    paper_id: UUID
    owner_token: UUID
    fencing_token: int


def acquire_paper_processing_lock(
    paper_id: UUID,
    owner_token: UUID,
    lease_duration_seconds: int = 300,
) -> PaperProcessingLease | None:

    _validate_lease_duration(lease_duration_seconds)

    with get_connection() as connection:
        row = connection.execute(
            """
            INSERT INTO paper_processing_locks (
                paper_id,
                owner_token,
                lease_expires_at
            )
            VALUES (
                %s,
                %s,
                clock_timestamp() + (%s * INTERVAL '1 second')
            )
            ON CONFLICT (paper_id)
            DO UPDATE SET
                owner_token = EXCLUDED.owner_token,
                fencing_token = nextval(
                    'paper_processing_fencing_token_seq'
                ),
                acquired_at = clock_timestamp(),
                lease_expires_at = EXCLUDED.lease_expires_at,
                updated_at = clock_timestamp()
            WHERE paper_processing_locks.lease_expires_at <= clock_timestamp()
            RETURNING fencing_token
            """,
            (paper_id, owner_token, lease_duration_seconds),
        ).fetchone()

    if row is None:
        return None

    return PaperProcessingLease(
        paper_id=paper_id,
        owner_token=owner_token,
        fencing_token=row["fencing_token"],
    )


def renew_paper_processing_lock(
    lease: PaperProcessingLease,
    lease_duration_seconds: int = 300,
) -> bool:

    _validate_lease_duration(lease_duration_seconds)

    with get_connection() as connection:
        row = connection.execute(
            """
            UPDATE paper_processing_locks
            SET
                lease_expires_at =
                    clock_timestamp() + (%s * INTERVAL '1 second'),
                updated_at = clock_timestamp()
            WHERE paper_id = %s
                AND owner_token = %s
                AND fencing_token = %s
                AND lease_expires_at > clock_timestamp()
            RETURNING paper_id
            """,
            (
                lease_duration_seconds,
                lease.paper_id,
                lease.owner_token,
                lease.fencing_token,
            ),
        ).fetchone()

    return row is not None


def release_paper_processing_lock(
    lease: PaperProcessingLease,
) -> bool:

    with get_connection() as connection:
        row = connection.execute(
            """
            DELETE FROM paper_processing_locks
            WHERE paper_id = %s
                AND owner_token = %s
                AND fencing_token = %s
            RETURNING paper_id
            """,
            (
                lease.paper_id,
                lease.owner_token,
                lease.fencing_token,
            ),
        ).fetchone()

    return row is not None


def assert_current_paper_processing_lease(
    connection,
    lease: PaperProcessingLease,
) -> None:
    """Fence a mutation with the active lease in its current transaction.

    Locking the lease row prevents a takeover from interleaving between this
    validation and the following mutation.  Callers must keep the supplied
    connection open until their mutation has completed.
    """
    row = connection.execute(
        """
        SELECT paper_id
        FROM paper_processing_locks
        WHERE paper_id = %s
            AND owner_token = %s
            AND fencing_token = %s
            AND lease_expires_at > clock_timestamp()
        FOR UPDATE
        """,
        (lease.paper_id, lease.owner_token, lease.fencing_token),
    ).fetchone()

    if row is None:
        raise PaperProcessingLeaseLostError(
            f"Paper processing lease is no longer current: {lease.paper_id}"
        )
