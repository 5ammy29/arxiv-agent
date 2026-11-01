"""Real PostgreSQL tests for lease ownership and transactional fencing."""

from threading import Barrier, Event, Thread
from uuid import uuid4

import psycopg
import pytest
from psycopg.rows import dict_row

from project.infrastructure.database.paper_locks import (
    PaperProcessingLeaseLostError,
    acquire_paper_processing_lock,
    assert_current_paper_processing_lease,
    release_paper_processing_lock,
    renew_paper_processing_lock,
)
from project.infrastructure.database.papers import update_paper_status


pytestmark = pytest.mark.integration


def _expire_lease(database_url: str, paper_id) -> None:
    with psycopg.connect(database_url) as connection:
        connection.execute(
            """
            UPDATE paper_processing_locks
            SET lease_expires_at = acquired_at + INTERVAL '1 microsecond'
            WHERE paper_id = %s
            """,
            (paper_id,),
        )


def test_same_paper_lease_contention_and_release(
    integration_paper,
):
    paper_id, _ = integration_paper()
    first = acquire_paper_processing_lock(paper_id, uuid4(), 60)

    assert first is not None
    assert acquire_paper_processing_lock(paper_id, uuid4(), 60) is None
    assert release_paper_processing_lock(first) is True

    second = acquire_paper_processing_lock(paper_id, uuid4(), 60)
    assert second is not None
    assert second.fencing_token > first.fencing_token


def test_expired_lease_takeover_rejects_stale_mutation(
    integration_database_url,
    integration_paper,
):
    paper_id, arxiv_id = integration_paper()
    first = acquire_paper_processing_lock(paper_id, uuid4(), 60)
    assert first is not None
    _expire_lease(integration_database_url, paper_id)

    second = acquire_paper_processing_lock(paper_id, uuid4(), 60)
    assert second is not None
    assert second.fencing_token > first.fencing_token

    update_paper_status(arxiv_id, "ready", lease=second)
    with pytest.raises(PaperProcessingLeaseLostError):
        update_paper_status(arxiv_id, "failed", lease=first)

    with psycopg.connect(integration_database_url, row_factory=dict_row) as connection:
        row = connection.execute(
            "SELECT processing_status FROM papers WHERE id = %s", (paper_id,)
        ).fetchone()
    assert row["processing_status"] == "ready"


def test_concurrent_acquisition_has_exactly_one_winner(integration_paper):
    paper_id, _ = integration_paper()
    barrier = Barrier(2)
    leases = []

    def acquire() -> None:
        barrier.wait()
        leases.append(acquire_paper_processing_lock(paper_id, uuid4(), 60))

    workers = [Thread(target=acquire), Thread(target=acquire)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    winners = [lease for lease in leases if lease is not None]
    assert len(winners) == 1


def test_fenced_transaction_blocks_takeover_until_mutation_commits(
    integration_database_url,
    integration_paper,
):
    paper_id, _ = integration_paper()
    lease = acquire_paper_processing_lock(paper_id, uuid4(), 60)
    assert lease is not None

    attempted = Event()
    completed = Event()
    result = {}

    def take_over() -> None:
        attempted.set()
        result["lease"] = acquire_paper_processing_lock(paper_id, uuid4(), 60)
        completed.set()

    with psycopg.connect(integration_database_url, row_factory=dict_row) as connection:
        assert_current_paper_processing_lease(connection, lease)
        # This is the interval protected by SELECT ... FOR UPDATE: after
        # validation, a real paper mutation executes before a takeover can
        # change ownership of the lock row.
        connection.execute(
            "UPDATE papers SET title = title WHERE id = %s", (paper_id,)
        )
        connection.execute(
            """
            UPDATE paper_processing_locks
            SET lease_expires_at = acquired_at + INTERVAL '1 microsecond'
            WHERE paper_id = %s
            """,
            (paper_id,),
        )
        worker = Thread(target=take_over)
        worker.start()
        assert attempted.wait(timeout=1)
        # Do not let the contender proceed until the fenced mutation commits.
        assert not completed.is_set()

    assert completed.wait(timeout=1)
    worker.join()
    assert result["lease"] is not None


def test_renewal_expiry_and_stale_release_rules(
    integration_database_url,
    integration_paper,
):
    paper_id, _ = integration_paper()
    first = acquire_paper_processing_lock(paper_id, uuid4(), 60)
    assert first is not None
    assert renew_paper_processing_lock(first, 60) is True

    _expire_lease(integration_database_url, paper_id)
    assert renew_paper_processing_lock(first, 60) is False
    second = acquire_paper_processing_lock(paper_id, uuid4(), 60)
    assert second is not None
    assert renew_paper_processing_lock(first, 60) is False
    assert release_paper_processing_lock(first) is False
    assert release_paper_processing_lock(second) is True
