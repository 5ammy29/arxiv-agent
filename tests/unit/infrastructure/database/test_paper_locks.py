from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from project.infrastructure.database.paper_locks import (
    PaperProcessingLease,
    PaperProcessingLeaseLostError,
    acquire_paper_processing_lock,
    assert_current_paper_processing_lease,
    release_paper_processing_lock,
    renew_paper_processing_lock,
)


def make_connection(row):
    connection = MagicMock()
    connection.execute.return_value.fetchone.return_value = row
    context = MagicMock()
    context.__enter__.return_value = connection
    return connection, context


@patch("project.infrastructure.database.paper_locks.get_connection")
def test_acquire_returns_fenced_lease(mock_get_connection):
    paper_id, owner_token = uuid4(), uuid4()
    connection, context = make_connection({"fencing_token": 8})
    mock_get_connection.return_value = context

    lease = acquire_paper_processing_lock(paper_id, owner_token)

    assert lease == PaperProcessingLease(paper_id, owner_token, 8)
    query, params = connection.execute.call_args.args
    assert "ON CONFLICT (paper_id)" in query
    assert "fencing_token = nextval" in query
    assert "clock_timestamp()" in query
    assert params == (paper_id, owner_token, 300)


@patch("project.infrastructure.database.paper_locks.get_connection")
def test_acquire_returns_none_for_active_lock(mock_get_connection):
    connection, context = make_connection(None)
    mock_get_connection.return_value = context
    assert acquire_paper_processing_lock(uuid4(), uuid4()) is None


@patch("project.infrastructure.database.paper_locks.get_connection")
def test_renew_requires_full_lease_identity(mock_get_connection):
    lease = PaperProcessingLease(uuid4(), uuid4(), 4)
    connection, context = make_connection({"paper_id": lease.paper_id})
    mock_get_connection.return_value = context

    assert renew_paper_processing_lock(lease) is True
    query, params = connection.execute.call_args.args
    assert "fencing_token = %s" in query
    assert params == (300, lease.paper_id, lease.owner_token, 4)


@patch("project.infrastructure.database.paper_locks.get_connection")
def test_release_requires_full_lease_identity(mock_get_connection):
    lease = PaperProcessingLease(uuid4(), uuid4(), 4)
    connection, context = make_connection({"paper_id": lease.paper_id})
    mock_get_connection.return_value = context

    assert release_paper_processing_lock(lease) is True
    query, params = connection.execute.call_args.args
    assert "fencing_token = %s" in query
    assert params == (lease.paper_id, lease.owner_token, 4)


def test_fencing_rejects_stale_lease():
    connection = MagicMock()
    connection.execute.return_value.fetchone.return_value = None

    with pytest.raises(PaperProcessingLeaseLostError):
        assert_current_paper_processing_lease(
            connection,
            PaperProcessingLease(uuid4(), uuid4(), 1),
        )


def test_fencing_locks_row_before_mutation():
    lease = PaperProcessingLease(uuid4(), uuid4(), 3)
    connection = MagicMock()
    connection.execute.return_value.fetchone.return_value = {"paper_id": lease.paper_id}

    assert_current_paper_processing_lease(connection, lease)

    query, params = connection.execute.call_args.args
    assert "FOR UPDATE" in query
    assert "lease_expires_at > clock_timestamp()" in query
    assert params == (lease.paper_id, lease.owner_token, lease.fencing_token)


@pytest.mark.parametrize("duration", [0, -1, True, 1.5, "300"])
def test_invalid_lease_duration_is_rejected(duration):
    with pytest.raises(ValueError, match="positive integer"):
        acquire_paper_processing_lock(uuid4(), uuid4(), duration)


@pytest.mark.parametrize("duration", [0, -1, True, 1.5, "300"])
def test_invalid_renewal_duration_is_rejected(duration):
    with pytest.raises(ValueError, match="positive integer"):
        renew_paper_processing_lock(PaperProcessingLease(uuid4(), uuid4(), 1), duration)
