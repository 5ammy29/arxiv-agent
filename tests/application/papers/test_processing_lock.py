from unittest.mock import patch
from uuid import uuid4

import pytest

from project.application.papers.processing_lock import (
    PaperProcessingLockError,
    PaperProcessingLockManager,
)
from project.infrastructure.database.paper_locks import PaperProcessingLease


class ImmediateThread:
    """Run the heartbeat deterministically when start() is called."""

    def __init__(self, *, target, **_kwargs):
        self.target = target

    def start(self):
        self.target()

    def join(self, *_args, **_kwargs):
        pass

    def is_alive(self):
        return False


class HeartbeatStopEvent:
    def __init__(self):
        self.wait_calls = 0
        self.set_called = False

    def wait(self, _timeout):
        self.wait_calls += 1
        return self.wait_calls > 1

    def set(self):
        self.set_called = True

    def is_set(self):
        return self.set_called


class LeaseLostEvent:
    def __init__(self):
        self.value = False

    def set(self):
        self.value = True

    def is_set(self):
        return self.value


def immediate_events():
    events = [HeartbeatStopEvent(), LeaseLostEvent()]
    return lambda: events.pop(0)


def manager():
    return PaperProcessingLockManager(
        wait_timeout=1,
        lease_duration=2,
        heartbeat_interval=1,
        retry_interval=0.01,
    )


def test_acquires_renews_and_releases_processing_lease():
    lease = PaperProcessingLease(uuid4(), uuid4(), 7)
    with (
        patch(
            "project.application.papers.processing_lock.acquire_paper_processing_lock",
            return_value=lease,
        ) as acquire,
        patch(
            "project.application.papers.processing_lock.renew_paper_processing_lock",
            return_value=True,
        ) as renew,
        patch(
            "project.application.papers.processing_lock.release_paper_processing_lock",
            return_value=True,
        ) as release,
        patch("project.application.papers.processing_lock.Event", side_effect=immediate_events()),
        patch("project.application.papers.processing_lock.Thread", ImmediateThread),
    ):
        with manager().acquire(lease.paper_id) as active:
            assert active.lease == lease
            active.assert_active()

    acquire.assert_called_once_with(lease.paper_id, acquire.call_args.args[1], 2)
    renew.assert_called_once_with(lease, 2)
    release.assert_called_once_with(lease)


def test_active_lease_contention_times_out_without_releasing():
    paper_id = uuid4()
    with (
        patch(
            "project.application.papers.processing_lock.acquire_paper_processing_lock",
            return_value=None,
        ) as acquire,
        patch(
            "project.application.papers.processing_lock.time.monotonic",
            side_effect=[0, 0, 2, 2],
        ),
        patch("project.application.papers.processing_lock.time.sleep"),
        patch("project.application.papers.processing_lock.release_paper_processing_lock") as release,
    ):
        with pytest.raises(PaperProcessingLockError, match="Timed out"):
            manager().acquire(paper_id).__enter__()

    acquire.assert_called_once()
    release.assert_not_called()


def test_renewal_rejection_cancels_processing_and_releases_lease():
    lease = PaperProcessingLease(uuid4(), uuid4(), 2)
    with (
        patch(
            "project.application.papers.processing_lock.acquire_paper_processing_lock",
            return_value=lease,
        ),
        patch(
            "project.application.papers.processing_lock.renew_paper_processing_lock",
            return_value=False,
        ),
        patch(
            "project.application.papers.processing_lock.release_paper_processing_lock",
            return_value=True,
        ) as release,
        patch("project.application.papers.processing_lock.Event", side_effect=immediate_events()),
        patch("project.application.papers.processing_lock.Thread", ImmediateThread),
    ):
        with pytest.raises(PaperProcessingLockError, match="lease was lost"):
            with manager().acquire(lease.paper_id) as active:
                active.assert_active()

    release.assert_called_once_with(lease)


def test_release_runs_when_processing_raises():
    lease = PaperProcessingLease(uuid4(), uuid4(), 3)
    with (
        patch(
            "project.application.papers.processing_lock.acquire_paper_processing_lock",
            return_value=lease,
        ),
        patch(
            "project.application.papers.processing_lock.renew_paper_processing_lock",
            return_value=True,
        ),
        patch(
            "project.application.papers.processing_lock.release_paper_processing_lock",
            return_value=True,
        ) as release,
        patch("project.application.papers.processing_lock.Event", side_effect=immediate_events()),
        patch("project.application.papers.processing_lock.Thread", ImmediateThread),
    ):
        with pytest.raises(ValueError, match="processing failure"):
            with manager().acquire(lease.paper_id):
                raise ValueError("processing failure")

    release.assert_called_once_with(lease)


def test_thread_startup_failure_releases_acquired_lease():
    lease = PaperProcessingLease(uuid4(), uuid4(), 31)

    class StartFailureThread:
        def __init__(self, **_kwargs):
            pass

        def start(self):
            raise RuntimeError("thread startup failed")

    with (
        patch(
            "project.application.papers.processing_lock.acquire_paper_processing_lock",
            return_value=lease,
        ),
        patch(
            "project.application.papers.processing_lock.release_paper_processing_lock",
            return_value=True,
        ) as release,
        patch("project.application.papers.processing_lock.Thread", StartFailureThread),
    ):
        with pytest.raises(RuntimeError, match="thread startup failed"):
            with manager().acquire(lease.paper_id):
                pass

    release.assert_called_once_with(lease)


def test_acquisition_error_propagates_without_release():
    with (
        patch(
            "project.application.papers.processing_lock.acquire_paper_processing_lock",
            side_effect=RuntimeError("database unavailable"),
        ),
        patch("project.application.papers.processing_lock.release_paper_processing_lock") as release,
    ):
        with pytest.raises(RuntimeError, match="database unavailable"):
            with manager().acquire(uuid4()):
                pass

    release.assert_not_called()


def test_renewal_error_cancels_processing():
    lease = PaperProcessingLease(uuid4(), uuid4(), 4)
    with (
        patch(
            "project.application.papers.processing_lock.acquire_paper_processing_lock",
            return_value=lease,
        ),
        patch(
            "project.application.papers.processing_lock.renew_paper_processing_lock",
            side_effect=RuntimeError("renewal unavailable"),
        ),
        patch(
            "project.application.papers.processing_lock.release_paper_processing_lock",
            return_value=True,
        ),
        patch("project.application.papers.processing_lock.Event", side_effect=immediate_events()),
        patch("project.application.papers.processing_lock.Thread", ImmediateThread),
    ):
        with pytest.raises(PaperProcessingLockError, match="lease was lost"):
            with manager().acquire(lease.paper_id):
                pass
