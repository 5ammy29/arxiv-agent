"""Application-level lease orchestration for mutable paper artifacts."""

import logging
import time

from contextlib import contextmanager
from dataclasses import dataclass
from threading import Event, Thread
from typing import Iterator
from uuid import UUID, uuid4

from project.infrastructure.database.paper_locks import (
    PaperProcessingLease,
    acquire_paper_processing_lock,
    release_paper_processing_lock,
    renew_paper_processing_lock,
)


logger = logging.getLogger(__name__)


class PaperProcessingLockError(RuntimeError):
    """Raised when a paper processing lease cannot be acquired or retained."""


@dataclass(frozen=True)
class ActivePaperProcessingLease:
    """A live lease plus cooperative cancellation state for one worker."""

    lease: PaperProcessingLease
    lease_lost: Event

    def assert_active(self) -> None:
        if self.lease_lost.is_set():
            raise PaperProcessingLockError(
                "Processing lease was lost for paper "
                f"{self.lease.paper_id}."
            )


class PaperProcessingLockManager:
    """Acquire, renew, and release a single-paper processing lease."""

    def __init__(
        self,
        *,
        wait_timeout: int = 600,
        lease_duration: int = 120,
        heartbeat_interval: int = 30,
        retry_interval: float = 2.0,
    ) -> None:
        self._validate_configuration(
            wait_timeout,
            lease_duration,
            heartbeat_interval,
            retry_interval,
        )
        self.wait_timeout = wait_timeout
        self.lease_duration = lease_duration
        self.heartbeat_interval = heartbeat_interval
        self.retry_interval = retry_interval

    @staticmethod
    def _validate_configuration(
        wait_timeout: int,
        lease_duration: int,
        heartbeat_interval: int,
        retry_interval: float,
    ) -> None:
        for name, value in {
            "wait_timeout": wait_timeout,
            "lease_duration": lease_duration,
            "heartbeat_interval": heartbeat_interval,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if heartbeat_interval >= lease_duration:
            raise ValueError(
                "Heartbeat interval must be shorter than lease duration."
            )
        if (
            isinstance(retry_interval, bool)
            or not isinstance(retry_interval, (int, float))
            or retry_interval <= 0
        ):
            raise ValueError("retry_interval must be a positive number.")

    @contextmanager
    def acquire(self, paper_id: UUID) -> Iterator[ActivePaperProcessingLease]:
        owner_token = uuid4()
        deadline = time.monotonic() + self.wait_timeout
        started = time.monotonic()

        while True:
            lease = acquire_paper_processing_lock(
                paper_id,
                owner_token,
                self.lease_duration,
            )
            if lease is not None:
                break
            if time.monotonic() >= deadline:
                logger.warning(
                    "paper_lock_timeout paper_id=%s wait_seconds=%.3f",
                    paper_id,
                    time.monotonic() - started,
                )
                raise PaperProcessingLockError(
                    f"Timed out waiting for paper processing lock: {paper_id}"
                )
            logger.info("paper_lock_waiting paper_id=%s", paper_id)
            time.sleep(self.retry_interval)

        logger.info(
            "paper_lock_acquired paper_id=%s fencing_token=%s wait_seconds=%.3f",
            paper_id,
            lease.fencing_token,
            time.monotonic() - started,
        )
        stop_heartbeat = Event()
        lease_lost = Event()

        def heartbeat() -> None:
            while not stop_heartbeat.wait(self.heartbeat_interval):
                try:
                    if not renew_paper_processing_lock(lease, self.lease_duration):
                        lease_lost.set()
                        logger.error(
                            "paper_lock_lost paper_id=%s fencing_token=%s reason=renewal_rejected",
                            paper_id,
                            lease.fencing_token,
                        )
                        return
                    logger.debug(
                        "paper_lock_renewed paper_id=%s fencing_token=%s",
                        paper_id,
                        lease.fencing_token,
                    )
                except Exception:
                    lease_lost.set()
                    logger.exception(
                        "paper_lock_lost paper_id=%s fencing_token=%s reason=renewal_error",
                        paper_id,
                        lease.fencing_token,
                    )
                    return

        heartbeat_thread: Thread | None = None
        heartbeat_started = False
        try:
            heartbeat_thread = Thread(
                target=heartbeat,
                name=f"paper-lock-{paper_id}",
                daemon=True,
            )
            heartbeat_thread.start()
            heartbeat_started = True
            active_lease = ActivePaperProcessingLease(lease, lease_lost)
            yield active_lease
            active_lease.assert_active()
        finally:
            stop_heartbeat.set()
            if heartbeat_thread is not None and heartbeat_started:
                heartbeat_thread.join(timeout=self.heartbeat_interval + 1)
                if heartbeat_thread.is_alive():
                    logger.warning(
                        "paper_lock_heartbeat_shutdown_timeout paper_id=%s fencing_token=%s",
                        paper_id,
                        lease.fencing_token,
                    )
            try:
                if release_paper_processing_lock(lease):
                    logger.info(
                        "paper_lock_released paper_id=%s fencing_token=%s held_seconds=%.3f",
                        paper_id,
                        lease.fencing_token,
                        time.monotonic() - started,
                    )
                else:
                    logger.warning(
                        "paper_lock_release_skipped paper_id=%s fencing_token=%s",
                        paper_id,
                        lease.fencing_token,
                    )
            except Exception:
                logger.exception(
                    "paper_lock_release_failed paper_id=%s fencing_token=%s",
                    paper_id,
                    lease.fencing_token,
                )
