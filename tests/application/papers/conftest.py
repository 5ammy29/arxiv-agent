from contextlib import contextmanager
from threading import Event
from uuid import uuid4

import pytest

from project.application.papers.processing_lock import ActivePaperProcessingLease
from project.infrastructure.database.paper_locks import PaperProcessingLease


@pytest.fixture(autouse=True)
def fake_processing_lease_manager(monkeypatch, request):
    """Keep unit tests focused on preparation logic, not PostgreSQL leases."""
    if not request.module.__name__.endswith(
        ("test_prepare_paper", "test_prepare_paper_failures")
    ):
        yield
        return

    @contextmanager
    def acquire(_self, paper_id):
        yield ActivePaperProcessingLease(
            PaperProcessingLease(paper_id, uuid4(), 1),
            Event(),
        )

    monkeypatch.setattr(
        "project.application.papers.processing_lock.PaperProcessingLockManager.acquire",
        acquire,
    )
    yield
