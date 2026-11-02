"""Fixtures for opt-in tests against an isolated local PostgreSQL database."""

import os
from collections.abc import Callable, Iterator
from urllib.parse import urlparse
from uuid import UUID, uuid4

import psycopg
import pytest
from psycopg.rows import dict_row

from project.infrastructure.database.connection import close_connection_pool


LOCAL_DATABASE_HOSTS = {"localhost", "127.0.0.1", "::1"}


@pytest.fixture
def integration_database_url(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """Provide an explicitly configured, local integration database only.

    This guard prevents the real-concurrency tests from ever using the normal
    developer or hosted Supabase database by accident.
    """
    database_url = os.getenv("ARXIV_AGENT_TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("ARXIV_AGENT_TEST_DATABASE_URL is not configured")

    parsed = urlparse(database_url)
    if parsed.hostname not in LOCAL_DATABASE_HOSTS:
        pytest.skip("Integration tests require an explicitly local PostgreSQL URL")

    try:
        with psycopg.connect(database_url, row_factory=dict_row) as connection:
            required_tables = connection.execute(
                """
                SELECT to_regclass('public.papers') AS papers,
                       to_regclass('public.paper_processing_locks') AS locks,
                       to_regclass('public.paper_artifacts') AS artifacts,
                       to_regclass('public.paper_pages') AS pages,
                       to_regclass('public.paper_chunks') AS chunks,
                       to_regclass('public.paper_vector_indexes') AS indexes
                """
            ).fetchone()
    except psycopg.Error as error:
        pytest.skip(f"Local integration database is unavailable: {error}")

    if not required_tables or not all(required_tables.values()):
        pytest.skip("Local integration database has not applied all migrations")

    monkeypatch.setenv("DATABASE_URL", database_url)
    close_connection_pool()
    try:
        yield database_url
    finally:
        close_connection_pool()


@pytest.fixture
def integration_paper(
    integration_database_url: str,
) -> Iterator[Callable[[], tuple[UUID, str]]]:
    """Create disposable rows; deleting papers cascades test lock metadata."""
    paper_ids: list[UUID] = []

    def create() -> tuple[UUID, str]:
        paper_id = uuid4()
        arxiv_id = f"test-{paper_id.hex}"
        with psycopg.connect(
            integration_database_url,
            row_factory=dict_row,
        ) as connection:
            connection.execute(
                """
                INSERT INTO papers (
                    id, arxiv_id, version, title, authors, abstract,
                    published_at, arxiv_updated_at, url, pdf_url, categories
                )
                VALUES (
                    %s, %s, 1, 'integration test paper', '[]'::jsonb,
                    'integration test abstract', clock_timestamp(),
                    clock_timestamp(), 'https://example.invalid/abs',
                    'https://example.invalid/pdf', '[]'::jsonb
                )
                """,
                (paper_id, arxiv_id),
            )
        paper_ids.append(paper_id)
        return paper_id, f"{arxiv_id}v1"

    yield create

    with psycopg.connect(integration_database_url) as connection:
        for paper_id in paper_ids:
            connection.execute("DELETE FROM papers WHERE id = %s", (paper_id,))
