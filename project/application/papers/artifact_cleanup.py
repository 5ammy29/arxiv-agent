"""Safe, database-authoritative cleanup of superseded paper artifacts."""

import logging

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Callable

from project.infrastructure.database.papers import (
    get_active_processing_artifact_prefixes,
    get_referenced_artifact_paths,
)
from project.infrastructure.storage.artifacts import list_artifact_objects
from project.infrastructure.storage.papers import delete_paper_pdf
from project.infrastructure.storage.vector_indexes import delete_vector_index


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ArtifactCleanupResult:
    scanned: int = 0
    deleted: int = 0
    skipped_referenced: int = 0
    skipped_active: int = 0
    skipped_grace_period: int = 0
    skipped_unknown_age: int = 0
    failures: int = 0


class ArtifactCleanupService:
    """Delete only old, unreferenced objects in application-owned prefixes."""

    def __init__(
        self,
        *,
        grace_period: timedelta = timedelta(hours=24),
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if grace_period.total_seconds() < 0:
            raise ValueError("grace_period must not be negative.")
        self.grace_period = grace_period
        self.clock = clock or (lambda: datetime.now(UTC))

    def cleanup(self) -> ArtifactCleanupResult:
        started = self.clock()
        referenced_paths = get_referenced_artifact_paths()
        active_prefixes = get_active_processing_artifact_prefixes()
        candidates = [
            *(self._with_kind("pdf", list_artifact_objects("papers"))),
            *(self._with_kind("index", list_artifact_objects("vector_indexes"))),
        ]
        totals = ArtifactCleanupResult(scanned=len(candidates))

        for kind, object_ in candidates:
            path = object_["path"]
            if path in referenced_paths:
                totals = self._increment(totals, "skipped_referenced")
                continue
            if any(path.startswith(prefix) for prefix in active_prefixes):
                totals = self._increment(totals, "skipped_active")
                continue

            updated_at = object_.get("updated_at")
            if updated_at is None:
                totals = self._increment(totals, "skipped_unknown_age")
                logger.warning("artifact_cleanup_skipped path=%s reason=unknown_age", path)
                continue
            if updated_at.tzinfo is None:
                updated_at = updated_at.replace(tzinfo=UTC)
            if started - updated_at < self.grace_period:
                totals = self._increment(totals, "skipped_grace_period")
                continue

            # A previously-unreferenced object can become active while a long
            # cleanup inventory is being processed. Recheck at the last safe
            # point before deletion; database failures abort rather than risk
            # deleting an object whose state cannot be verified.
            latest_references = get_referenced_artifact_paths()
            latest_active_prefixes = get_active_processing_artifact_prefixes()
            if path in latest_references:
                totals = self._increment(totals, "skipped_referenced")
                logger.info(
                    "artifact_cleanup_skipped path=%s reason=became_referenced",
                    path,
                )
                continue
            if any(path.startswith(prefix) for prefix in latest_active_prefixes):
                totals = self._increment(totals, "skipped_active")
                logger.info(
                    "artifact_cleanup_skipped path=%s reason=processing_active",
                    path,
                )
                continue

            try:
                (delete_paper_pdf if kind == "pdf" else delete_vector_index)(path)
            except Exception as error:
                if not _is_missing_storage_object(error):
                    totals = self._increment(totals, "failures")
                    logger.exception("artifact_cleanup_failed path=%s kind=%s", path, kind)
                    continue
            totals = self._increment(totals, "deleted")
            logger.info("artifact_cleanup_deleted path=%s kind=%s", path, kind)

        logger.info(
            "artifact_cleanup_complete scanned=%s deleted=%s referenced=%s active=%s grace=%s unknown_age=%s failures=%s duration_seconds=%.3f",
            totals.scanned,
            totals.deleted,
            totals.skipped_referenced,
            totals.skipped_active,
            totals.skipped_grace_period,
            totals.skipped_unknown_age,
            totals.failures,
            (self.clock() - started).total_seconds(),
        )
        return totals

    @staticmethod
    def _with_kind(kind: str, objects: list[dict]) -> list[tuple[str, dict]]:
        return [(kind, object_) for object_ in objects]

    @staticmethod
    def _increment(result: ArtifactCleanupResult, field: str) -> ArtifactCleanupResult:
        values = result.__dict__.copy()
        values[field] += 1
        return ArtifactCleanupResult(**values)


def _is_missing_storage_object(error: Exception) -> bool:
    return isinstance(error, FileNotFoundError) or any(
        getattr(error, attribute, None) == 404
        for attribute in ("status", "status_code", "code")
    )
