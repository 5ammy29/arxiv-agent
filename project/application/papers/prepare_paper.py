import logging
import time

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from uuid import UUID

from project.application.papers.processing_lock import (
    ActivePaperProcessingLease,
    PaperProcessingLockError,
    PaperProcessingLockManager,
)
from project.infrastructure.database.paper_locks import PaperProcessingLeaseLostError
from project.infrastructure.database.papers import (
    get_paper_artifact,
    get_paper_content,
    replace_paper_pdf,
    save_paper_content,
    update_paper_status,
)
from project.infrastructure.storage.papers import (
    delete_paper_pdf,
    download_paper_pdf,
    upload_paper_pdf,
)
from project.infrastructure.storage.vector_indexes import delete_vector_index
from project.processing.chunker import TextChunk, TextChunker
from project.processing.fingerprints import content_fingerprint
from project.processing.pdf import (
    PDFDownloadError,
    PDFPage,
    PDFParseError,
    PDFProcessor,
)


logger = logging.getLogger(__name__)


@dataclass
class PaperPreparationResult:
    pdf_path: str | None = None
    chunks: list[TextChunk] | None = None
    content_fingerprint: str | None = None
    chunking_fingerprint: str | None = None
    error: str | None = None


def _is_missing_storage_object(error: Exception) -> bool:
    return isinstance(error, FileNotFoundError) or any(
        getattr(error, attribute, None) == 404
        for attribute in ("status", "status_code", "code")
    )


class PaperPreparationService:
    """Prepare one paper under a lease and fence every mutable write."""

    def __init__(
        self,
        pdf_processor: PDFProcessor,
        chunker: TextChunker,
        *,
        lock_manager: PaperProcessingLockManager | None = None,
    ) -> None:
        self.pdf_processor = pdf_processor
        self.chunker = chunker
        self.lock_manager = lock_manager or PaperProcessingLockManager()

    def _update_status(
        self,
        arxiv_id: str,
        status: str,
        active_lease: ActivePaperProcessingLease,
    ) -> None:
        active_lease.assert_active()
        update_paper_status(arxiv_id, status, lease=active_lease.lease)
        logger.info(
            "paper_processing_status paper_id=%s status=%s",
            active_lease.lease.paper_id,
            status,
        )

    def _mark_failed(
        self,
        arxiv_id: str,
        active_lease: ActivePaperProcessingLease,
    ) -> None:
        try:
            self._update_status(arxiv_id, "failed", active_lease)
        except (PaperProcessingLockError, PaperProcessingLeaseLostError):
            logger.warning(
                "paper_processing_failure_status_skipped paper_id=%s reason=lease_lost",
                active_lease.lease.paper_id,
            )
        except Exception:
            logger.exception(
                "paper_processing_failure_status_failed paper_id=%s",
                active_lease.lease.paper_id,
            )

    def _failure(
        self,
        arxiv_id: str,
        message: str,
        active_lease: ActivePaperProcessingLease,
    ) -> PaperPreparationResult:
        self._mark_failed(arxiv_id, active_lease)
        return PaperPreparationResult(error=message)

    @staticmethod
    def _cleanup_storage_object(
        delete_function: Callable[[str], None],
        storage_path: str | None,
        description: str,
    ) -> None:
        if not isinstance(storage_path, str) or not storage_path:
            return
        try:
            delete_function(storage_path)
            logger.info("artifact_cleanup_deleted path=%s reason=%s", storage_path, description)
        except Exception as error:
            if _is_missing_storage_object(error):
                logger.info("artifact_cleanup_already_missing path=%s", storage_path)
            else:
                logger.warning(
                    "artifact_cleanup_failed path=%s reason=%s",
                    storage_path,
                    description,
                    exc_info=True,
                )

    def _cleanup_invalidated_index(self, storage_path: str | None) -> None:
        if storage_path:
            logger.info(
                "artifact_cleanup_deferred path=%s reason=invalidated_index",
                storage_path,
            )

    def _cleanup_replaced_artifacts(self, replacement: dict[str, str | None]) -> None:
        if replacement.get("old_pdf_storage_path"):
            logger.info(
                "artifact_cleanup_deferred path=%s reason=superseded_pdf",
                replacement["old_pdf_storage_path"],
            )
        self._cleanup_invalidated_index(replacement.get("old_index_storage_path"))

    def prepare(
        self,
        pdf_url: str,
        arxiv_id: str,
        paper_id: UUID,
    ) -> PaperPreparationResult:
        """Run download and text preparation under one lease."""
        started = time.monotonic()
        logger.info("paper_processing_started paper_id=%s", paper_id)
        try:
            with self.lock_manager.acquire(paper_id) as active_lease:
                download_result = self._download(pdf_url, arxiv_id, paper_id, active_lease)
                if download_result.error:
                    return download_result
                content_result = self._parse_and_chunk(
                    download_result.pdf_path or "", arxiv_id, paper_id, active_lease
                )
                if content_result.error:
                    return PaperPreparationResult(
                        pdf_path=download_result.pdf_path,
                        chunks=content_result.chunks,
                        error=content_result.error,
                    )
                logger.info(
                    "paper_processing_completed paper_id=%s outcome=ready duration_seconds=%.3f",
                    paper_id,
                    time.monotonic() - started,
                )
                return PaperPreparationResult(
                    pdf_path=download_result.pdf_path,
                    chunks=content_result.chunks,
                    content_fingerprint=content_result.content_fingerprint,
                    chunking_fingerprint=content_result.chunking_fingerprint,
                )
        except (PaperProcessingLockError, PaperProcessingLeaseLostError) as error:
            logger.warning("paper_processing_stopped paper_id=%s reason=%s", paper_id, error)
            return PaperPreparationResult(error=str(error))
        except Exception:
            logger.exception("paper_processing_failed paper_id=%s", paper_id)
            return PaperPreparationResult(error=f"Failed to prepare paper {arxiv_id}")

    # Standalone stages retain their API, but cannot bypass a lease.
    def download(self, pdf_url: str, arxiv_id: str, paper_id: UUID) -> PaperPreparationResult:
        try:
            with self.lock_manager.acquire(paper_id) as active_lease:
                return self._download(pdf_url, arxiv_id, paper_id, active_lease)
        except (PaperProcessingLockError, PaperProcessingLeaseLostError) as error:
            return PaperPreparationResult(error=str(error))

    def parse_and_chunk(self, pdf_path: str, arxiv_id: str, paper_id: UUID) -> PaperPreparationResult:
        try:
            with self.lock_manager.acquire(paper_id) as active_lease:
                return self._parse_and_chunk(pdf_path, arxiv_id, paper_id, active_lease)
        except (PaperProcessingLockError, PaperProcessingLeaseLostError) as error:
            return PaperPreparationResult(error=str(error))

    def _download(
        self,
        pdf_url: str,
        arxiv_id: str,
        paper_id: UUID,
        active_lease: ActivePaperProcessingLease,
    ) -> PaperPreparationResult:
        try:
            self._update_status(arxiv_id, "downloading", active_lease)
            artifact = get_paper_artifact(paper_id)
            if artifact and artifact.get("pdf_storage_path"):
                pdf_path = self.pdf_processor.output_dir / f"{arxiv_id}.pdf"
                try:
                    download_paper_pdf(artifact["pdf_storage_path"], pdf_path)
                    self.pdf_processor.parse_pdf(pdf_path)
                    active_lease.assert_active()
                    logger.info("paper_pdf_cache_hit paper_id=%s", paper_id)
                    return PaperPreparationResult(pdf_path=str(pdf_path))
                except PDFParseError:
                    logger.warning("paper_pdf_cache_invalid paper_id=%s", paper_id)
                    pdf_path.unlink(missing_ok=True)
                except Exception as error:
                    pdf_path.unlink(missing_ok=True)
                    if not _is_missing_storage_object(error):
                        logger.exception("paper_pdf_cache_download_failed paper_id=%s", paper_id)
                        return self._failure(
                            arxiv_id, f"Failed to retrieve stored PDF for {arxiv_id}", active_lease
                        )
                    logger.warning("paper_pdf_cache_missing paper_id=%s", paper_id)

            active_lease.assert_active()
            pdf_path = self.pdf_processor.download_pdf(pdf_url, arxiv_id)
            active_lease.assert_active()
            uploaded_storage_path = upload_paper_pdf(pdf_path, arxiv_id)
            logger.info("paper_pdf_uploaded paper_id=%s path=%s", paper_id, uploaded_storage_path)
            try:
                active_lease.assert_active()
                replacement = replace_paper_pdf(
                    paper_id, uploaded_storage_path, lease=active_lease.lease
                )
            except Exception:
                self._cleanup_storage_object(
                    delete_paper_pdf, uploaded_storage_path, "unreferenced_replacement_pdf"
                )
                raise
            self._cleanup_replaced_artifacts(replacement)
            return PaperPreparationResult(pdf_path=str(pdf_path))
        except (PaperProcessingLockError, PaperProcessingLeaseLostError):
            raise
        except PDFDownloadError:
            logger.exception("paper_pdf_download_failed paper_id=%s", paper_id)
            return self._failure(
                arxiv_id, f"Failed to download PDF for {arxiv_id}", active_lease
            )
        except Exception:
            logger.exception("paper_pdf_persist_failed paper_id=%s", paper_id)
            return self._failure(
                arxiv_id, f"Failed to store PDF for {arxiv_id}", active_lease
            )

    @staticmethod
    def _pages_from_cache(rows: list[dict[str, object]]) -> list[PDFPage] | None:
        if not rows:
            return None
        pages: list[PDFPage] = []
        for expected_number, row in enumerate(rows, start=1):
            page_number = row.get("page_number")
            text = row.get("text")
            if page_number != expected_number or not isinstance(text, str):
                return None
            pages.append(PDFPage(page=page_number, text=text))
        return pages if any(page.text.strip() for page in pages) else None

    @staticmethod
    def _chunks_from_cache(
        rows: list[dict[str, object]], pages: list[PDFPage]
    ) -> list[TextChunk] | None:
        if not rows:
            return None
        page_numbers = {page.page for page in pages}
        chunks: list[TextChunk] = []
        for expected_id, row in enumerate(rows):
            chunk_id = row.get("chunk_id")
            page_number = row.get("page_number")
            text = row.get("text")
            if (
                chunk_id != expected_id
                or page_number not in page_numbers
                or not isinstance(text, str)
                or not text.strip()
            ):
                return None
            chunks.append(TextChunk(chunk_id=chunk_id, page=page_number, text=text))
        return chunks

    def _parse_and_chunk(
        self,
        pdf_path: str,
        arxiv_id: str,
        paper_id: UUID,
        active_lease: ActivePaperProcessingLease,
    ) -> PaperPreparationResult:
        try:
            cached_content = get_paper_content(paper_id)
            cached_pages = self._pages_from_cache(cached_content["pages"])
            config_fingerprint = self.chunker.configuration_fingerprint
            artifact = get_paper_artifact(paper_id) if cached_pages else None
            content_is_valid = (
                artifact is not None
                and isinstance(artifact.get("content_fingerprint"), str)
                and bool(artifact["content_fingerprint"])
                and cached_pages is not None
                and content_fingerprint(cached_pages) == artifact["content_fingerprint"]
            )
            chunks = None
            if content_is_valid:
                cached_chunks = self._chunks_from_cache(cached_content["chunks"], cached_pages)
                if cached_chunks is not None and artifact.get("chunking_fingerprint") == config_fingerprint:
                    active_lease.assert_active()
                    expected_chunks = self.chunker.chunk(cached_pages)
                    if cached_chunks == expected_chunks:
                        self._update_status(arxiv_id, "ready", active_lease)
                        logger.info("paper_content_cache_hit paper_id=%s", paper_id)
                        return PaperPreparationResult(
                            chunks=cached_chunks,
                            content_fingerprint=artifact["content_fingerprint"],
                            chunking_fingerprint=config_fingerprint,
                        )
                    chunks = expected_chunks
                pages = cached_pages
            else:
                try:
                    active_lease.assert_active()
                    pages = self.pdf_processor.parse_pdf(Path(pdf_path))
                    active_lease.assert_active()
                except PDFParseError:
                    logger.exception("paper_pdf_parse_failed paper_id=%s", paper_id)
                    return self._failure(arxiv_id, "Failed to parse PDF", active_lease)

            self._update_status(arxiv_id, "parsing", active_lease)
            fingerprint = content_fingerprint(pages)
            if chunks is None:
                try:
                    active_lease.assert_active()
                    chunks = self.chunker.chunk(pages)
                    active_lease.assert_active()
                except Exception:
                    logger.exception("paper_chunk_generation_failed paper_id=%s", paper_id)
                    return self._failure(
                        arxiv_id, "Failed to generate paper chunks", active_lease
                    )
            if not chunks:
                self._mark_failed(arxiv_id, active_lease)
                return PaperPreparationResult(chunks=[], error="No usable text was found in the paper")
            active_lease.assert_active()
            stale_index_path = save_paper_content(
                paper_id,
                [{"page_number": page.page, "text": page.text} for page in pages],
                [
                    {"chunk_id": chunk.chunk_id, "page_number": chunk.page, "text": chunk.text}
                    for chunk in chunks
                ],
                fingerprint,
                config_fingerprint,
                lease=active_lease.lease,
            )
            self._cleanup_invalidated_index(stale_index_path)
            self._update_status(arxiv_id, "ready", active_lease)
            logger.info("paper_content_persisted paper_id=%s", paper_id)
            return PaperPreparationResult(
                chunks=chunks,
                content_fingerprint=fingerprint,
                chunking_fingerprint=config_fingerprint,
            )
        except (PaperProcessingLockError, PaperProcessingLeaseLostError):
            raise
        except Exception:
            logger.exception("paper_content_processing_failed paper_id=%s", paper_id)
            return self._failure(arxiv_id, "Failed to prepare paper content", active_lease)
