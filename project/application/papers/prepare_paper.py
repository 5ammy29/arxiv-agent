import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from uuid import UUID

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
    def __init__(self, pdf_processor: PDFProcessor, chunker: TextChunker) -> None:
        self.pdf_processor = pdf_processor
        self.chunker = chunker

    @staticmethod
    def _mark_failed(arxiv_id: str) -> None:
        """Best-effort status update; never hide the original failure."""
        try:
            update_paper_status(arxiv_id, "failed")
        except Exception:
            logger.exception(
                "Could not mark paper %s as failed. "
                "The database may be unavailable.",
                arxiv_id,
            )

    @classmethod
    def _failure(
        cls,
        arxiv_id: str,
        message: str,
    ) -> PaperPreparationResult:
        cls._mark_failed(arxiv_id)
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
        except Exception as error:
            if _is_missing_storage_object(error):
                logger.info(
                    "%s was already missing: %s",
                    description,
                    storage_path,
                )
            else:
                logger.warning(
                    "Failed to clean up %s: %s",
                    description,
                    storage_path,
                    exc_info=True,
                )

    def _cleanup_invalidated_index(self, storage_path: str | None) -> None:
        self._cleanup_storage_object(
            delete_vector_index,
            storage_path,
            "invalidated FAISS index",
        )

    def _cleanup_replaced_artifacts(
        self,
        replacement: dict[str, str | None],
    ) -> None:
        self._cleanup_storage_object(
            delete_paper_pdf,
            replacement.get("old_pdf_storage_path"),
            "superseded PDF",
        )
        self._cleanup_invalidated_index(
            replacement.get("old_index_storage_path")
        )

    def download(
        self,
        pdf_url: str,
        arxiv_id: str,
        paper_id: UUID,
    ) -> PaperPreparationResult:
        try:
            update_paper_status(arxiv_id, "downloading")

            artifact = get_paper_artifact(paper_id)

            if artifact and artifact.get("pdf_storage_path"):
                pdf_path = self.pdf_processor.output_dir / f"{arxiv_id}.pdf"

                try:
                    download_paper_pdf(
                        artifact["pdf_storage_path"],
                        pdf_path,
                    )
                    self.pdf_processor.parse_pdf(pdf_path)

                    logger.info("Reusing cached PDF for paper %s.", arxiv_id)
                    return PaperPreparationResult(pdf_path=str(pdf_path))

                except PDFParseError:
                    logger.warning(
                        "Cached PDF for paper %s is invalid; downloading again.",
                        arxiv_id,
                    )
                    pdf_path.unlink(missing_ok=True)

                except Exception as error:
                    pdf_path.unlink(missing_ok=True)

                    if not _is_missing_storage_object(error):
                        logger.exception(
                            "Failed to retrieve cached PDF for paper %s.",
                            arxiv_id,
                        )
                        return self._failure(
                            arxiv_id,
                            f"Failed to retrieve stored PDF for {arxiv_id}",
                        )

                    logger.warning(
                        "Cached PDF for paper %s is missing; downloading again.",
                        arxiv_id,
                    )

            uploaded_storage_path = None

            try:
                pdf_path = self.pdf_processor.download_pdf(pdf_url, arxiv_id)

                uploaded_storage_path = upload_paper_pdf(
                    pdf_path,
                    arxiv_id,
                )

                try:
                    replacement = replace_paper_pdf(
                        paper_id,
                        uploaded_storage_path,
                    )
                except Exception:
                    self._cleanup_storage_object(
                        delete_paper_pdf,
                        uploaded_storage_path,
                        "unreferenced replacement PDF",
                    )
                    raise

                self._cleanup_replaced_artifacts(replacement)

                logger.info("Successfully stored PDF for paper %s.", arxiv_id)

                return PaperPreparationResult(pdf_path=str(pdf_path))

            except PDFDownloadError:
                logger.exception("PDF download failed for paper %s.", arxiv_id)
                return self._failure(
                    arxiv_id,
                    f"Failed to download PDF for {arxiv_id}",
                )

            except Exception:
                logger.exception(
                    "Failed to download or persist PDF for paper %s.",
                    arxiv_id,
                )
                return self._failure(
                    arxiv_id,
                    f"Failed to store PDF for {arxiv_id}",
                )

        except Exception:
            logger.exception(
                "Unexpected failure during PDF preparation for paper %s.",
                arxiv_id,
            )
            return self._failure(
                arxiv_id,
                f"Failed to prepare PDF for {arxiv_id}",
            )

    @staticmethod
    def _pages_from_cache(
        rows: list[dict[str, object]],
    ) -> list[PDFPage] | None:
        if not rows:
            return None

        pages: list[PDFPage] = []

        for expected_number, row in enumerate(rows, start=1):
            page_number = row.get("page_number")
            text = row.get("text")

            if (
                page_number != expected_number
                or not isinstance(text, str)
            ):
                return None

            pages.append(
                PDFPage(page=page_number, text=text)
            )

        return pages if any(page.text.strip() for page in pages) else None

    @staticmethod
    def _chunks_from_cache(
        rows: list[dict[str, object]],
        pages: list[PDFPage],
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

            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    page=page_number,
                    text=text,
                )
            )

        return chunks

    def parse_and_chunk(
        self,
        pdf_path: str,
        arxiv_id: str,
        paper_id: UUID,
    ) -> PaperPreparationResult:
        try:
            cached_content = get_paper_content(paper_id)

            cached_pages = self._pages_from_cache(
                cached_content["pages"]
            )

            config_fingerprint = self.chunker.configuration_fingerprint

            artifact = (
                get_paper_artifact(paper_id)
                if cached_pages
                else None
            )

            content_is_valid = (
                artifact is not None
                and isinstance(artifact.get("content_fingerprint"), str)
                and bool(artifact["content_fingerprint"])
                and cached_pages is not None
                and content_fingerprint(cached_pages)
                == artifact["content_fingerprint"]
            )

            chunks = None

            if content_is_valid:
                cached_chunks = self._chunks_from_cache(
                    cached_content["chunks"],
                    cached_pages,
                )

                if (
                    cached_chunks is not None
                    and artifact.get("chunking_fingerprint")
                    == config_fingerprint
                ):
                    expected_chunks = self.chunker.chunk(cached_pages)

                    if cached_chunks == expected_chunks:
                        update_paper_status(arxiv_id, "ready")

                        logger.info(
                            "Reusing cached pages and chunks for paper %s.",
                            arxiv_id,
                        )

                        return PaperPreparationResult(
                            chunks=cached_chunks,
                            content_fingerprint=artifact["content_fingerprint"],
                            chunking_fingerprint=config_fingerprint,
                        )

                    chunks = expected_chunks

                pages = cached_pages

            else:
                try:
                    pages = self.pdf_processor.parse_pdf(Path(pdf_path))
                except PDFParseError:
                    logger.exception("PDF parsing failed for paper %s.", arxiv_id)
                    return self._failure(
                        arxiv_id,
                        "Failed to parse PDF",
                    )

            update_paper_status(arxiv_id, "parsing")

            fingerprint = content_fingerprint(pages)

            if chunks is None:
                try:
                    chunks = self.chunker.chunk(pages)
                except Exception:
                    logger.exception(
                        "Chunk generation failed for paper %s.",
                        arxiv_id,
                    )
                    return self._failure(
                        arxiv_id,
                        "Failed to generate paper chunks",
                    )

            if not chunks:
                self._mark_failed(arxiv_id)
                return PaperPreparationResult(
                    chunks=[],
                    error="No usable text was found in the paper",
                )

            stale_index_path = save_paper_content(
                paper_id,
                [
                    {
                        "page_number": page.page,
                        "text": page.text,
                    }
                    for page in pages
                ],
                [
                    {
                        "chunk_id": chunk.chunk_id,
                        "page_number": chunk.page,
                        "text": chunk.text,
                    }
                    for chunk in chunks
                ],
                fingerprint,
                config_fingerprint,
            )

            self._cleanup_invalidated_index(stale_index_path)

            update_paper_status(arxiv_id, "ready")

            logger.info("Paper %s is ready for retrieval.", arxiv_id)

            return PaperPreparationResult(
                chunks=chunks,
                content_fingerprint=fingerprint,
                chunking_fingerprint=config_fingerprint,
            )

        except Exception:
            logger.exception(
                "Unexpected failure while parsing or chunking paper %s.",
                arxiv_id,
            )
            return self._failure(
                arxiv_id,
                "Failed to prepare paper content",
            )
