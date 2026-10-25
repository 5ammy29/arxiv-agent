from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from project.infrastructure.database.papers import (
    get_paper_artifact,
    get_paper_content,
    save_paper_artifact,
    save_paper_content,
    update_paper_status,
)
from project.infrastructure.storage.papers import (
    download_paper_pdf,
    upload_paper_pdf,
)
from project.infrastructure.storage.papers import upload_paper_pdf
from project.processing.chunker import TextChunk, TextChunker
from project.processing.pdf import (
    PDFDownloadError,
    PDFParseError,
    PDFProcessor,
)


@dataclass
class PaperPreparationResult:
    pdf_path: str | None = None
    chunks: list[TextChunk] | None = None
    error: str | None = None


class PaperPreparationService:
    def __init__(self, pdf_processor: PDFProcessor, chunker: TextChunker) -> None:
        self.pdf_processor = pdf_processor
        self.chunker = chunker

    def download(
        self,
        pdf_url: str,
        arxiv_id: str,
        paper_id: UUID,
    ) -> PaperPreparationResult:
        update_paper_status(arxiv_id, "downloading")

        try:
            artifact = get_paper_artifact(paper_id)

            if artifact and artifact.get("pdf_storage_path"):
                pdf_path = (
                    self.pdf_processor.output_dir / f"{arxiv_id}.pdf"
                )

                try:
                    download_paper_pdf(
                        artifact["pdf_storage_path"],
                        pdf_path,
                    )

                    self.pdf_processor.parse_pdf(pdf_path)

                    return PaperPreparationResult(pdf_path=str(pdf_path))

                except Exception:
                    pdf_path.unlink(missing_ok=True)

        except Exception:
            pass

        try:
            pdf_path = self.pdf_processor.download_pdf(pdf_url, arxiv_id)
        except PDFDownloadError:
            update_paper_status(arxiv_id, "failed")
            return PaperPreparationResult(
                error=f"Failed to download PDF for {arxiv_id}"
            )

        try:
            storage_path = upload_paper_pdf(pdf_path, arxiv_id)
            save_paper_artifact(paper_id, storage_path)
        except Exception:
            update_paper_status(arxiv_id, "failed")
            return PaperPreparationResult(
                error=f"Failed to store PDF for {arxiv_id}"
            )

        return PaperPreparationResult(pdf_path=str(pdf_path))

    def parse_and_chunk(self, pdf_path: str, arxiv_id: str, paper_id: UUID) -> PaperPreparationResult:
        cached_content = get_paper_content(paper_id)
        cached_chunks = cached_content["chunks"]

        if cached_content["pages"] and cached_chunks:
            chunks = [
                TextChunk(
                    chunk_id=chunk["chunk_id"],
                    page=chunk["page_number"],
                    text=chunk["text"],
                )
                for chunk in cached_chunks
            ]

            update_paper_status(arxiv_id, "ready")
            return PaperPreparationResult(chunks=chunks)

        update_paper_status(arxiv_id, "parsing")

        try:
            pages = self.pdf_processor.parse_pdf(Path(pdf_path))
        except PDFParseError:
            update_paper_status(arxiv_id, "failed")
            return PaperPreparationResult(error="Failed to parse PDF")

        chunks = self.chunker.chunk(pages)

        if not chunks:
            update_paper_status(arxiv_id, "failed")
            return PaperPreparationResult(
                chunks=[],
                error="No usable text was found in the paper",
            )

        save_paper_content(
            paper_id,
            [
                {"page_number": page.page, "text": page.text}
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
        )

        update_paper_status(arxiv_id, "ready")
        return PaperPreparationResult(chunks=chunks)
