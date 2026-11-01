from dataclasses import dataclass
from pathlib import Path

import pymupdf
import requests


class PDFDownloadError(Exception):
    pass


class PDFParseError(Exception):
    pass


class PDFValidationError(PDFParseError):
    """A downloaded object is not a usable PDF artifact."""


@dataclass
class PDFPage:
    page: int
    text: str


class PDFProcessor:
    def __init__(self, output_dir: str = "data/papers"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_pdf(self, pdf_url: str, arxiv_id: str) -> Path:
        pdf_path = self.output_dir / f"{arxiv_id}.pdf"

        try:
            response = requests.get(pdf_url, timeout=30)
            response.raise_for_status()
        except requests.RequestException as error:
            raise PDFDownloadError(
                f"Failed to download PDF for {arxiv_id}"
            ) from error

        pdf_path.write_bytes(response.content)

        try:
            self.validate_pdf(pdf_path)
        except PDFValidationError:
            pdf_path.unlink(missing_ok=True)
            raise PDFDownloadError(
                f"Downloaded file for {arxiv_id} is not a valid PDF"
            )

        return pdf_path

    @staticmethod
    def validate_pdf(pdf_path: Path) -> None:
        """Perform a cheap, deterministic guard before invoking PyMuPDF."""
        try:
            with pdf_path.open("rb") as pdf_file:
                initial_bytes = pdf_file.read(1024)
        except OSError as error:
            raise PDFValidationError("PDF is missing or unreadable") from error

        if not initial_bytes:
            raise PDFValidationError("PDF is empty")

        # PDF comments/binary prefixes are permitted, but the header must occur
        # near the beginning of the object.
        if b"%PDF-" not in initial_bytes:
            raise PDFValidationError("PDF signature is missing")

    def parse_pdf(self, pdf_path: Path) -> list[PDFPage]:
        self.validate_pdf(pdf_path)
        try:
            document = pymupdf.open(pdf_path)
        except Exception as error:
            raise PDFParseError("Failed to parse PDF") from error

        pages = []

        try:
            for page_number, page in enumerate(document, start=1):
                pages.append(PDFPage(page=page_number, text=page.get_text()))
        except Exception as error:
            raise PDFParseError("Failed to parse PDF") from error
        finally:
            document.close()

        if not pages or not any(page.text.strip() for page in pages):
            raise PDFValidationError("PDF contains no usable extracted text")

        return pages
