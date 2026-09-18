import pymupdf
import requests
from pathlib import Path

class PDFDownloadError(Exception):
    pass

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

        return pdf_path

    def parse_pdf(self, pdf_path: Path) -> str:
        document = pymupdf.open(pdf_path)

        pages = []

        for page in document:
            text = page.get_text()
            pages.append(text)

        document.close()

        return "\n".join(pages)
    