import pymupdf
import pytest
import requests
from pathlib import Path
from unittest.mock import MagicMock
import project.nodes.pdf as pdf_module
from project.nodes.pdf import PDFDownloadError, PDFPage, PDFProcessor

def create_test_pdf(path: Path):
    document = pymupdf.open()

    page = document.new_page()
    page.insert_text((50, 100), "This is the first page.")

    page = document.new_page()
    page.insert_text((50, 100), "This is the second page.")

    document.save(path)
    document.close()


def test_pdf_processor_creates_output_directory(tmp_path):
    output_dir = tmp_path / "papers"

    PDFProcessor(output_dir=str(output_dir))

    assert output_dir.exists()
    assert output_dir.is_dir()


def test_download_pdf(tmp_path, monkeypatch):
    processor = PDFProcessor(output_dir=str(tmp_path))

    response = MagicMock()
    response.content = b"fake pdf content"

    mock_get = MagicMock(return_value=response)
    monkeypatch.setattr(pdf_module.requests, "get", mock_get)

    pdf_path = processor.download_pdf("https://arxiv.org/pdf/2401.12345", "2401.12345")

    mock_get.assert_called_once_with("https://arxiv.org/pdf/2401.12345", timeout=30)

    response.raise_for_status.assert_called_once()

    assert pdf_path == tmp_path / "2401.12345.pdf"
    assert pdf_path.exists()
    assert pdf_path.read_bytes() == b"fake pdf content"


def test_download_pdf_raises_download_error(tmp_path, monkeypatch):
    processor = PDFProcessor(output_dir=str(tmp_path))

    response = MagicMock()
    response.raise_for_status.side_effect = requests.RequestException("download failed")

    mock_get = MagicMock(return_value=response)
    monkeypatch.setattr(pdf_module.requests, "get", mock_get)

    with pytest.raises(PDFDownloadError, match="Failed to download PDF"):
        processor.download_pdf("https://arxiv.org/pdf/2401.12345", "2401.12345")


def test_parse_pdf(tmp_path):
    pdf_path = tmp_path / "test.pdf"

    create_test_pdf(pdf_path)

    processor = PDFProcessor(output_dir=str(tmp_path))

    pages = processor.parse_pdf(pdf_path)

    assert len(pages) == 2

    assert isinstance(pages[0], PDFPage)
    assert isinstance(pages[1], PDFPage)

    assert pages[0].page == 1
    assert pages[0].text.strip() == "This is the first page."

    assert pages[1].page == 2
    assert pages[1].text.strip() == "This is the second page."


def test_parse_pdf_preserves_page_numbers(tmp_path):
    pdf_path = tmp_path / "test.pdf"

    create_test_pdf(pdf_path)

    processor = PDFProcessor(output_dir=str(tmp_path))

    pages = processor.parse_pdf(pdf_path)

    assert [page.page for page in pages] == [1, 2]
