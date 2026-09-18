import pymupdf
import pytest
import requests
from pathlib import Path
from unittest.mock import MagicMock
import project.nodes.pdf as pdf_module
from project.nodes.pdf import PDFDownloadError, PDFProcessor


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


def test_download_pdf(tmp_path):
    processor = PDFProcessor(output_dir=str(tmp_path))

    response = MagicMock()
    response.content = b"fake pdf content"

    processor_response = MagicMock()
    processor_response.get = MagicMock()

    original_get = pdf_module.requests.get
    pdf_module.requests.get = MagicMock(return_value=response)

    try:
        pdf_path = processor.download_pdf("https://arxiv.org/pdf/2401.12345", "2401.12345")

        pdf_module.requests.get.assert_called_once_with("https://arxiv.org/pdf/2401.12345", timeout=30)

        response.raise_for_status.assert_called_once()

        assert pdf_path == tmp_path / "2401.12345.pdf"
        assert pdf_path.exists()
        assert pdf_path.read_bytes() == b"fake pdf content"

    finally:
        pdf_module.requests.get = original_get

def test_download_pdf_raises_download_error(tmp_path, monkeypatch):
    processor = PDFProcessor(output_dir=str(tmp_path))

    response = MagicMock()
    response.raise_for_status.side_effect = requests.RequestException("Download failed")

    mock_get = MagicMock(return_value=response)
    monkeypatch.setattr(pdf_module.requests, "get", mock_get)

    with pytest.raises(PDFDownloadError, match="Failed to download PDF"):
        processor.download_pdf("https://arxiv.org/pdf/2401.12345", "2401.12345")

def test_parse_pdf(tmp_path):
    pdf_path = tmp_path / "test.pdf"

    create_test_pdf(pdf_path)

    processor = PDFProcessor(output_dir=str(tmp_path))

    text = processor.parse_pdf(pdf_path)

    assert "This is the first page." in text
    assert "This is the second page." in text


def test_parse_pdf_reads_multiple_pages(tmp_path):
    pdf_path = tmp_path / "test.pdf"

    create_test_pdf(pdf_path)

    processor = PDFProcessor(output_dir=str(tmp_path))

    text = processor.parse_pdf(pdf_path)

    pages = text.split("\n")

    assert len(text) > 0
