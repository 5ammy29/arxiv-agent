from pathlib import Path
from unittest.mock import Mock, patch
from uuid import uuid4

from project.application.papers.prepare_paper import PaperPreparationService
from project.processing.chunker import TextChunk
from project.processing.pdf import PDFDownloadError, PDFPage, PDFParseError


@patch("project.application.papers.prepare_paper.update_paper_status")
@patch("project.application.papers.prepare_paper.save_paper_artifact")
@patch("project.application.papers.prepare_paper.upload_paper_pdf")
def test_download_returns_pdf_path(mock_upload, mock_save_artifact, mock_update_status):
    pdf_processor = Mock()
    chunker = Mock()
    paper_id = uuid4()
    pdf_path = Path("data/papers/1234v1.pdf")
    pdf_processor.download_pdf.return_value = pdf_path
    mock_upload.return_value = "papers/1234v1.pdf"

    service = PaperPreparationService(pdf_processor, chunker)

    result = service.download(
        "https://example.com/paper.pdf", "1234v1", paper_id
    )

    assert result.pdf_path == str(pdf_path)
    assert result.error is None
    pdf_processor.download_pdf.assert_called_once_with(
        "https://example.com/paper.pdf", "1234v1"
    )
    mock_upload.assert_called_once_with(pdf_path, "1234v1")
    mock_save_artifact.assert_called_once_with(paper_id, "papers/1234v1.pdf")
    mock_update_status.assert_called_once_with("1234v1", "downloading")


@patch("project.application.papers.prepare_paper.update_paper_status")
def test_download_handles_download_error(mock_update_status):
    pdf_processor = Mock()
    pdf_processor.download_pdf.side_effect = PDFDownloadError("Download failed")
    paper_id = uuid4()

    service = PaperPreparationService(pdf_processor, Mock())

    result = service.download(
        "https://example.com/paper.pdf", "1234v1", paper_id
    )

    assert result.pdf_path is None
    assert result.error == "Failed to download PDF for 1234v1"
    mock_update_status.assert_any_call("1234v1", "downloading")
    mock_update_status.assert_any_call("1234v1", "failed")


@patch("project.application.papers.prepare_paper.update_paper_status")
@patch("project.application.papers.prepare_paper.save_paper_content")
def test_parse_and_chunk_returns_chunks(mock_save_content, mock_update_status):
    pdf_processor = Mock()
    chunker = Mock()
    paper_id = uuid4()

    pages = [PDFPage(page=1, text="Paper content")]
    chunks = [TextChunk(chunk_id=0, page=1, text="Paper content")]
    pdf_processor.parse_pdf.return_value = pages
    chunker.chunk.return_value = chunks

    service = PaperPreparationService(pdf_processor, chunker)

    result = service.parse_and_chunk("data/papers/1234v1.pdf", "1234v1", paper_id)

    assert result.chunks == chunks
    assert result.error is None
    pdf_processor.parse_pdf.assert_called_once_with(Path("data/papers/1234v1.pdf"))
    chunker.chunk.assert_called_once_with(pages)
    mock_save_content.assert_called_once_with(
        paper_id,
        [{"page_number": 1, "text": "Paper content"}],
        [{"chunk_id": 0, "page_number": 1, "text": "Paper content"}],
    )
    mock_update_status.assert_any_call("1234v1", "parsing")
    mock_update_status.assert_any_call("1234v1", "ready")


@patch("project.application.papers.prepare_paper.update_paper_status")
def test_parse_and_chunk_handles_parse_error(mock_update_status):
    pdf_processor = Mock()
    chunker = Mock()
    paper_id = uuid4()
    pdf_processor.parse_pdf.side_effect = PDFParseError("Invalid PDF")

    service = PaperPreparationService(pdf_processor, chunker)

    result = service.parse_and_chunk("invalid.pdf", "1234v1", paper_id)

    assert result.chunks is None
    assert result.error == "Failed to parse PDF"
    chunker.chunk.assert_not_called()
    mock_update_status.assert_any_call("1234v1", "failed")


@patch("project.application.papers.prepare_paper.update_paper_status")
@patch("project.application.papers.prepare_paper.save_paper_content")
def test_parse_and_chunk_handles_empty_chunks(mock_save_content, mock_update_status):
    pdf_processor = Mock()
    chunker = Mock()
    paper_id = uuid4()

    pdf_processor.parse_pdf.return_value = [PDFPage(page=1, text="")]
    chunker.chunk.return_value = []

    service = PaperPreparationService(pdf_processor, chunker)

    result = service.parse_and_chunk("empty.pdf", "1234v1", paper_id)

    assert result.chunks == []
    assert result.error == "No usable text was found in the paper"
    mock_save_content.assert_not_called()
    mock_update_status.assert_any_call("1234v1", "failed")
