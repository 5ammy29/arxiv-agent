from pathlib import Path
from unittest.mock import Mock, patch
from uuid import uuid4

from project.application.papers.prepare_paper import PaperPreparationService
from project.processing.chunker import TextChunk
from project.processing.pdf import PDFPage


@patch("project.application.papers.prepare_paper.update_paper_status")
@patch("project.application.papers.prepare_paper.get_paper_artifact")
def test_download_handles_artifact_database_failure(
    mock_get_artifact,
    mock_update_status,
):
    mock_get_artifact.side_effect = RuntimeError("Database unavailable")

    service = PaperPreparationService(Mock(), Mock())

    result = service.download(
        "https://example.com/paper.pdf",
        "1234v1",
        uuid4(),
    )

    assert result.pdf_path is None
    assert result.error == "Failed to prepare PDF for 1234v1"
    mock_update_status.assert_any_call("1234v1", "downloading")
    mock_update_status.assert_any_call("1234v1", "failed")


@patch("project.application.papers.prepare_paper.update_paper_status")
@patch("project.application.papers.prepare_paper.get_paper_artifact")
@patch("project.application.papers.prepare_paper.download_paper_pdf")
def test_download_handles_cached_storage_failure(
    mock_download_cached,
    mock_get_artifact,
    mock_update_status,
):
    mock_get_artifact.return_value = {
        "pdf_storage_path": "papers/1234v1/cached.pdf"
    }
    mock_download_cached.side_effect = RuntimeError("Storage unavailable")

    pdf_processor = Mock()
    pdf_processor.output_dir = Path("data/papers")

    service = PaperPreparationService(pdf_processor, Mock())

    result = service.download(
        "https://example.com/paper.pdf",
        "1234v1",
        uuid4(),
    )

    assert result.pdf_path is None
    assert result.error == "Failed to retrieve stored PDF for 1234v1"
    pdf_processor.download_pdf.assert_not_called()
    mock_update_status.assert_any_call("1234v1", "failed")


@patch("project.application.papers.prepare_paper.update_paper_status")
@patch("project.application.papers.prepare_paper.get_paper_content")
def test_parse_and_chunk_handles_content_database_failure(
    mock_get_content,
    mock_update_status,
):
    mock_get_content.side_effect = RuntimeError("Database unavailable")

    service = PaperPreparationService(Mock(), Mock())

    result = service.parse_and_chunk(
        "paper.pdf",
        "1234v1",
        uuid4(),
    )

    assert result.chunks is None
    assert result.error == "Failed to prepare paper content"
    mock_update_status.assert_called_once_with("1234v1", "failed")


@patch("project.application.papers.prepare_paper.update_paper_status")
@patch("project.application.papers.prepare_paper.get_paper_content")
def test_parse_and_chunk_handles_chunk_configuration_failure(
    mock_get_content,
    mock_update_status,
):
    class BrokenChunker:
        @property
        def configuration_fingerprint(self):
            raise RuntimeError("Configuration failure")

    mock_get_content.return_value = {
        "pages": [],
        "chunks": [],
    }

    service = PaperPreparationService(Mock(), BrokenChunker())

    result = service.parse_and_chunk(
        "paper.pdf",
        "1234v1",
        uuid4(),
    )

    assert result.error == "Failed to prepare paper content"
    mock_update_status.assert_called_once_with("1234v1", "failed")


@patch("project.application.papers.prepare_paper.update_paper_status")
@patch("project.application.papers.prepare_paper.save_paper_content")
@patch("project.application.papers.prepare_paper.get_paper_content")
def test_parse_and_chunk_handles_content_persistence_failure(
    mock_get_content,
    mock_save_content,
    mock_update_status,
):
    mock_get_content.return_value = {
        "pages": [],
        "chunks": [],
    }
    mock_save_content.side_effect = RuntimeError("Database write failed")

    pages = [PDFPage(page=1, text="Paper content")]
    chunks = [TextChunk(chunk_id=0, page=1, text="Paper content")]

    pdf_processor = Mock()
    pdf_processor.parse_pdf.return_value = pages

    chunker = Mock()
    chunker.configuration_fingerprint = "chunking-fingerprint"
    chunker.chunk.return_value = chunks

    service = PaperPreparationService(pdf_processor, chunker)

    result = service.parse_and_chunk(
        "paper.pdf",
        "1234v1",
        uuid4(),
    )

    assert result.chunks is None
    assert result.error == "Failed to prepare paper content"
    mock_save_content.assert_called_once()
    mock_update_status.assert_any_call("1234v1", "parsing")
    mock_update_status.assert_any_call("1234v1", "failed")


@patch("project.application.papers.prepare_paper.update_paper_status")
@patch("project.application.papers.prepare_paper.save_paper_content")
@patch("project.application.papers.prepare_paper.get_paper_content")
def test_parse_and_chunk_handles_ready_status_failure(
    mock_get_content,
    mock_save_content,
    mock_update_status,
):
    mock_get_content.return_value = {
        "pages": [],
        "chunks": [],
    }

    pages = [PDFPage(page=1, text="Paper content")]
    chunks = [TextChunk(chunk_id=0, page=1, text="Paper content")]

    pdf_processor = Mock()
    pdf_processor.parse_pdf.return_value = pages

    chunker = Mock()
    chunker.configuration_fingerprint = "chunking-fingerprint"
    chunker.chunk.return_value = chunks

    def update_status_side_effect(arxiv_id, status):
        if status == "ready":
            raise RuntimeError("Database unavailable")

    mock_update_status.side_effect = update_status_side_effect

    service = PaperPreparationService(pdf_processor, chunker)

    result = service.parse_and_chunk(
        "paper.pdf",
        "1234v1",
        uuid4(),
    )

    assert result.chunks is None
    assert result.error == "Failed to prepare paper content"

    mock_update_status.assert_any_call("1234v1", "parsing")
    mock_update_status.assert_any_call("1234v1", "ready")
    mock_update_status.assert_any_call("1234v1", "failed")
