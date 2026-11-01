
from pathlib import Path
from unittest.mock import Mock, patch
from uuid import uuid4

from project.application.papers.prepare_paper import PaperPreparationService
from project.processing.chunker import TextChunk
from project.processing.fingerprints import content_fingerprint
from project.processing.pdf import PDFDownloadError, PDFPage, PDFParseError


@patch("project.application.papers.prepare_paper.get_paper_artifact")
@patch("project.application.papers.prepare_paper.update_paper_status")
@patch("project.application.papers.prepare_paper.replace_paper_pdf")
@patch("project.application.papers.prepare_paper.upload_paper_pdf")
def test_download_returns_pdf_path(
    mock_upload,
    mock_replace,
    mock_update_status,
    mock_get_artifact,
):
    pdf_processor = Mock()
    chunker = Mock()
    chunker.configuration_fingerprint = "chunking-fingerprint"
    paper_id = uuid4()
    pdf_path = Path("data/papers/1234v1.pdf")

    mock_get_artifact.return_value = None
    pdf_processor.download_pdf.return_value = pdf_path
    mock_upload.return_value = "papers/1234v1/new.pdf"
    mock_replace.return_value = {
        "old_pdf_storage_path": None,
        "old_index_storage_path": None,
    }

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
    mock_replace.assert_called_once_with(
        paper_id, "papers/1234v1/new.pdf"
    )
    mock_update_status.assert_called_once_with("1234v1", "downloading")


@patch("project.application.papers.prepare_paper.get_paper_artifact")
@patch("project.application.papers.prepare_paper.update_paper_status")
def test_download_handles_download_error(
    mock_update_status,
    mock_get_artifact,
):
    mock_get_artifact.return_value = None

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


@patch("project.application.papers.prepare_paper.get_paper_artifact")
@patch("project.application.papers.prepare_paper.download_paper_pdf")
@patch("project.application.papers.prepare_paper.upload_paper_pdf")
@patch("project.application.papers.prepare_paper.replace_paper_pdf")
@patch("project.application.papers.prepare_paper.update_paper_status")
def test_download_reuses_cached_pdf(
    mock_update_status,
    mock_replace,
    mock_upload,
    mock_download_cached,
    mock_get_artifact,
    tmp_path,
):
    pdf_processor = Mock()
    pdf_processor.output_dir = tmp_path
    chunker = Mock()
    paper_id = uuid4()

    mock_get_artifact.return_value = {
        "pdf_storage_path": "papers/1234v1.pdf"
    }

    cached_pdf_path = tmp_path / "1234v1.pdf"
    pdf_processor.parse_pdf.return_value = [
        PDFPage(page=1, text="Cached paper content")
    ]

    service = PaperPreparationService(pdf_processor, chunker)

    result = service.download(
        "https://example.com/paper.pdf", "1234v1", paper_id
    )

    assert result.pdf_path == str(cached_pdf_path)
    assert result.error is None

    mock_download_cached.assert_called_once_with(
        "papers/1234v1.pdf", cached_pdf_path
    )
    pdf_processor.download_pdf.assert_not_called()
    mock_upload.assert_not_called()
    mock_replace.assert_not_called()
    mock_update_status.assert_called_once_with("1234v1", "downloading")


@patch("project.application.papers.prepare_paper.get_paper_content")
@patch("project.application.papers.prepare_paper.update_paper_status")
@patch("project.application.papers.prepare_paper.save_paper_content")
def test_parse_and_chunk_returns_chunks(
    mock_save_content,
    mock_update_status,
    mock_get_content,
):
    pdf_processor = Mock()
    chunker = Mock()
    chunker.configuration_fingerprint = "chunking-fingerprint"
    paper_id = uuid4()

    mock_get_content.return_value = {"pages": [], "chunks": []}

    pages = [PDFPage(page=1, text="Paper content")]
    chunks = [TextChunk(chunk_id=0, page=1, text="Paper content")]
    pdf_processor.parse_pdf.return_value = pages
    chunker.chunk.return_value = chunks

    service = PaperPreparationService(pdf_processor, chunker)

    result = service.parse_and_chunk(
        "data/papers/1234v1.pdf", "1234v1", paper_id
    )

    assert result.chunks == chunks
    assert result.error is None

    pdf_processor.parse_pdf.assert_called_once_with(
        Path("data/papers/1234v1.pdf")
    )
    chunker.chunk.assert_called_once_with(pages)

    mock_save_content.assert_called_once_with(
        paper_id,
        [{"page_number": 1, "text": "Paper content"}],
        [{"chunk_id": 0, "page_number": 1, "text": "Paper content"}],
        content_fingerprint(pages),
        "chunking-fingerprint",
    )

    mock_get_content.assert_called_once_with(paper_id)
    mock_update_status.assert_any_call("1234v1", "parsing")
    mock_update_status.assert_any_call("1234v1", "ready")


@patch("project.application.papers.prepare_paper.get_paper_content")
@patch("project.application.papers.prepare_paper.update_paper_status")
def test_parse_and_chunk_handles_parse_error(
    mock_update_status,
    mock_get_content,
):
    mock_get_content.return_value = {"pages": [], "chunks": []}

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


@patch("project.application.papers.prepare_paper.get_paper_content")
@patch("project.application.papers.prepare_paper.update_paper_status")
@patch("project.application.papers.prepare_paper.save_paper_content")
def test_parse_and_chunk_handles_empty_chunks(
    mock_save_content,
    mock_update_status,
    mock_get_content,
):
    mock_get_content.return_value = {"pages": [], "chunks": []}

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


@patch("project.application.papers.prepare_paper.get_paper_artifact")
@patch("project.application.papers.prepare_paper.get_paper_content")
@patch("project.application.papers.prepare_paper.update_paper_status")
def test_parse_and_chunk_reuses_valid_cached_chunks(
    mock_update_status,
    mock_get_content,
    mock_get_artifact,
):
    pages = [PDFPage(page=1, text="Paper content")]
    chunks = [TextChunk(chunk_id=0, page=1, text="Paper content")]

    mock_get_content.return_value = {
        "pages": [{"page_number": 1, "text": "Paper content"}],
        "chunks": [
            {"chunk_id": 0, "page_number": 1, "text": "Paper content"}
        ],
    }

    mock_get_artifact.return_value = {
        "content_fingerprint": content_fingerprint(pages),
        "chunking_fingerprint": "chunking-fingerprint",
    }

    pdf_processor = Mock()
    chunker = Mock()
    chunker.configuration_fingerprint = "chunking-fingerprint"
    chunker.chunk.return_value = chunks

    service = PaperPreparationService(pdf_processor, chunker)

    result = service.parse_and_chunk("paper.pdf", "1234v1", uuid4())

    assert result.chunks == chunks
    assert result.error is None

    pdf_processor.parse_pdf.assert_not_called()
    chunker.chunk.assert_called_once_with(pages)
    mock_update_status.assert_called_once_with("1234v1", "ready")


@patch("project.application.papers.prepare_paper.get_paper_artifact")
@patch("project.application.papers.prepare_paper.get_paper_content")
@patch("project.application.papers.prepare_paper.update_paper_status")
@patch("project.application.papers.prepare_paper.save_paper_content")
def test_parse_and_chunk_regenerates_corrupted_cached_chunks(
    mock_save_content,
    mock_update_status,
    mock_get_content,
    mock_get_artifact,
):
    pages = [PDFPage(page=1, text="Paper content")]

    cached_chunks = [
        {"chunk_id": 0, "page_number": 1, "text": "Corrupted content"}
    ]

    expected_chunks = [
        TextChunk(chunk_id=0, page=1, text="Paper content")
    ]

    mock_get_content.return_value = {
        "pages": [{"page_number": 1, "text": "Paper content"}],
        "chunks": cached_chunks,
    }

    mock_get_artifact.return_value = {
        "content_fingerprint": content_fingerprint(pages),
        "chunking_fingerprint": "chunking-fingerprint",
    }

    pdf_processor = Mock()
    chunker = Mock()
    chunker.configuration_fingerprint = "chunking-fingerprint"
    chunker.chunk.return_value = expected_chunks

    service = PaperPreparationService(pdf_processor, chunker)

    result = service.parse_and_chunk("paper.pdf", "1234v1", uuid4())

    assert result.chunks == expected_chunks
    assert result.error is None

    pdf_processor.parse_pdf.assert_not_called()
    chunker.chunk.assert_called_once_with(pages)
    mock_save_content.assert_called_once()
    mock_update_status.assert_any_call("1234v1", "ready")


@patch("project.application.papers.prepare_paper.get_paper_artifact")
@patch("project.application.papers.prepare_paper.get_paper_content")
@patch("project.application.papers.prepare_paper.update_paper_status")
@patch("project.application.papers.prepare_paper.save_paper_content")
def test_parse_and_chunk_regenerates_chunks_when_configuration_changes(
    mock_save_content,
    mock_update_status,
    mock_get_content,
    mock_get_artifact,
):
    pages = [PDFPage(page=1, text="Paper content")]

    cached_chunks = [
        {"chunk_id": 0, "page_number": 1, "text": "Old chunk"}
    ]

    regenerated_chunks = [
        TextChunk(chunk_id=0, page=1, text="Paper content")
    ]

    mock_get_content.return_value = {
        "pages": [{"page_number": 1, "text": "Paper content"}],
        "chunks": cached_chunks,
    }

    mock_get_artifact.return_value = {
        "content_fingerprint": content_fingerprint(pages),
        "chunking_fingerprint": "old-configuration",
    }

    pdf_processor = Mock()
    chunker = Mock()
    chunker.configuration_fingerprint = "new-configuration"
    chunker.chunk.return_value = regenerated_chunks

    service = PaperPreparationService(pdf_processor, chunker)

    result = service.parse_and_chunk("paper.pdf", "1234v1", uuid4())

    assert result.chunks == regenerated_chunks
    assert result.error is None

    pdf_processor.parse_pdf.assert_not_called()
    chunker.chunk.assert_called_once_with(pages)

    mock_save_content.assert_called_once()
    assert mock_save_content.call_args.args[4] == "new-configuration"

    mock_update_status.assert_any_call("1234v1", "ready")


@patch("project.application.papers.prepare_paper.get_paper_artifact")
@patch("project.application.papers.prepare_paper.update_paper_status")
@patch("project.application.papers.prepare_paper.download_paper_pdf")
@patch("project.application.papers.prepare_paper.upload_paper_pdf")
@patch("project.application.papers.prepare_paper.replace_paper_pdf")
@patch("project.application.papers.prepare_paper.delete_paper_pdf")
def test_download_recovers_missing_cached_pdf(
    mock_delete_pdf,
    mock_replace,
    mock_upload,
    mock_download,
    mock_update_status,
    mock_get_artifact,
    tmp_path,
):
    paper_id = uuid4()
    pdf_path = tmp_path / "1234v1.pdf"

    mock_get_artifact.return_value = {
        "pdf_storage_path": "papers/1234v1/old.pdf"
    }
    mock_download.side_effect = FileNotFoundError()
    mock_upload.return_value = "papers/1234v1/new.pdf"
    mock_replace.return_value = {
        "old_pdf_storage_path": "papers/1234v1/old.pdf",
        "old_index_storage_path": None,
    }

    pdf_processor = Mock()
    pdf_processor.output_dir = tmp_path
    pdf_processor.download_pdf.return_value = pdf_path

    service = PaperPreparationService(pdf_processor, Mock())

    result = service.download(
        "https://example.com/paper.pdf", "1234v1", paper_id
    )

    assert result.pdf_path == str(pdf_path)
    assert result.error is None

    pdf_processor.download_pdf.assert_called_once()
    mock_upload.assert_called_once_with(pdf_path, "1234v1")
    mock_replace.assert_called_once_with(
        paper_id, "papers/1234v1/new.pdf"
    )
    mock_delete_pdf.assert_called_once_with("papers/1234v1/old.pdf")


@patch("project.application.papers.prepare_paper.get_paper_artifact")
@patch("project.application.papers.prepare_paper.update_paper_status")
@patch("project.application.papers.prepare_paper.download_paper_pdf")
@patch("project.application.papers.prepare_paper.upload_paper_pdf")
@patch("project.application.papers.prepare_paper.replace_paper_pdf")
@patch("project.application.papers.prepare_paper.delete_paper_pdf")
def test_download_recovers_corrupted_cached_pdf(
    mock_delete_pdf,
    mock_replace,
    mock_upload,
    mock_download,
    mock_update_status,
    mock_get_artifact,
    tmp_path,
):
    paper_id = uuid4()
    pdf_path = tmp_path / "1234v1.pdf"

    mock_get_artifact.return_value = {
        "pdf_storage_path": "papers/1234v1/old.pdf"
    }
    mock_upload.return_value = "papers/1234v1/new.pdf"
    mock_replace.return_value = {
        "old_pdf_storage_path": "papers/1234v1/old.pdf",
        "old_index_storage_path": None,
    }

    pdf_processor = Mock()
    pdf_processor.output_dir = tmp_path
    pdf_processor.parse_pdf.side_effect = PDFParseError("Invalid PDF")
    pdf_processor.download_pdf.return_value = pdf_path

    service = PaperPreparationService(pdf_processor, Mock())

    result = service.download(
        "https://example.com/paper.pdf", "1234v1", paper_id
    )

    assert result.pdf_path == str(pdf_path)
    assert result.error is None

    pdf_processor.parse_pdf.assert_called_once()
    pdf_processor.download_pdf.assert_called_once()
    mock_upload.assert_called_once_with(pdf_path, "1234v1")
    mock_replace.assert_called_once_with(
        paper_id, "papers/1234v1/new.pdf"
    )
    mock_delete_pdf.assert_called_once_with("papers/1234v1/old.pdf")


@patch("project.application.papers.prepare_paper.get_paper_artifact")
@patch("project.application.papers.prepare_paper.get_paper_content")
@patch("project.application.papers.prepare_paper.update_paper_status")
@patch("project.application.papers.prepare_paper.save_paper_content")
def test_parse_and_chunk_recovers_invalid_cached_pages(
    mock_save_content,
    mock_update_status,
    mock_get_content,
    mock_get_artifact,
):
    paper_id = uuid4()

    mock_get_content.return_value = {
        "pages": [{"page_number": 2, "text": "Cached content"}],
        "chunks": [
            {"chunk_id": 0, "page_number": 2, "text": "Cached content"}
        ],
    }

    fresh_pages = [PDFPage(page=1, text="Fresh PDF content")]
    expected_chunks = [
        TextChunk(chunk_id=0, page=1, text="Fresh PDF content")
    ]

    pdf_processor = Mock()
    pdf_processor.parse_pdf.return_value = fresh_pages

    chunker = Mock()
    chunker.configuration_fingerprint = "new-configuration"
    chunker.chunk.return_value = expected_chunks

    service = PaperPreparationService(pdf_processor, chunker)

    result = service.parse_and_chunk("paper.pdf", "1234v1", paper_id)

    pdf_processor.parse_pdf.assert_called_once_with(Path("paper.pdf"))
    chunker.chunk.assert_called_once_with(fresh_pages)

    assert result.chunks == expected_chunks
    assert result.error is None

    mock_save_content.assert_called_once_with(
        paper_id,
        [{"page_number": 1, "text": "Fresh PDF content"}],
        [{"chunk_id": 0, "page_number": 1, "text": "Fresh PDF content"}],
        content_fingerprint(fresh_pages),
        "new-configuration",
    )

    mock_update_status.assert_any_call("1234v1", "ready")
    mock_get_artifact.assert_not_called()
