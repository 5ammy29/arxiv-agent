from pathlib import Path
from unittest.mock import Mock, patch
from uuid import uuid4

from project.application.papers.prepare_paper import PaperPreparationService
from project.processing.chunker import TextChunk
from project.processing.fingerprints import content_fingerprint
from project.processing.pdf import PDFDownloadError, PDFPage, PDFParseError


def make_service(pdf_processor=None, chunker=None):
    pdf_processor = pdf_processor or Mock()
    chunker = chunker or Mock()
    chunker.configuration_fingerprint = "chunking-fingerprint"
    return PaperPreparationService(pdf_processor, chunker), pdf_processor, chunker


@patch("project.application.papers.prepare_paper.get_paper_artifact", return_value=None)
@patch("project.application.papers.prepare_paper.update_paper_status")
@patch("project.application.papers.prepare_paper.replace_paper_pdf")
@patch("project.application.papers.prepare_paper.upload_paper_pdf")
def test_download_publishes_pdf_with_lease(mock_upload, mock_replace, mock_status, _artifact):
    paper_id, pdf_path = uuid4(), Path("paper.pdf")
    service, pdf_processor, _ = make_service()
    pdf_processor.download_pdf.return_value = pdf_path
    mock_upload.return_value = "papers/1234v1/new.pdf"
    mock_replace.return_value = {"old_pdf_storage_path": None, "old_index_storage_path": None}

    result = service.download("https://example.test/paper.pdf", "1234v1", paper_id)

    assert result == result.__class__(pdf_path=str(pdf_path))
    assert mock_replace.call_args.args == (paper_id, "papers/1234v1/new.pdf")
    assert mock_replace.call_args.kwargs["lease"].paper_id == paper_id
    assert mock_status.call_args.kwargs["lease"].paper_id == paper_id


@patch("project.application.papers.prepare_paper.get_paper_artifact")
@patch("project.application.papers.prepare_paper.update_paper_status")
def test_download_reuses_cached_pdf(mock_status, mock_artifact, tmp_path):
    paper_id = uuid4()
    mock_artifact.return_value = {"pdf_storage_path": "papers/1234v1/cached.pdf"}
    service, processor, _ = make_service()
    processor.output_dir = tmp_path
    processor.parse_pdf.return_value = [PDFPage(page=1, text="cached")]

    with patch("project.application.papers.prepare_paper.download_paper_pdf") as download:
        result = service.download("url", "1234v1", paper_id)

    assert result.pdf_path == str(tmp_path / "1234v1.pdf")
    download.assert_called_once()
    processor.download_pdf.assert_not_called()
    assert mock_status.call_args.kwargs["lease"].paper_id == paper_id


@patch("project.application.papers.prepare_paper.get_paper_artifact")
@patch("project.application.papers.prepare_paper.update_paper_status")
@patch("project.application.papers.prepare_paper.replace_paper_pdf")
@patch("project.application.papers.prepare_paper.upload_paper_pdf")
def test_corrupt_cached_pdf_is_replaced_from_source(
    mock_upload, mock_replace, _status, mock_artifact, tmp_path
):
    paper_id = uuid4()
    cached_path = tmp_path / "1234v1.pdf"
    downloaded_path = tmp_path / "new.pdf"
    mock_artifact.return_value = {"pdf_storage_path": "papers/1234v1/cached.pdf"}
    mock_upload.return_value = "papers/1234v1/replacement.pdf"
    mock_replace.return_value = {"old_pdf_storage_path": None, "old_index_storage_path": None}
    service, processor, _ = make_service()
    processor.output_dir = tmp_path
    processor.parse_pdf.side_effect = PDFParseError("corrupt stored PDF")
    downloaded_path.write_bytes(b"%PDF-replacement")
    processor.download_pdf.return_value = downloaded_path

    def restore_corrupt_pdf(_storage_path, destination):
        Path(destination).write_bytes(b"not a PDF")

    with patch(
        "project.application.papers.prepare_paper.download_paper_pdf",
        side_effect=restore_corrupt_pdf,
    ):
        result = service.download("https://example.test/paper.pdf", "1234v1", paper_id)

    assert result.pdf_path == str(downloaded_path)
    processor.download_pdf.assert_called_once_with("https://example.test/paper.pdf", "1234v1")
    mock_upload.assert_called_once_with(downloaded_path, "1234v1")
    assert mock_replace.call_args.args == (paper_id, "papers/1234v1/replacement.pdf")
    assert not cached_path.exists()


@patch("project.application.papers.prepare_paper.get_paper_content", return_value={"pages": [], "chunks": []})
@patch("project.application.papers.prepare_paper.update_paper_status")
@patch("project.application.papers.prepare_paper.save_paper_content")
def test_parse_and_chunk_persists_content_with_lease(mock_save, mock_status, _content):
    paper_id = uuid4()
    pages = [PDFPage(page=1, text="Paper content")]
    chunks = [TextChunk(chunk_id=0, page=1, text="Paper content")]
    service, processor, chunker = make_service()
    processor.parse_pdf.return_value = pages
    chunker.chunk.return_value = chunks

    result = service.parse_and_chunk("paper.pdf", "1234v1", paper_id)

    assert result.chunks == chunks
    assert mock_save.call_args.args[:5] == (
        paper_id,
        [{"page_number": 1, "text": "Paper content"}],
        [{"chunk_id": 0, "page_number": 1, "text": "Paper content"}],
        content_fingerprint(pages),
        "chunking-fingerprint",
    )
    assert mock_save.call_args.kwargs["lease"].paper_id == paper_id
    assert {call.args[1] for call in mock_status.call_args_list} == {"parsing", "ready"}


@patch("project.application.papers.prepare_paper.get_paper_artifact")
@patch("project.application.papers.prepare_paper.get_paper_content")
@patch("project.application.papers.prepare_paper.update_paper_status")
def test_parse_and_chunk_reuses_valid_content(mock_status, mock_content, mock_artifact):
    pages = [PDFPage(page=1, text="Paper content")]
    chunks = [TextChunk(chunk_id=0, page=1, text="Paper content")]
    mock_content.return_value = {
        "pages": [{"page_number": 1, "text": "Paper content"}],
        "chunks": [{"chunk_id": 0, "page_number": 1, "text": "Paper content"}],
    }
    mock_artifact.return_value = {
        "content_fingerprint": content_fingerprint(pages),
        "chunking_fingerprint": "chunking-fingerprint",
    }
    service, processor, chunker = make_service()
    chunker.chunk.return_value = chunks

    result = service.parse_and_chunk("unused.pdf", "1234v1", uuid4())

    assert result.chunks == chunks
    processor.parse_pdf.assert_not_called()
    assert mock_status.call_args.args[1] == "ready"


@patch("project.application.papers.prepare_paper.get_paper_artifact", return_value=None)
@patch("project.application.papers.prepare_paper.update_paper_status")
def test_download_marks_failure(mock_status, _artifact):
    service, processor, _ = make_service()
    processor.download_pdf.side_effect = PDFDownloadError("network")

    result = service.download("url", "1234v1", uuid4())

    assert result.error == "Failed to download PDF for 1234v1"
    assert [call.args[1] for call in mock_status.call_args_list] == ["downloading", "failed"]


@patch("project.application.papers.prepare_paper.get_paper_content", return_value={"pages": [], "chunks": []})
@patch("project.application.papers.prepare_paper.update_paper_status")
def test_parse_error_marks_failure(mock_status, _content):
    service, processor, _ = make_service()
    processor.parse_pdf.side_effect = PDFParseError("bad")

    result = service.parse_and_chunk("paper.pdf", "1234v1", uuid4())

    assert result.error == "Failed to parse PDF"
    assert [call.args[1] for call in mock_status.call_args_list] == ["failed"]
