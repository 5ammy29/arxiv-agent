from unittest.mock import MagicMock, patch

from project.infrastructure.storage.papers import (
    delete_paper_pdf,
    download_paper_pdf,
    upload_paper_pdf,
)


@patch("project.infrastructure.storage.papers.get_storage_bucket_client")
def test_upload_paper_pdf(mock_get_bucket_client, tmp_path):
    paper_id = "1706.03762v7"
    pdf_path = tmp_path / "paper.pdf"
    pdf_content = b"%PDF-1.7 test PDF content"
    pdf_path.write_bytes(pdf_content)

    bucket = MagicMock()
    mock_get_bucket_client.return_value = bucket

    result = upload_paper_pdf(pdf_path, paper_id)

    assert result.startswith(f"papers/{paper_id}/")
    assert result.endswith(".pdf")

    mock_get_bucket_client.assert_called_once_with()
    bucket.upload.assert_called_once()

    upload_kwargs = bucket.upload.call_args.kwargs

    assert upload_kwargs["path"] == result
    assert upload_kwargs["file_options"] == {
        "content-type": "application/pdf",
        "upsert": "false",
    }

    uploaded_file = upload_kwargs["file"]
    assert uploaded_file.closed


@patch("project.infrastructure.storage.papers.get_storage_bucket_client")
def test_download_paper_pdf(mock_get_bucket_client, tmp_path):
    storage_path = "papers/1706.03762v7/document.pdf"
    destination_path = tmp_path / "1706.03762v7.pdf"
    pdf_content = b"%PDF-1.7 test PDF content"

    bucket = MagicMock()
    bucket.download.return_value = pdf_content
    mock_get_bucket_client.return_value = bucket

    result = download_paper_pdf(storage_path, destination_path)

    assert result == destination_path
    assert result.read_bytes() == pdf_content

    mock_get_bucket_client.assert_called_once_with()
    bucket.download.assert_called_once_with(storage_path)


@patch("project.infrastructure.storage.papers.get_storage_bucket_client")
def test_delete_paper_pdf(mock_get_bucket_client):
    from project.infrastructure.storage.papers import delete_paper_pdf

    bucket = MagicMock()
    mock_get_bucket_client.return_value = bucket

    delete_paper_pdf("papers/1706.03762v7/document.pdf")

    bucket.remove.assert_called_once_with(
        ["papers/1706.03762v7/document.pdf"]
    )