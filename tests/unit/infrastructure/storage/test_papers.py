from unittest.mock import MagicMock, patch

from project.infrastructure.storage.papers import upload_paper_pdf


@patch("project.infrastructure.storage.papers.get_storage_bucket_client")
def test_upload_paper_pdf(mock_get_bucket_client, tmp_path):
    paper_id = "1706.03762v7"
    pdf_path = tmp_path / "paper.pdf"
    pdf_content = b"%PDF-1.7 test PDF content"
    pdf_path.write_bytes(pdf_content)

    bucket = MagicMock()
    mock_get_bucket_client.return_value = bucket

    result = upload_paper_pdf(pdf_path, paper_id)

    assert result == f"papers/{paper_id}.pdf"
    mock_get_bucket_client.assert_called_once_with()
    bucket.upload.assert_called_once()

    upload_kwargs = bucket.upload.call_args.kwargs
    assert upload_kwargs["path"] == f"papers/{paper_id}.pdf"
    assert upload_kwargs["file_options"] == {
        "content-type": "application/pdf",
        "upsert": "true",
    }

    uploaded_file = upload_kwargs["file"]
    assert uploaded_file.closed
