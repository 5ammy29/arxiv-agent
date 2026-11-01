from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from project.infrastructure.storage.vector_indexes import (
    download_vector_index,
    upload_vector_index,
)

from concurrent.futures import ThreadPoolExecutor

@patch(
    "project.infrastructure.storage.vector_indexes.get_storage_bucket_client"
)
def test_upload_vector_index(mock_get_bucket):
    paper_id = uuid4()
    index_bytes = b"serialized-faiss-index"

    bucket = MagicMock()
    mock_get_bucket.return_value = bucket

    result = upload_vector_index(index_bytes, paper_id)

    assert result.startswith(f"vector_indexes/{paper_id}/")
    assert result.endswith(".faiss")

    bucket.upload.assert_called_once_with(
        path=result,
        file=index_bytes,
        file_options={
            "content-type": "application/octet-stream",
            "upsert": "false",
        },
    )


def test_upload_vector_index_rejects_empty_bytes():
    with patch(
        "project.infrastructure.storage.vector_indexes.get_storage_bucket_client"
    ) as mock_get_bucket:
        with pytest.raises(ValueError, match="must not be empty"):
            upload_vector_index(b"", uuid4())

        mock_get_bucket.assert_not_called()


@patch(
    "project.infrastructure.storage.vector_indexes.get_storage_bucket_client"
)
def test_download_vector_index(mock_get_bucket):
    paper_id = uuid4()
    storage_path = f"vector_indexes/{paper_id}/{uuid4().hex}.faiss"
    index_bytes = b"serialized-faiss-index"

    bucket = MagicMock()
    bucket.download.return_value = index_bytes
    mock_get_bucket.return_value = bucket

    result = download_vector_index(storage_path)

    assert result == index_bytes
    bucket.download.assert_called_once_with(storage_path)


def test_download_vector_index_rejects_empty_path():
    with patch(
        "project.infrastructure.storage.vector_indexes.get_storage_bucket_client"
    ) as mock_get_bucket:
        with pytest.raises(ValueError, match="must not be empty"):
            download_vector_index(" ")

        mock_get_bucket.assert_not_called()


def test_concurrent_uploads_use_unique_storage_paths():
    paper_id = uuid4()
    index_bytes = b"serialized-faiss-index"

    bucket = MagicMock()

    with patch(
        "project.infrastructure.storage.vector_indexes.get_storage_bucket_client",
        return_value=bucket,
    ):
        with ThreadPoolExecutor(max_workers=10) as executor:
            paths = list(
                executor.map(
                    lambda _: upload_vector_index(index_bytes, paper_id),
                    range(20),
                )
            )

    assert len(paths) == 20
    assert len(set(paths)) == 20

    for path in paths:
        assert path.startswith(f"vector_indexes/{paper_id}/")
        assert path.endswith(".faiss")

    assert bucket.upload.call_count == 20

    for call in bucket.upload.call_args_list:
        assert call.kwargs["file"] == index_bytes
        assert call.kwargs["file_options"]["upsert"] == "false"