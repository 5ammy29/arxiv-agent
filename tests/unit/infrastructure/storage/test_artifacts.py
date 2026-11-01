from unittest.mock import MagicMock, patch

import pytest

from project.infrastructure.storage.artifacts import list_artifact_objects


@patch("project.infrastructure.storage.artifacts.get_storage_bucket_client")
def test_lists_every_storage_page(mock_get_bucket):
    bucket = MagicMock()
    first_page = [
        {"id": str(index), "name": f"first-{index}.pdf", "updated_at": "2026-01-01T00:00:00Z"}
        for index in range(1000)
    ]
    second_page = [{"id": "last", "name": "last.pdf", "updated_at": "2026-01-02T00:00:00Z"}]
    bucket.list.side_effect = [first_page, second_page]
    mock_get_bucket.return_value = bucket

    objects = list_artifact_objects("papers")

    assert len(objects) == 1001
    assert objects[-1]["path"] == "papers/last.pdf"
    assert bucket.list.call_args_list[0].kwargs["options"] == {"limit": 1000, "offset": 0}
    assert bucket.list.call_args_list[1].kwargs["options"] == {"limit": 1000, "offset": 1000}


@patch("project.infrastructure.storage.artifacts.get_storage_bucket_client")
def test_recursively_lists_directories_and_preserves_unknown_timestamp(mock_get_bucket):
    bucket = MagicMock()
    bucket.list.side_effect = [
        [{"name": "1234v1"}],
        [{"id": "pdf", "name": "paper.pdf", "updated_at": "not-a-date"}],
    ]
    mock_get_bucket.return_value = bucket

    assert list_artifact_objects("papers") == [
        {"path": "papers/1234v1/paper.pdf", "updated_at": None}
    ]


@pytest.mark.parametrize("prefix", ["", "/papers", "papers/../private"])
def test_rejects_non_managed_or_unsafe_prefix(prefix):
    with pytest.raises(ValueError, match="relative managed prefix"):
        list_artifact_objects(prefix)


@patch("project.infrastructure.storage.artifacts.get_storage_bucket_client")
def test_deduplicates_objects_returned_on_overlapping_pages(mock_get_bucket):
    bucket = MagicMock()
    first_page = [
        {"id": str(index), "name": f"paper-{index}.pdf"}
        for index in range(1000)
    ]
    bucket.list.side_effect = [
        first_page,
        [{"id": "999", "name": "paper-999.pdf"}],
    ]
    mock_get_bucket.return_value = bucket

    objects = list_artifact_objects("papers")

    assert len(objects) == 1000
