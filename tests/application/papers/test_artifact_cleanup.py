from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest

from project.application.papers.artifact_cleanup import ArtifactCleanupService


NOW = datetime(2026, 11, 1, tzinfo=UTC)
OLD = NOW - timedelta(days=2)
RECENT = NOW - timedelta(hours=1)
PDF_PATH = "papers/1234.5678v1/unique.pdf"
INDEX_PATH = "vector_indexes/12345678-1234-5678-1234-567812345678/test.faiss"


def service():
    return ArtifactCleanupService(clock=lambda: NOW)


def inventory(pdf_objects=None, index_objects=None):
    return [pdf_objects or [], index_objects or []]


@patch("project.application.papers.artifact_cleanup.delete_paper_pdf")
@patch("project.application.papers.artifact_cleanup.get_active_processing_artifact_prefixes", return_value=set())
@patch("project.application.papers.artifact_cleanup.get_referenced_artifact_paths", return_value=set())
@patch("project.application.papers.artifact_cleanup.list_artifact_objects")
def test_deletes_old_unreferenced_pdf(list_objects, _references, _active, delete_pdf):
    list_objects.side_effect = inventory([{"path": PDF_PATH, "updated_at": OLD}])

    result = service().cleanup()

    assert result.scanned == 1
    assert result.deleted == 1
    delete_pdf.assert_called_once_with(PDF_PATH)


@patch("project.application.papers.artifact_cleanup.delete_paper_pdf")
@patch("project.application.papers.artifact_cleanup.get_active_processing_artifact_prefixes", return_value=set())
@patch("project.application.papers.artifact_cleanup.get_referenced_artifact_paths", return_value={PDF_PATH})
@patch("project.application.papers.artifact_cleanup.list_artifact_objects")
def test_preserves_database_referenced_artifact(list_objects, _references, _active, delete_pdf):
    list_objects.side_effect = inventory([{"path": PDF_PATH, "updated_at": OLD}])

    result = service().cleanup()

    assert result.skipped_referenced == 1
    delete_pdf.assert_not_called()


@patch("project.application.papers.artifact_cleanup.delete_paper_pdf")
@patch(
    "project.application.papers.artifact_cleanup.get_active_processing_artifact_prefixes",
    return_value={"papers/1234.5678v1/"},
)
@patch("project.application.papers.artifact_cleanup.get_referenced_artifact_paths", return_value=set())
@patch("project.application.papers.artifact_cleanup.list_artifact_objects")
def test_preserves_pdf_for_active_processing_prefix(list_objects, _references, _active, delete_pdf):
    list_objects.side_effect = inventory([{"path": PDF_PATH, "updated_at": OLD}])

    result = service().cleanup()

    assert result.skipped_active == 1
    delete_pdf.assert_not_called()


@patch("project.application.papers.artifact_cleanup.delete_paper_pdf")
@patch("project.application.papers.artifact_cleanup.get_active_processing_artifact_prefixes", return_value=set())
@patch("project.application.papers.artifact_cleanup.get_referenced_artifact_paths", return_value=set())
@patch("project.application.papers.artifact_cleanup.list_artifact_objects")
def test_preserves_new_artifact_inside_grace_period(list_objects, _references, _active, delete_pdf):
    list_objects.side_effect = inventory([{"path": PDF_PATH, "updated_at": RECENT}])

    result = service().cleanup()

    assert result.skipped_grace_period == 1
    delete_pdf.assert_not_called()


@patch("project.application.papers.artifact_cleanup.delete_paper_pdf")
@patch("project.application.papers.artifact_cleanup.get_active_processing_artifact_prefixes", return_value=set())
@patch("project.application.papers.artifact_cleanup.get_referenced_artifact_paths", return_value=set())
@patch("project.application.papers.artifact_cleanup.list_artifact_objects")
def test_preserves_artifact_with_missing_or_malformed_timestamp(
    list_objects, _references, _active, delete_pdf
):
    # Storage parsing represents malformed timestamps as None; cleanup must
    # remain conservative for both missing and malformed source values.
    list_objects.side_effect = inventory(
        [
            {"path": PDF_PATH, "updated_at": None},
            {"path": "papers/1234.5678v1/malformed.pdf", "updated_at": None},
        ]
    )

    result = service().cleanup()

    assert result.skipped_unknown_age == 2
    delete_pdf.assert_not_called()


@patch("project.application.papers.artifact_cleanup.delete_paper_pdf")
@patch("project.application.papers.artifact_cleanup.delete_vector_index")
@patch("project.application.papers.artifact_cleanup.get_active_processing_artifact_prefixes", return_value=set())
@patch("project.application.papers.artifact_cleanup.get_referenced_artifact_paths", return_value=set())
@patch("project.application.papers.artifact_cleanup.list_artifact_objects")
def test_empty_inventory_and_empty_references_are_idempotent(
    list_objects, _references, _active, delete_index, delete_pdf
):
    list_objects.side_effect = inventory()

    result = service().cleanup()

    assert result.scanned == 0
    assert result.deleted == 0
    delete_pdf.assert_not_called()
    delete_index.assert_not_called()


@patch("project.application.papers.artifact_cleanup.delete_paper_pdf", side_effect=RuntimeError("storage down"))
@patch("project.application.papers.artifact_cleanup.get_active_processing_artifact_prefixes", return_value=set())
@patch("project.application.papers.artifact_cleanup.get_referenced_artifact_paths", return_value=set())
@patch("project.application.papers.artifact_cleanup.list_artifact_objects")
def test_storage_deletion_failure_is_reported_without_claiming_deletion(
    list_objects, _references, _active, _delete_pdf
):
    list_objects.side_effect = inventory([{"path": PDF_PATH, "updated_at": OLD}])

    result = service().cleanup()

    assert result.deleted == 0
    assert result.failures == 1


@patch("project.application.papers.artifact_cleanup.delete_paper_pdf")
@patch("project.application.papers.artifact_cleanup.get_active_processing_artifact_prefixes", return_value=set())
@patch(
    "project.application.papers.artifact_cleanup.get_referenced_artifact_paths",
    side_effect=RuntimeError("database unavailable"),
)
@patch("project.application.papers.artifact_cleanup.list_artifact_objects")
def test_database_reference_failure_aborts_before_any_deletion(
    list_objects, _references, _active, delete_pdf
):
    list_objects.return_value = [{"path": PDF_PATH, "updated_at": OLD}]

    with pytest.raises(RuntimeError, match="database unavailable"):
        service().cleanup()

    delete_pdf.assert_not_called()


@patch("project.application.papers.artifact_cleanup.delete_paper_pdf")
@patch("project.application.papers.artifact_cleanup.get_active_processing_artifact_prefixes", return_value=set())
@patch("project.application.papers.artifact_cleanup.get_referenced_artifact_paths")
@patch("project.application.papers.artifact_cleanup.list_artifact_objects")
def test_rechecks_reference_immediately_before_deletion(
    list_objects, references, _active, delete_pdf
):
    list_objects.side_effect = inventory([{"path": PDF_PATH, "updated_at": OLD}])
    references.side_effect = [set(), {PDF_PATH}]

    result = service().cleanup()

    assert result.skipped_referenced == 1
    delete_pdf.assert_not_called()


@patch("project.application.papers.artifact_cleanup.delete_paper_pdf")
@patch("project.application.papers.artifact_cleanup.get_active_processing_artifact_prefixes")
@patch("project.application.papers.artifact_cleanup.get_referenced_artifact_paths", return_value=set())
@patch("project.application.papers.artifact_cleanup.list_artifact_objects")
def test_rechecks_active_prefix_immediately_before_deletion(
    list_objects, _references, active_prefixes, delete_pdf
):
    list_objects.side_effect = inventory([{"path": PDF_PATH, "updated_at": OLD}])
    active_prefixes.side_effect = [set(), {"papers/1234.5678v1/"}]

    result = service().cleanup()

    assert result.skipped_active == 1
    delete_pdf.assert_not_called()


@patch("project.application.papers.artifact_cleanup.delete_paper_pdf")
@patch("project.application.papers.artifact_cleanup.get_active_processing_artifact_prefixes", return_value=set())
@patch("project.application.papers.artifact_cleanup.get_referenced_artifact_paths", return_value=set())
@patch("project.application.papers.artifact_cleanup.list_artifact_objects")
def test_logs_cleanup_summary(list_objects, _references, _active, _delete_pdf, caplog):
    list_objects.side_effect = inventory([{"path": PDF_PATH, "updated_at": RECENT}])
    caplog.set_level("INFO")

    service().cleanup()

    assert any("artifact_cleanup_complete" in record.message for record in caplog.records)


@patch("project.application.papers.artifact_cleanup.delete_vector_index")
@patch(
    "project.application.papers.artifact_cleanup.get_active_processing_artifact_prefixes",
    return_value={"vector_indexes/12345678-1234-5678-1234-567812345678/"},
)
@patch("project.application.papers.artifact_cleanup.get_referenced_artifact_paths", return_value=set())
@patch("project.application.papers.artifact_cleanup.list_artifact_objects")
def test_preserves_versioned_index_for_active_processing_prefix(
    list_objects, _references, _active, delete_index
):
    list_objects.side_effect = inventory(index_objects=[{"path": INDEX_PATH, "updated_at": OLD}])

    result = service().cleanup()

    assert result.skipped_active == 1
    delete_index.assert_not_called()
