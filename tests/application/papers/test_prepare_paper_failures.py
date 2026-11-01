from threading import Event
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest

from project.application.papers.prepare_paper import PaperPreparationService
from project.application.papers.processing_lock import (
    ActivePaperProcessingLease,
    PaperProcessingLockError,
)
from project.infrastructure.database.paper_locks import PaperProcessingLease
from project.processing.chunker import TextChunk
from project.processing.pdf import PDFPage


def test_lease_loss_stops_before_upload():
    paper_id = uuid4()
    processor = Mock()
    processor.download_pdf.return_value = "paper.pdf"
    chunker = Mock()
    chunker.configuration_fingerprint = "x"
    service = PaperPreparationService(processor, chunker)
    lost = Event()
    lost.set()
    active = ActivePaperProcessingLease(PaperProcessingLease(paper_id, uuid4(), 1), lost)

    with patch("project.application.papers.prepare_paper.update_paper_status"), pytest.raises(
        PaperProcessingLockError
    ):
        service._download("url", "1234v1", paper_id, active)

    processor.download_pdf.assert_not_called()


@patch("project.application.papers.prepare_paper.get_paper_content", return_value={"pages": [], "chunks": []})
@patch("project.application.papers.prepare_paper.update_paper_status")
@patch("project.application.papers.prepare_paper.save_paper_content", side_effect=RuntimeError("db"))
def test_content_persistence_failure_marks_failed(_save, status, _content):
    processor = Mock()
    processor.parse_pdf.return_value = [PDFPage(page=1, text="text")]
    chunker = Mock()
    chunker.configuration_fingerprint = "x"
    chunker.chunk.return_value = [TextChunk(chunk_id=0, page=1, text="text")]

    result = PaperPreparationService(processor, chunker).parse_and_chunk("paper.pdf", "1234v1", uuid4())

    assert result.error == "Failed to prepare paper content"
    assert [call.args[1] for call in status.call_args_list] == ["parsing", "failed"]
