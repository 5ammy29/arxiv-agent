"""Fresh-process persistence recovery with PostgreSQL and local object storage.

These tests deliberately patch only the storage boundary.  The registry,
artifact/content/index repositories and processing leases use the local
PostgreSQL integration database so a recreated service cannot accidentally
reuse an in-memory database double.
"""

from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import faiss
import numpy as np

from project.application.papers.prepare_paper import PaperPreparationService
from project.application.retrieval.vector_index_cache import VectorIndexCache
from project.infrastructure.database.connection import (
    close_connection_pool,
    get_connection_pool,
)
from project.infrastructure.database.papers import (
    get_paper_artifact,
    get_paper_content,
    get_paper,
)
from project.infrastructure.database.vector_indexes import get_paper_vector_index
from project.processing.chunker import TextChunker
from project.processing.pdf import PDFPage


pytestmark = pytest.mark.integration


class LocalObjectStorage:
    """Small filesystem-backed stand-in for the application storage boundary."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.pdf_uploads = 0
        self.index_uploads = 0

    def upload_pdf(self, source: str | Path, arxiv_id: str) -> str:
        self.pdf_uploads += 1
        path = f"papers/{arxiv_id}/pdf-{self.pdf_uploads}.pdf"
        destination = self.root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(Path(source).read_bytes())
        return path

    def download_pdf(self, storage_path: str, destination: str | Path) -> Path:
        source = self.root / storage_path
        if not source.exists():
            raise FileNotFoundError(storage_path)
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        return destination

    def delete_pdf(self, storage_path: str) -> None:
        (self.root / storage_path).unlink(missing_ok=True)

    def upload_index(self, index_bytes: bytes, paper_id) -> str:
        self.index_uploads += 1
        path = f"vector_indexes/{paper_id}/index-{self.index_uploads}.faiss"
        destination = self.root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(index_bytes)
        return path

    def download_index(self, storage_path: str) -> bytes:
        path = self.root / storage_path
        if not path.exists():
            raise FileNotFoundError(storage_path)
        return path.read_bytes()

    def delete_index(self, storage_path: str) -> None:
        (self.root / storage_path).unlink(missing_ok=True)


class FakeEmbeddingModel:
    def get_sentence_embedding_dimension(self) -> int:
        return 3


class FakeVectorStore:
    """Tracks whether recovery loaded persisted bytes or re-embedded chunks."""

    model_name = "persistence-recovery-model"

    def __init__(self) -> None:
        self.model = FakeEmbeddingModel()
        self.add_calls = 0
        self.load_calls = 0
        self.chunks = []
        self.index = faiss.IndexFlatIP(3)

    def reset(self) -> None:
        self.chunks = []
        self.index = faiss.IndexFlatIP(3)

    def add_chunks(self, chunks) -> None:
        self.add_calls += 1
        self.chunks = list(chunks)
        embeddings = np.array(
            [[float(chunk.chunk_id + 1), 1.0, 0.5] for chunk in chunks],
            dtype=np.float32,
        )
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

    def serialize_index(self) -> bytes:
        return faiss.serialize_index(self.index).tobytes()

    def load_index(self, index_bytes: bytes, chunks, chunk_ids) -> None:
        restored = faiss.deserialize_index(np.frombuffer(index_bytes, dtype=np.uint8))
        if restored.ntotal != len(chunk_ids) or restored.d != 3:
            raise ValueError("corrupt test index")
        self.load_calls += 1
        self.index = restored
        chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        self.chunks = [chunk_by_id[chunk_id] for chunk_id in chunk_ids]


def _processor(output_dir: Path, pages: list[PDFPage]) -> Mock:
    processor = Mock()
    processor.output_dir = output_dir
    processor.parse_pdf.return_value = pages
    return processor


def _download_to(path: Path):
    def download(_url: str, _arxiv_id: str) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"%PDF-local-test")
        return path

    return download


@contextmanager
def _storage_patches(storage: LocalObjectStorage):
    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "project.application.papers.prepare_paper.upload_paper_pdf",
                storage.upload_pdf,
            )
        )
        stack.enter_context(
            patch(
                "project.application.papers.prepare_paper.download_paper_pdf",
                storage.download_pdf,
            )
        )
        stack.enter_context(
            patch(
                "project.application.papers.prepare_paper.delete_paper_pdf",
                storage.delete_pdf,
            )
        )
        stack.enter_context(
            patch(
                "project.application.retrieval.vector_index_cache.upload_vector_index",
                storage.upload_index,
            )
        )
        stack.enter_context(
            patch(
                "project.application.retrieval.vector_index_cache.download_vector_index",
                storage.download_index,
            )
        )
        stack.enter_context(
            patch(
                "project.application.retrieval.vector_index_cache.delete_vector_index",
                storage.delete_index,
            )
        )
        yield


def test_fresh_instances_restore_persisted_artifacts_and_faiss_index(
    integration_paper, tmp_path
):
    paper_id, arxiv_id = integration_paper()
    pages = [PDFPage(page=1, text="Persistent page text for recovery.")]
    storage = LocalObjectStorage(tmp_path / "objects")
    first_pdf = tmp_path / "first" / "paper.pdf"
    first_processor = _processor(tmp_path / "first", pages)
    first_processor.download_pdf.side_effect = _download_to(first_pdf)
    chunker = TextChunker(chunk_size=100, chunk_overlap=10)

    with _storage_patches(storage):
        first_result = PaperPreparationService(first_processor, chunker).prepare(
            "https://example.invalid/original.pdf", arxiv_id, paper_id
        )
        assert first_result.error is None
        assert first_result.chunks

        first_store = FakeVectorStore()
        assert VectorIndexCache().get_or_build_index(
            paper_id,
            first_result.chunks,
            first_store,
            content_fingerprint=first_result.content_fingerprint,
            chunking_fingerprint=first_result.chunking_fingerprint,
        ) is False

        artifact_before = get_paper_artifact(paper_id)
        content_before = get_paper_content(paper_id)
        index_before = get_paper_vector_index(paper_id)
        paper_before = get_paper(arxiv_id)
        assert artifact_before is not None
        assert index_before is not None
        assert paper_before is not None

        # A new application process receives a new connection pool, services,
        # processors and vector store. Only PostgreSQL and object storage remain.
        first_pool = get_connection_pool()
        close_connection_pool()
        second_processor = _processor(tmp_path / "second", pages)
        second_service = PaperPreparationService(
            second_processor, TextChunker(100, 10)
        )
        second_result = second_service.prepare(
            "https://example.invalid/original.pdf", arxiv_id, paper_id
        )

        assert second_result.error is None
        assert get_connection_pool() is not first_pool
        assert second_result.chunks == first_result.chunks
        second_processor.download_pdf.assert_not_called()
        second_processor.parse_pdf.assert_called_once()
        assert get_paper_artifact(paper_id) == artifact_before
        assert get_paper_content(paper_id) == content_before
        paper_after = get_paper(arxiv_id)

        assert paper_after is not None

        assert {
            key: value
            for key, value in paper_after.items()
            if key != "updated_at"
        } == {
            key: value
            for key, value in paper_before.items()
            if key != "updated_at"
        }

        close_connection_pool()
        second_store = FakeVectorStore()
        assert VectorIndexCache().get_or_build_index(
            paper_id,
            second_result.chunks,
            second_store,
            content_fingerprint=second_result.content_fingerprint,
            chunking_fingerprint=second_result.chunking_fingerprint,
        ) is True

    assert second_store.load_calls == 1
    assert second_store.add_calls == 0
    assert storage.pdf_uploads == 1
    assert storage.index_uploads == 1
    assert get_paper_vector_index(paper_id) == index_before


def test_missing_persisted_pdf_rebuilds_content_and_index(
    integration_paper, tmp_path
):
    paper_id, arxiv_id = integration_paper()
    initial_pages = [PDFPage(page=1, text="Initial persistent content.")]
    replacement_pages = [PDFPage(page=1, text="Replacement persistent content.")]
    storage = LocalObjectStorage(tmp_path / "objects")
    chunker = TextChunker(chunk_size=100, chunk_overlap=10)
    first_processor = _processor(tmp_path / "first", initial_pages)
    first_processor.download_pdf.side_effect = _download_to(
        tmp_path / "first" / "paper.pdf"
    )

    with _storage_patches(storage):
        first_result = PaperPreparationService(first_processor, chunker).prepare(
            "https://example.invalid/original.pdf", arxiv_id, paper_id
        )
        assert first_result.error is None

        first_store = FakeVectorStore()
        assert VectorIndexCache().get_or_build_index(
            paper_id,
            first_result.chunks,
            first_store,
            content_fingerprint=first_result.content_fingerprint,
            chunking_fingerprint=first_result.chunking_fingerprint,
        ) is False

        artifact = get_paper_artifact(paper_id)
        index = get_paper_vector_index(paper_id)
        assert artifact is not None
        assert index is not None

        storage.delete_pdf(artifact["pdf_storage_path"])

        close_connection_pool()
        second_processor = _processor(tmp_path / "second", replacement_pages)
        second_processor.download_pdf.side_effect = _download_to(
            tmp_path / "second" / "paper.pdf"
        )
        second_result = PaperPreparationService(
            second_processor, TextChunker(100, 10)
        ).prepare("https://example.invalid/original.pdf", arxiv_id, paper_id)

        assert second_result.error is None
        assert second_result.chunks != first_result.chunks
        second_processor.download_pdf.assert_called_once()
        second_processor.parse_pdf.assert_called_once()

        second_store = FakeVectorStore()
        assert VectorIndexCache().get_or_build_index(
            paper_id,
            second_result.chunks,
            second_store,
            content_fingerprint=second_result.content_fingerprint,
            chunking_fingerprint=second_result.chunking_fingerprint,
        ) is False

    assert second_store.add_calls == 1
    assert second_store.load_calls == 0
    assert storage.pdf_uploads == 2
    assert storage.index_uploads == 2
    assert get_paper_vector_index(paper_id) != index


def test_corrupt_persisted_faiss_is_replaced_after_fresh_content_recovery(
    integration_paper, tmp_path
):
    paper_id, arxiv_id = integration_paper()
    pages = [PDFPage(page=1, text="Persistent content for corrupt index recovery.")]
    storage = LocalObjectStorage(tmp_path / "objects")
    first_processor = _processor(tmp_path / "first", pages)
    first_processor.download_pdf.side_effect = _download_to(
        tmp_path / "first" / "paper.pdf"
    )
    chunker = TextChunker(chunk_size=100, chunk_overlap=10)

    with _storage_patches(storage):
        first_result = PaperPreparationService(first_processor, chunker).prepare(
            "https://example.invalid/original.pdf", arxiv_id, paper_id
        )
        assert first_result.error is None

        first_store = FakeVectorStore()
        assert VectorIndexCache().get_or_build_index(
            paper_id,
            first_result.chunks,
            first_store,
            content_fingerprint=first_result.content_fingerprint,
            chunking_fingerprint=first_result.chunking_fingerprint,
        ) is False

        index_before = get_paper_vector_index(paper_id)
        assert index_before is not None
        storage.root.joinpath(index_before["index_storage_path"]).write_bytes(
            b"corrupt-faiss-index"
        )

        close_connection_pool()
        second_processor = _processor(tmp_path / "second", pages)
        second_result = PaperPreparationService(
            second_processor, TextChunker(100, 10)
        ).prepare("https://example.invalid/original.pdf", arxiv_id, paper_id)

        assert second_result.error is None
        assert second_result.chunks == first_result.chunks
        second_processor.download_pdf.assert_not_called()
        second_processor.parse_pdf.assert_called_once()

        second_store = FakeVectorStore()
        assert VectorIndexCache().get_or_build_index(
            paper_id,
            second_result.chunks,
            second_store,
            content_fingerprint=second_result.content_fingerprint,
            chunking_fingerprint=second_result.chunking_fingerprint,
        ) is False

        rebuilt_index = get_paper_vector_index(paper_id)
        assert rebuilt_index is not None
        assert rebuilt_index["index_storage_path"] != index_before["index_storage_path"]

        close_connection_pool()
        third_store = FakeVectorStore()
        assert VectorIndexCache().get_or_build_index(
            paper_id,
            second_result.chunks,
            third_store,
            content_fingerprint=second_result.content_fingerprint,
            chunking_fingerprint=second_result.chunking_fingerprint,
        ) is True

    assert second_store.add_calls == 1
    assert second_store.load_calls == 0
    assert third_store.add_calls == 0
    assert third_store.load_calls == 1
    assert storage.pdf_uploads == 1
    assert storage.index_uploads == 2
    assert get_paper_vector_index(paper_id) == rebuilt_index
