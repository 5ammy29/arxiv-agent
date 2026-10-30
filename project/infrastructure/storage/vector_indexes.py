from uuid import UUID

from project.infrastructure.storage.connection import (
    get_storage_bucket_client,
)


def upload_vector_index(index_bytes: bytes, paper_id: UUID) -> str:
    """Upload a serialized FAISS index and return its storage path."""
    if not index_bytes:
        raise ValueError("Vector index data must not be empty.")

    storage_path = f"vector_indexes/{paper_id}.faiss"
    bucket = get_storage_bucket_client()

    bucket.upload(
        path=storage_path,
        file=index_bytes,
        file_options={
            "content-type": "application/octet-stream",
            "upsert": "true",
        },
    )

    return storage_path


def download_vector_index(storage_path: str) -> bytes:
    """Download a serialized FAISS index from storage."""
    if not storage_path.strip():
        raise ValueError("Vector index storage path must not be empty.")

    bucket = get_storage_bucket_client()
    return bucket.download(storage_path)
