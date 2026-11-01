from pathlib import Path
from uuid import uuid4

from project.infrastructure.storage.connection import (
    get_storage_bucket_client,
)


def upload_paper_pdf(pdf_path: str | Path, arxiv_id: str) -> str:
    path = Path(pdf_path)
    storage_path = f"papers/{arxiv_id}/{uuid4().hex}.pdf"

    bucket = get_storage_bucket_client()

    with path.open("rb") as pdf_file:
        bucket.upload(
            path=storage_path,
            file=pdf_file,
            file_options={
                "content-type": "application/pdf",
                "upsert": "false",
            },
        )

    return storage_path


def download_paper_pdf(
    storage_path: str,
    destination_path: str | Path,
) -> Path:
    path = Path(destination_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    bucket = get_storage_bucket_client()
    pdf_bytes = bucket.download(storage_path)
    path.write_bytes(pdf_bytes)

    return path


def delete_paper_pdf(storage_path: str) -> None:
    if not storage_path.strip():
        raise ValueError("PDF storage path must not be empty.")

    get_storage_bucket_client().remove([storage_path])