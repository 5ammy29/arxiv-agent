from pathlib import Path

from project.infrastructure.storage.connection import (
    get_storage_bucket_client,
)


def upload_paper_pdf(pdf_path: str | Path, arxiv_id: str) -> str:
    """Upload a paper PDF and return its storage path."""
    path = Path(pdf_path)
    storage_path = f"papers/{arxiv_id}.pdf"
    bucket = get_storage_bucket_client()

    with path.open("rb") as pdf_file:
        bucket.upload(
            path=storage_path,
            file=pdf_file,
            file_options={"content-type": "application/pdf", "upsert": "true"},
        )

    return storage_path


def download_paper_pdf(storage_path: str, destination_path: str | Path) -> Path:
    """Download a stored paper PDF to a local path."""
    path = Path(destination_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    bucket = get_storage_bucket_client()
    pdf_bytes = bucket.download(storage_path)
    path.write_bytes(pdf_bytes)

    return path
