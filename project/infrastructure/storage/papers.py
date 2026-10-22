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
