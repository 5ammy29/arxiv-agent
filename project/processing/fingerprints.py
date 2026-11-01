"""Stable fingerprints for persisted paper-processing artifacts."""

from __future__ import annotations

import hashlib
import json
from typing import Iterable

from project.processing.pdf import PDFPage


CHUNKING_ALGORITHM_VERSION = "page-word-boundary-v1"


def _digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def content_fingerprint(pages: Iterable[PDFPage]) -> str:
    """Fingerprint ordered, extracted page content without lossy joining."""
    return _digest(
        [
            {"page_number": page.page, "text": page.text}
            for page in pages
        ]
    )


def chunking_fingerprint(
    *,
    chunk_size: int,
    chunk_overlap: int,
    algorithm_version: str = CHUNKING_ALGORITHM_VERSION,
) -> str:
    """Fingerprint every input that can change chunk boundaries or text."""
    return _digest(
        {
            "algorithm_version": algorithm_version,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "normalization": "whitespace-collapse-v1",
        }
    )
