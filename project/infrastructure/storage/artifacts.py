"""Conservative storage inventory helpers used only by artifact cleanup."""

from datetime import datetime
from typing import Any

from project.infrastructure.storage.connection import get_storage_bucket_client


def list_artifact_objects(prefix: str) -> list[dict[str, Any]]:
    """Recursively list objects below a known managed prefix.

    Supabase Storage returns directory entries without an object id.  Unknown
    entries are treated as directories and traversed; malformed timestamps are
    deliberately retained so cleanup can skip them conservatively.
    """
    if not prefix or prefix.startswith("/") or ".." in prefix.split("/"):
        raise ValueError("Artifact prefix must be a relative managed prefix.")

    bucket = get_storage_bucket_client()
    objects: list[dict[str, Any]] = []
    pending = [prefix.rstrip("/")]
    visited_directories: set[str] = set()
    seen_objects: set[str] = set()
    page_size = 1000

    while pending:
        current = pending.pop()
        if current in visited_directories:
            continue
        visited_directories.add(current)

        offset = 0
        while True:
            entries = bucket.list(
                path=current,
                options={"limit": page_size, "offset": offset},
            )
            if not entries:
                break

            for entry in entries:
                name = entry.get("name") if isinstance(entry, dict) else None
                if not isinstance(name, str) or not name:
                    continue
                path = f"{current}/{name}" if current else name
                if isinstance(entry, dict) and entry.get("id"):
                    if path not in seen_objects:
                        seen_objects.add(path)
                        objects.append(
                            {
                                "path": path,
                                "updated_at": _parse_timestamp(
                                    entry.get("updated_at")
                                    or entry.get("created_at")
                                ),
                            }
                        )
                elif path not in visited_directories:
                    pending.append(path)

            if len(entries) < page_size:
                break
            offset += len(entries)

    return objects


def _parse_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
