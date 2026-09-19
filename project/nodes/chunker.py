import re
from dataclasses import dataclass

from project.nodes.pdf import PDFPage


@dataclass
class TextChunk:
    chunk_id: int
    page: int
    text: str


class TextChunker:
    def __init__(
        self,
        chunk_size: int = 1500,
        chunk_overlap: int = 200,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")

        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, pages: list[PDFPage]) -> list[TextChunk]:
        chunks = []

        for page in pages:
            text = self._normalize_text(page.text)

            if not text:
                continue

            page_chunks = self._chunk_text(text)

            for text_chunk in page_chunks:
                chunks.append(
                    TextChunk(
                        chunk_id=len(chunks),
                        page=page.page,
                        text=text_chunk,
                    )
                )

        return chunks

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def _chunk_text(self, text: str) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            if end >= len(text):
                chunks.append(text[start:].strip())
                break

            split_at = text.rfind(" ", start, end)

            if split_at <= start:
                split_at = end

            chunks.append(text[start:split_at].strip())

            start = split_at - self.chunk_overlap

        return chunks
