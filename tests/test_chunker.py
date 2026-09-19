import pytest
from project.nodes.chunker import TextChunk, TextChunker
from project.nodes.pdf import PDFPage

def test_chunker_creates_single_chunk_for_short_text():
    pages = [PDFPage(page=1, text="This is a short piece of text.")]

    chunker = TextChunker(chunk_size=100, chunk_overlap=20)

    chunks = chunker.chunk(pages)

    assert len(chunks) == 1
    assert chunks[0].text == "This is a short piece of text."
    assert chunks[0].page == 1
    assert chunks[0].chunk_id == 0


def test_chunker_splits_long_text():
    text = "word " * 100

    pages = [PDFPage(page=1, text=text)]

    chunker = TextChunker(chunk_size=100, chunk_overlap=20)

    chunks = chunker.chunk(pages)

    assert len(chunks) > 1
    assert all(chunk.page == 1 for chunk in chunks)
    assert [chunk.chunk_id for chunk in chunks] == list(range(len(chunks)))


def test_chunker_preserves_page_numbers():
    pages = [PDFPage(page=1, text="This is page one."), PDFPage(page=2, text="This is page two.")]

    chunker = TextChunker(chunk_size=100, chunk_overlap=20)

    chunks = chunker.chunk(pages)

    assert len(chunks) == 2

    assert chunks[0].page == 1
    assert chunks[0].text == "This is page one."

    assert chunks[1].page == 2
    assert chunks[1].text == "This is page two."


def test_chunker_skips_empty_pages():
    pages = [PDFPage(page=1, text=""), PDFPage(page=2, text="   \n\n   "), PDFPage(page=3, text="Actual content.")]

    chunker = TextChunker()

    chunks = chunker.chunk(pages)

    assert len(chunks) == 1
    assert chunks[0].page == 3
    assert chunks[0].text == "Actual content."
    assert chunks[0].chunk_id == 0


def test_chunker_normalizes_whitespace():
    pages = [PDFPage(page=1, text="This   is\n\nsome\ttext.")]

    chunker = TextChunker()

    chunks = chunker.chunk(pages)

    assert chunks[0].text == "This is some text."


def test_chunker_creates_overlap():
    text = " ".join(f"word{i}" for i in range(100))

    pages = [PDFPage(page=1, text=text)]

    chunker = TextChunker(chunk_size=100, chunk_overlap=20)

    chunks = chunker.chunk(pages)

    assert len(chunks) > 1

    first_chunk_words = set(chunks[0].text.split())
    second_chunk_words = set(chunks[1].text.split())

    assert first_chunk_words & second_chunk_words


def test_chunker_rejects_invalid_chunk_size():
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        TextChunker(chunk_size=0)


def test_chunker_rejects_negative_overlap():
    with pytest.raises(ValueError, match="chunk_overlap cannot be negative"):
        TextChunker(chunk_size=100, chunk_overlap=-1)


def test_chunker_rejects_overlap_larger_than_chunk_size():
    with pytest.raises(ValueError, match="chunk_overlap must be smaller than chunk_size"):
        TextChunker(chunk_size=100, chunk_overlap=100)


def test_chunker_handles_empty_input():
    chunker = TextChunker()

    assert chunker.chunk([]) == []


def test_chunk_ids_are_global_across_pages():
    pages = [PDFPage(page=1, text="First page."), PDFPage(page=2, text="Second page."), PDFPage(page=3, text="Third page.")]

    chunker = TextChunker()

    chunks = chunker.chunk(pages)

    assert [chunk.chunk_id for chunk in chunks] == [0, 1, 2]
