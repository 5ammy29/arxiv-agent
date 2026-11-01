from project.processing.chunker import TextChunker
from project.processing.fingerprints import chunking_fingerprint, content_fingerprint
from project.processing.pdf import PDFPage


def test_content_fingerprint_is_deterministic_and_order_sensitive():
    pages = [PDFPage(page=1, text="First"), PDFPage(page=2, text="Second")]

    assert content_fingerprint(pages) == content_fingerprint(list(pages))
    assert content_fingerprint(pages) != content_fingerprint(list(reversed(pages)))
    assert content_fingerprint(pages) != content_fingerprint(
        [PDFPage(page=1, text="Changed"), PDFPage(page=2, text="Second")]
    )


def test_chunking_fingerprint_covers_all_chunking_inputs():
    current = TextChunker(chunk_size=500, chunk_overlap=50)

    assert current.configuration_fingerprint == chunking_fingerprint(
        chunk_size=500,
        chunk_overlap=50,
    )
    assert current.configuration_fingerprint != TextChunker(
        chunk_size=501, chunk_overlap=50
    ).configuration_fingerprint
    assert current.configuration_fingerprint != TextChunker(
        chunk_size=500, chunk_overlap=49
    ).configuration_fingerprint
    assert current.configuration_fingerprint != TextChunker(
        chunk_size=500,
        chunk_overlap=50,
        algorithm_version="page-word-boundary-v2",
    ).configuration_fingerprint
