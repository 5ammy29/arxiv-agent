import pytest

from project.nodes.query import QueryProcessor


def test_process_query():
    processor = QueryProcessor()

    query = processor.process("What datasets were used?")

    assert query == "What datasets were used?"


def test_process_query_strips_whitespace():
    processor = QueryProcessor()

    query = processor.process("   What datasets were used?   ")

    assert query == "What datasets were used?"


def test_process_query_normalizes_whitespace():
    processor = QueryProcessor()

    query = processor.process(
        "What    datasets\nwere\tused?"
    )

    assert query == "What datasets were used?"


def test_process_empty_query():
    processor = QueryProcessor()

    with pytest.raises(ValueError, match="query cannot be empty"):
        processor.process("")


def test_process_whitespace_only_query():
    processor = QueryProcessor()

    with pytest.raises(ValueError, match="query cannot be empty"):
        processor.process("   ")


def test_process_non_string_query():
    processor = QueryProcessor()

    with pytest.raises(TypeError, match="query must be a string"):
        processor.process(None)


def test_process_integer_query():
    processor = QueryProcessor()

    with pytest.raises(TypeError, match="query must be a string"):
        processor.process(123)
