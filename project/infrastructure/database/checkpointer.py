from contextlib import contextmanager
from typing import Generator

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from project.config import get_database_url


@contextmanager
def get_checkpointer() -> Generator[PostgresSaver, None, None]:
    serializer = JsonPlusSerializer(
        allowed_msgpack_modules=[
            ("project.nodes.chunker", "TextChunk"),
            ("project.nodes.arxiv", "Paper"),
            ("project.nodes.vector_store", "SearchResult"),
        ]
    )

    with PostgresSaver.from_conn_string(get_database_url()) as saver:
        checkpointer = PostgresSaver(
            conn=saver.conn,
            pipe=saver.pipe,
            serde=serializer,
        )
        checkpointer.setup()
        yield checkpointer
