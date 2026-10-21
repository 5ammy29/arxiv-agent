from contextlib import contextmanager
from threading import Lock
from typing import Generator

from psycopg import Connection
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from project.config import get_database_url


connection_pool: ConnectionPool | None = None
pool_lock = Lock()     # synchronizes pool initialization and shutdown across threads


def get_connection_pool() -> ConnectionPool:
    global connection_pool

    if connection_pool is None:
        with pool_lock:     # only one thread at a time can initialize the shared pool
            if connection_pool is None:     # recheck: another thread may have initialized the pool while we waited
                pool = ConnectionPool(
                    conninfo=get_database_url(),
                    min_size=1,
                    max_size=10,
                    timeout=30,
                    open=False,
                    kwargs={"row_factory": dict_row},
                )
                pool.open(wait=True)
                connection_pool = pool

    return connection_pool


@contextmanager
def get_connection() -> Generator[Connection, None, None]:
    pool = get_connection_pool()

    with pool.connection() as connection:
        yield connection


def close_connection_pool() -> None:
    global connection_pool

    with pool_lock:
        if connection_pool is not None:
            connection_pool.close()
            connection_pool = None
