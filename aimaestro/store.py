"""SQLite-backed persistence: long-term memory plus conversation threads."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore


def _connect(db_path: str) -> sqlite3.Connection:
    """Open a connection configured the way langgraph's SQLite backends expect."""
    return sqlite3.connect(
        db_path,
        check_same_thread=False,
        # Autocommit. SqliteStore issues its own BEGIN, and without this every
        # write fails with "cannot start a transaction within a transaction".
        isolation_level=None,
    )


class MemoryBackend:
    """Owns the database connections for one aiMaestro session.

    Two things live in the same file: the store (what aiMaestro remembers about
    you, indefinitely) and the checkpointer (the back-and-forth of a single
    conversation). Use as a context manager so connections always close::

        with MemoryBackend("data/aimaestro.db") as backend:
            backend.store.put(...)
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # Separate connections: the store and the checkpointer each manage
        # their own cursors and transaction state.
        self._store_conn = _connect(db_path)
        self._saver_conn = _connect(db_path)

        self.store = SqliteStore(self._store_conn)
        self.store.setup()

        self.checkpointer = SqliteSaver(self._saver_conn)
        self.checkpointer.setup()

    def close(self) -> None:
        self._store_conn.close()
        self._saver_conn.close()

    def __enter__(self) -> "MemoryBackend":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
