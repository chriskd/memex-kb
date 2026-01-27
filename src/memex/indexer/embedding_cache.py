"""Persistent embedding cache for semantic indexing.

Stores embeddings by hash of embedding text. Cache location:
{index_root}/embedding_cache.sqlite
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

from ..config import EMBEDDING_MODEL, get_index_root

CACHE_FILENAME = "embedding_cache.sqlite"
SCHEMA_VERSION = 1


def hash_embedding_text(text: str) -> str:
    """Return a stable hash for embedding text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _chunked(items: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


@dataclass(frozen=True)
class CachedEmbedding:
    hash: str
    embedding: list[float]


class EmbeddingCache:
    """SQLite-backed cache for embedding vectors."""

    def __init__(self, index_root: Path | None = None, model_name: str | None = None) -> None:
        self._index_root = index_root or get_index_root()
        self._path = self._index_root / CACHE_FILENAME
        self._model_name = model_name or EMBEDDING_MODEL

    @property
    def path(self) -> Path:
        return self._path

    def _connect(self) -> sqlite3.Connection:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        self._ensure_schema(conn)
        return conn

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embedding_cache (
                model TEXT NOT NULL,
                hash TEXT NOT NULL,
                embedding TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (model, hash)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embedding_cache_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        row = conn.execute(
            "SELECT value FROM embedding_cache_meta WHERE key = 'schema_version'"
        ).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO embedding_cache_meta (key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
        conn.commit()

    def get_many(self, hashes: list[str]) -> dict[str, list[float]]:
        if not hashes:
            return {}

        results: dict[str, list[float]] = {}
        with self._connect() as conn:
            for batch in _chunked(hashes, 900):
                placeholders = ",".join("?" for _ in batch)
                query = (
                    "SELECT hash, embedding FROM embedding_cache "
                    f"WHERE model = ? AND hash IN ({placeholders})"
                )
                rows = conn.execute(query, [self._model_name, *batch]).fetchall()
                for hash_value, embedding_json in rows:
                    try:
                        results[str(hash_value)] = json.loads(embedding_json)
                    except (TypeError, json.JSONDecodeError):
                        continue
        return results

    def set_many(self, embeddings: dict[str, list[float]]) -> None:
        if not embeddings:
            return

        now = datetime.now(tz=UTC).isoformat()
        rows = [
            (self._model_name, hash_value, json.dumps(vector), now)
            for hash_value, vector in embeddings.items()
        ]
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO embedding_cache (model, hash, embedding, created_at)
                VALUES (?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()

    def clear(self) -> None:
        if self._path.exists():
            self._path.unlink()
