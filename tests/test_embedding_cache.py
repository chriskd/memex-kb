from datetime import datetime

from memex.indexer.chroma_index import ChromaIndex
from memex.indexer.embedding_cache import EmbeddingCache, hash_embedding_text
from memex.models import DocumentChunk, EntryMetadata


class DummyCollection:
    def __init__(self) -> None:
        self.upserts: list[dict[str, object]] = []

    def upsert(self, ids, embeddings, documents, metadatas) -> None:
        self.upserts.append(
            {
                "ids": ids,
                "embeddings": embeddings,
                "documents": documents,
                "metadatas": metadatas,
            }
        )


def test_embedding_cache_roundtrip(tmp_path, monkeypatch) -> None:
    index_root = tmp_path / ".indices"
    index_root.mkdir()
    monkeypatch.setenv("MEMEX_INDEX_ROOT", str(index_root))

    cache = EmbeddingCache()
    embedding_hash = hash_embedding_text("hello world")
    cache.set_many({embedding_hash: [0.1, 0.2, 0.3]})

    cached = cache.get_many([embedding_hash])
    assert cached[embedding_hash] == [0.1, 0.2, 0.3]


def test_chroma_index_reuses_embedding_cache(tmp_path, monkeypatch) -> None:
    index_root = tmp_path / ".indices"
    index_root.mkdir()
    monkeypatch.setenv("MEMEX_INDEX_ROOT", str(index_root))

    chroma = ChromaIndex(index_dir=index_root / "chroma")
    dummy = DummyCollection()
    monkeypatch.setattr(chroma, "_get_collection", lambda: dummy)

    calls = {"count": 0}

    def fake_embed(texts: list[str]) -> list[list[float]]:
        calls["count"] += 1
        return [[float(len(text))] for text in texts]

    monkeypatch.setattr(chroma, "_embed", fake_embed)

    metadata = EntryMetadata(
        title="Doc",
        tags=["test"],
        created=datetime(2024, 1, 1),
    )
    chunk = DocumentChunk(
        path="doc.md",
        section="Section",
        content="Hello world",
        metadata=metadata,
        token_count=2,
    )

    chroma.index_documents([chunk])
    assert calls["count"] == 1

    chroma.index_documents([chunk])
    assert calls["count"] == 1

    assert dummy.upserts
    last = dummy.upserts[-1]
    metadatas = last.get("metadatas")
    assert isinstance(metadatas, list)
    assert metadatas
    metadata0 = metadatas[0]
    assert isinstance(metadata0, dict)
    assert "embedding_hash" in metadata0
