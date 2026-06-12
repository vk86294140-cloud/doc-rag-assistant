"""Tests for the numpy vector store: search ranking and persistence."""

from __future__ import annotations

from ragassistant.embeddings import HashingEmbedder
from ragassistant.store import VectorStore


def _populate():
    emb = HashingEmbedder(dim=1024)
    store = VectorStore(dim=1024)
    texts = [
        "The cat sat on the warm mat by the fire.",
        "Quarterly revenue grew due to strong cloud sales.",
        "Refunds are issued within five business days.",
    ]
    store.add(texts, emb.embed(texts), [{"source": "d", "chunk_index": i} for i in range(3)])
    return emb, store


def test_query_ranks_relevant_chunk_first():
    emb, store = _populate()
    q = emb.embed(["How long do refunds take?"])[0]
    matches = store.query(q, top_k=3)
    assert matches[0].text.startswith("Refunds are issued")
    assert matches[0].score >= matches[-1].score  # sorted by score desc


def test_persistence_roundtrip(tmp_path):
    emb, store = _populate()
    store.save(tmp_path)
    loaded = VectorStore.load(tmp_path)
    assert len(loaded) == len(store)
    q = emb.embed(["refund"])[0]
    assert loaded.query(q, top_k=1)[0].text.startswith("Refunds")


def test_empty_store_returns_no_matches():
    store = VectorStore(dim=1024)
    emb = HashingEmbedder(dim=1024)
    assert store.query(emb.embed(["anything"])[0]) == []
