"""Shared fixtures: an in-memory RAG engine wired to the offline EchoLLM."""

from __future__ import annotations

import pytest

from ragassistant.embeddings import HashingEmbedder
from ragassistant.llm import EchoLLM
from ragassistant.rag import RAGEngine
from ragassistant.store import VectorStore

DOCS = {
    "returns.md": (
        "Items may be returned within 30 days of delivery for a full refund. "
        "Final-sale items are not eligible for return. Refunds are issued to the "
        "original payment method within 5 business days."
    ),
    "shipping.md": (
        "Standard domestic shipping takes 3 to 5 business days and is free on "
        "orders over 50 dollars. International orders ship within 7 to 14 days."
    ),
}


@pytest.fixture
def engine() -> RAGEngine:
    embedder = HashingEmbedder(dim=1024)
    eng = RAGEngine(embedder=embedder, store=VectorStore(dim=1024), llm=EchoLLM())
    for source, text in DOCS.items():
        eng.ingest_text(text, source)
    return eng
