"""API tests via FastAPI TestClient with the engine swapped for an offline one."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import ragassistant.api as api_module
from ragassistant.embeddings import HashingEmbedder
from ragassistant.llm import EchoLLM
from ragassistant.rag import RAGEngine
from ragassistant.store import VectorStore


@pytest.fixture
def client(monkeypatch):
    # Fresh in-memory engine with the offline LLM; skip disk persistence in tests.
    engine = RAGEngine(HashingEmbedder(1024), VectorStore(1024), EchoLLM())
    monkeypatch.setattr(engine, "save", lambda *a, **k: None)
    monkeypatch.setattr(api_module, "engine", engine)
    return TestClient(api_module.app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_ingest_then_query(client):
    ingest = client.post(
        "/ingest",
        json={"source": "policy.md", "text": "Refunds are issued within 5 business days."},
    )
    assert ingest.status_code == 200
    assert ingest.json()["chunks_added"] >= 1

    resp = client.post("/query", json={"question": "How long for a refund?", "top_k": 2})
    assert resp.status_code == 200
    body = resp.json()
    assert body["citations"]
    assert body["citations"][0]["source"] == "policy.md"


def test_query_validation_error(client):
    resp = client.post("/query", json={"question": "", "top_k": 2})
    assert resp.status_code == 422
