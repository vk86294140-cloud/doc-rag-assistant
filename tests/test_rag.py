"""End-to-end RAG tests using the offline EchoLLM (no API key needed)."""

from __future__ import annotations


def test_retrieval_finds_relevant_source(engine):
    matches = engine.retrieve("How many days do I have to return an item?", top_k=2)
    assert matches
    assert matches[0].metadata["source"] == "returns.md"


def test_answer_includes_citations(engine):
    result = engine.answer("When are refunds issued?")
    assert result.citations
    # The top citation should come from the returns document.
    assert result.citations[0].source == "returns.md"
    assert "[offline answer]" in result.answer  # EchoLLM marker


def test_answer_without_documents_is_honest():
    from ragassistant.embeddings import HashingEmbedder
    from ragassistant.llm import EchoLLM
    from ragassistant.rag import RAGEngine
    from ragassistant.store import VectorStore

    empty = RAGEngine(HashingEmbedder(1024), VectorStore(1024), EchoLLM())
    result = empty.answer("anything?")
    assert result.citations == []
    assert "don't have" in result.answer.lower()


def test_shipping_question_routes_to_shipping_doc(engine):
    matches = engine.retrieve("Is domestic shipping free over fifty dollars?", top_k=1)
    assert matches[0].metadata["source"] == "shipping.md"
