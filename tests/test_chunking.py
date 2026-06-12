"""Tests for paragraph-aware overlapping chunking."""

from __future__ import annotations

from ragassistant.chunking import chunk_text


def test_chunks_respect_size_budget():
    text = " ".join(f"Sentence number {i}." for i in range(200))
    chunks = chunk_text(text, "doc.md", chunk_size=200, overlap=40)
    assert len(chunks) > 1
    # Allow a little slack since we never split a sentence.
    assert all(len(c.text) <= 260 for c in chunks)


def test_chunks_overlap_for_context():
    text = " ".join(f"S{i} word word word." for i in range(60))
    chunks = chunk_text(text, "doc.md", chunk_size=120, overlap=40)
    # Consecutive chunks should share some trailing/leading text.
    first_tokens = set(chunks[0].text.split())
    second_tokens = set(chunks[1].text.split())
    assert first_tokens & second_tokens


def test_indices_are_sequential():
    text = "a. " * 500
    chunks = chunk_text(text, "doc.md", chunk_size=100, overlap=20)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_overlap_must_be_smaller_than_chunk():
    import pytest

    with pytest.raises(ValueError):
        chunk_text("hello", "d", chunk_size=50, overlap=50)
