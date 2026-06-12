"""Split documents into overlapping, paragraph-aware chunks.

Chunking strategy matters more for RAG quality than the embedding model: we
prefer to break on paragraph and sentence boundaries so a chunk is a coherent
unit of meaning, and we overlap consecutive chunks so an answer that straddles
a boundary is still retrievable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List


@dataclass
class Chunk:
    text: str
    source: str
    index: int


_PARAGRAPH = re.compile(r"\n\s*\n")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def _split_units(text: str) -> List[str]:
    """Split into paragraphs, then sentences, dropping empties."""
    units: List[str] = []
    for para in _PARAGRAPH.split(text):
        para = para.strip()
        if not para:
            continue
        for sentence in _SENTENCE.split(para):
            sentence = sentence.strip()
            if sentence:
                units.append(sentence)
    return units


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 800,
    overlap: int = 150,
) -> List[Chunk]:
    """Greedily pack sentence units into ~chunk_size character chunks."""

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    units = _split_units(text)
    chunks: List[Chunk] = []
    current: List[str] = []
    current_len = 0
    idx = 0

    for unit in units:
        unit_len = len(unit) + 1
        if current and current_len + unit_len > chunk_size:
            chunk_str = " ".join(current).strip()
            chunks.append(Chunk(text=chunk_str, source=source, index=idx))
            idx += 1
            # Carry over a tail of the previous chunk for context overlap.
            current = _tail(current, overlap)
            current_len = sum(len(u) + 1 for u in current)
        current.append(unit)
        current_len += unit_len

    if current:
        chunks.append(Chunk(text=" ".join(current).strip(), source=source, index=idx))

    return chunks


def _tail(units: List[str], overlap: int) -> List[str]:
    """Return the trailing units whose combined length is ~`overlap` chars."""
    tail: List[str] = []
    total = 0
    for unit in reversed(units):
        if total >= overlap:
            break
        tail.insert(0, unit)
        total += len(unit) + 1
    return tail
