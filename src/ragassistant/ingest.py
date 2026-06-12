"""CLI to ingest documents into the persisted vector store.

    python -m ragassistant.ingest sample_docs/
    python -m ragassistant.ingest path/to/file.md
"""

from __future__ import annotations

import argparse
import sys

from .config import settings
from .rag import RAGEngine
from .store import VectorStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="file or directory of .md/.txt documents")
    parser.add_argument("--persist", default=settings.persist_dir)
    args = parser.parse_args(argv)

    # Continue an existing index if present, else start fresh.
    if VectorStore.exists(args.persist):
        engine = RAGEngine.load(args.persist)
    else:
        engine = RAGEngine()

    added = engine.ingest_path(args.path)
    engine.save(args.persist)

    print(f"Ingested {added} chunks from {args.path}")
    print(f"Index now holds {len(engine.store)} chunks at {args.persist}")
    return 0 if added else 1


if __name__ == "__main__":
    raise SystemExit(main())
