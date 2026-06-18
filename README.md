# doc-rag-assistant

[![CI](https://github.com/vk86294140-cloud/doc-rag-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/vk86294140-cloud/doc-rag-assistant/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Claude](https://img.shields.io/badge/LLM-Claude%20Opus%204.8-8A2BE2)
![License](https://img.shields.io/badge/license-MIT-green)

A **retrieval-augmented generation (RAG)** service that answers questions over
**your own documents** — grounded in the source text and returned **with
citations** — using the **Anthropic Claude API**. Ships with a FastAPI backend
and a tiny zero-build web UI.

```
documents ─► chunk ─► embed ─► vector store ─► retrieve top-k ─► Claude ─► answer + citations
```

## Why it's built this way

- **Grounded, cited answers.** The system prompt forces Claude to answer only
  from retrieved context and cite the chunks it used — and to say *"I don't
  know"* when the context doesn't cover the question. That's the difference
  between a useful RAG tool and a confident hallucinator.
- **Runs offline and in CI.** Embeddings use scikit-learn's stateless
  `HashingVectorizer` (no model download), and the vector store is a ~100-line
  numpy cosine-similarity index. An `EchoLLM` exercises the full pipeline with
  **no API key**, so the test suite is fast and hermetic.
- **Pluggable by design.** `Embedder`, `VectorStore`, and `LLM` are small
  interfaces. Swapping in a semantic embedding model, Chroma/Pinecone, or a
  different LLM is a localized change — see [Swapping components](#swapping-components).
- **Claude-native.** Uses the official `anthropic` SDK with **adaptive
  thinking** and `claude-opus-4-8`.

## Quickstart

```bash
pip install -e ".[dev]"

# 1. Ingest some documents into a persistent index
python -m ragassistant.ingest sample_docs/

# 2. Set your key and serve (needed only for live Claude answers)
export ANTHROPIC_API_KEY=sk-ant-...      # PowerShell: $env:ANTHROPIC_API_KEY="sk-ant-..."
uvicorn ragassistant.api:app --reload --port 8000
```

Open `http://localhost:8000` for the chat UI, or call the API:

```bash
curl -s http://localhost:8000/query -H "Content-Type: application/json" -d '{
  "question": "How long do I have to return an item?", "top_k": 4
}'
```

```json
{
  "answer": "You may return items within 30 days of delivery for a full refund [1].",
  "citations": [
    {"source": "returns_policy.md", "chunk_index": 0, "score": 0.41,
     "preview": "Items may be returned within 30 days of delivery..."}
  ]
}
```

> **No API key?** Everything except the final Claude call works without one. The
> test suite and the retrieval pipeline run fully offline via the built-in
> `EchoLLM`.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | liveness + number of indexed chunks |
| `POST` | `/ingest` | add a document (`{source, text}`) to the index |
| `POST` | `/query` | ask a question, get a grounded answer + citations |
| `DELETE` | `/documents/{source}` | remove a document and all its chunks |
| `GET` | `/` | minimal chat UI |
| `GET` | `/docs` | OpenAPI docs |

## How it works

1. **Chunking** ([`chunking.py`](src/ragassistant/chunking.py)) splits documents
   on paragraph/sentence boundaries into overlapping ~800-char chunks so each
   chunk is a coherent, retrievable unit.
2. **Embedding** ([`embeddings.py`](src/ragassistant/embeddings.py)) maps chunks
   and queries into the same L2-normalized vector space.
3. **Retrieval** ([`store.py`](src/ragassistant/store.py)) ranks chunks by cosine
   similarity and returns the top-k.
4. **Generation** ([`rag.py`](src/ragassistant/rag.py)) builds a numbered-context
   prompt and asks Claude to answer **only** from that context, with citations.

## Swapping components

The engine depends on three small interfaces, so production swaps are local:

- **Semantic embeddings** — implement `Embedder` (`embed(texts) -> (n, dim)`
  normalized array) around sentence-transformers or a hosted embeddings API.
- **Managed vector DB** — implement the `add` / `query` surface of `VectorStore`
  against Chroma or Pinecone; nothing else changes.
- **Different model** — `AnthropicLLM` already takes a `model=` argument; or
  implement the `LLM` protocol for another provider.

## Run with Docker

```bash
export ANTHROPIC_API_KEY=sk-ant-...
docker compose up --build
# ingest + ask via the UI at http://localhost:8000
```

## Testing

```bash
pytest -v
```

Covers chunking (size budget, overlap, ordering), the vector store (ranking +
persistence round-trip), end-to-end retrieval (correct source routing,
citations, honest "I don't know"), and the API surface — all offline.

## Project layout

```
src/ragassistant/
  chunking.py    paragraph-aware overlapping chunker
  embeddings.py  HashingVectorizer embedder (Embedder protocol)
  store.py       numpy cosine-similarity vector store (+ persistence)
  llm.py         Anthropic Claude client + offline EchoLLM
  rag.py         ingest -> retrieve -> answer-with-citations engine
  schema.py      pydantic request/response models
  api.py         FastAPI app
  ingest.py      CLI: python -m ragassistant.ingest <path>
frontend/        zero-build chat UI
sample_docs/     example documents to index
tests/           offline pytest suite
```

## License

MIT — see [LICENSE](LICENSE).
