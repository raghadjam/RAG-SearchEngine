
# RAG Search Engine

A CLI movie search engine that layers keyword search, semantic search, hybrid retrieval, LLM query enhancement/re-ranking, RAG, and image-based search on top of a movie dataset.

## Setup

```bash
uv sync
```

Add a `.env` with an [OpenRouter](https://openrouter.ai) key for LLM-powered commands:

```
OPENROUTER_API_KEY=your-key-here
```
## Features

- **Keyword search** — BM25 over titles/descriptions (`keyword_search_cli.py`)
- **Semantic search** — sentence embeddings + chunking (`semantic_search_cli.py`)
- **Hybrid search** — weighted score fusion and RRF, plus:
  - Query enhancement: spelling fix, rewrite, expand
  - Re-ranking: individual LLM scoring, batch LLM ranking, local cross-encoder
  - LLM-graded evaluation of results
  (`hybrid_search_cli.py`)
- **RAG** — direct answers, summaries, cited answers, casual Q&A, all grounded in search results (`augmented_generation_cli.py`)
- **Offline eval** — precision@k / recall@k / F1 against a golden dataset (`evaluation_cli.py`)
- **Multimodal search** — CLIP image embeddings for image-to-movie search, plus image-aware query rewriting (`multimodal_search_cli.py`, `describe_image_cli.py`)

## Example usage

```bash
uv run cli/hybrid_search_cli.py rrf-search "space adventure" --limit 5
uv run cli/hybrid_search_cli.py rrf-search "math movie" --enhance expand
uv run cli/hybrid_search_cli.py rrf-search "family bear movie" --rerank-method batch
uv run cli/augmented_generation_cli.py question "good dinosaur movie to watch tonight"
uv run cli/evaluation_cli.py --limit 10
uv run cli/multimodal_search_cli.py image_search data/paddington.jpeg
```

## Notes

- `cache/` holds generated embeddings/index files and is gitignored — delete it to force a rebuild after logic changes.
- LLM commands hit OpenRouter's free tier, which can be inconsistent; retry on failure.
```

```bash
git add README.md
git commit -m "Add project README"
git push
```
