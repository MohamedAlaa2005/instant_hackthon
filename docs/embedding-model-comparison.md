# Embedding Model Comparison — Cohere vs MedEmbed

Does a medical-domain embedding model beat a general-purpose one on this corpus?

| | Cohere | MedEmbed |
|---|---|---|
| model | `embed-v4.0` | `abhinand/MedEmbed-large-v0.1` |
| dimensions | 1024 | 1024 |
| where it runs | hosted API | local (~1.3 GB download) |
| trained for | general text | medical retrieval |
| rate limits | trial key: 1000 calls/month | none |

Reranking is Cohere `rerank-v3.5` in both runs, so the only variable is the
embedder. Same 35-query set, same 348 chunks, same contextual retrieval, same
fusion weights, `k=5`.

```bash
python -m src.evaluation.evaluate --queries data/eval/queries.jsonl --k 5 --config full                    # Cohere
EMBEDDING_PROVIDER=medembed python -m src.evaluation.evaluate --queries data/eval/queries.jsonl --k 5 --config full
```

### `dense` — embeddings only, no BM25, no rerank

This is the cleaner test: it isolates the embedder with nothing downstream to
mask a difference.

| embedder | P@5 | R@5 | MAP@5 | MRR | sec/q |
|---|---|---|---|---|---|
| **Cohere `embed-v4.0`** | **0.354** | **0.936** | **0.877** | **0.967** | 1.41 |
| MedEmbed-large | 0.343 | 0.921 | 0.854 | 0.944 | 0.96 |

