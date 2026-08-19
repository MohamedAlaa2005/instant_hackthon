# Retrieval Evaluation

> **Superseded — kept for the method and the caveats, not the numbers.**
> These runs predate the merge that replaced LLM query rewriting with the
> clinical dictionary, and the reranker rows here are ~15% degraded (see the
> caveat section). `docs/metrices_retrieval1.md` carries the current figures.

**Corpus** 348 chunks (276 NIDDK patient pages, 72 USPSTF guideline sections)
**Query set** `data/eval/queries.jsonl` — 35 queries, 70 labels, 19 topics
**Embeddings** Cohere `embed-v4.0` (1024d) · **Rerank** Cohere `rerank-v3.5` · **LLM** Gemini 3.1 Flash Lite via Lightning

```bash
python -m src.evaluation.evaluate --queries data/eval/queries.jsonl --k 1 3 5 10
```

## Configurations

| name | query rewrite | dense | BM25 | rerank |
|---|:--:|:--:|:--:|:--:|
| `dense` | – | ✓ | – | – |
| `hybrid` | – | ✓ | ✓ | – |
| `hybrid_parse` | ✓ | ✓ | ✓ | – |
| `hybrid_rerank` | – | ✓ | ✓ | ✓ |
| `full` | ✓ | ✓ | ✓ | ✓ |

## Results

### k = 1

| config | P@1 | R@1 | MAP@1 | MRR | sec/q |
|---|---|---|---|---|---|
| **dense** | **0.943** | 0.595 | **0.943** | **0.943** | 0.83 |
| hybrid | 0.914 | 0.567 | 0.914 | 0.914 | 0.01 |
| hybrid_parse | 0.800 | 0.500 | 0.800 | 0.800 | 3.07 |
| hybrid_rerank | 0.914 | **0.600** | 0.914 | 0.914 | 4.26 |
| full | 0.914 | **0.600** | 0.914 | 0.914 | 4.55 |

### k = 3

| config | P@3 | R@3 | MAP@3 | MRR | sec/q |
|---|---|---|---|---|---|
| **dense** | **0.524** | **0.852** | **0.849** | **0.957** | 0.01 |
| hybrid | 0.486 | 0.814 | 0.789 | 0.943 | 0.02 |
| hybrid_parse | 0.438 | 0.731 | 0.697 | 0.848 | 0.02 |
| hybrid_rerank | 0.467 | 0.802 | 0.792 | 0.938 | 5.14 |
| full | 0.467 | 0.781 | 0.792 | 0.929 | 5.32 |

### k = 5

| config | P@5 | R@5 | MAP@5 | MRR | sec/q |
|---|---|---|---|---|---|
| **dense** | **0.331** | **0.876** | **0.845** | **0.957** | 0.00 |
| hybrid | 0.303 | 0.831 | 0.784 | 0.943 | 0.00 |
| hybrid_parse | 0.303 | 0.826 | 0.723 | 0.862 | 0.00 |
| hybrid_rerank | 0.309 | 0.855 | 0.799 | 0.944 | 5.17 |
| full | 0.314 | 0.836 | 0.758 | 0.887 | 5.20 |

### k = 10

| config | P@10 | R@10 | MAP@10 | MRR | sec/q |
|---|---|---|---|---|---|
| **dense** | 0.183 | **0.955** | **0.869** | **0.962** | 0.00 |
| hybrid | 0.174 | 0.931 | 0.815 | 0.948 | 0.00 |
| hybrid_parse | 0.171 | 0.893 | 0.750 | 0.862 | 0.00 |
| hybrid_rerank | 0.183 | 0.948 | 0.829 | 0.930 | 5.28 |
| full | 0.180 | 0.921 | 0.835 | 0.934 | 5.21 |

## Headline

**Dense retrieval alone is the strongest configuration at every cut-off.**
`P@1 = 0.943` — the top result is correct for 33 of 35 queries.

## Reading the numbers

**Precision falls as k rises, and that is expected.** Most queries have 1–3
relevant chunks, so P@5 cannot exceed ~0.28 and P@10 cannot exceed ~0.14 no
matter how good retrieval is. Report `P@1` as the precision headline and use
`R@10` (0.955) for coverage; `P@5` looks like failure when it is a ceiling.

**Query rewriting is actively hurting.** `hybrid_parse` is the worst config at
every k — MAP@3 of 0.697 against dense at 0.849. Rewriting a question into
clinical prose moves it *away* from a corpus whose headings are already plain
patient questions ("What causes cirrhosis?").

**Reranking does not pay for itself here.** It costs ~5 s/query and never wins
on MAP. It does give the best R@1 (0.600), so it is placing a relevant chunk
first slightly more often — but not enough to justify the latency.

**BM25 does not help either.** `hybrid` trails `dense` at every k. Cohere
`embed-v4.0` already handles the rare clinical tokens (`HBsAg`, `anti-HCV`)
that the lexical leg was added to catch.

## Caveat on the rerank rows

41 of 280 rerank calls (**~15%**) failed with `TooManyRequestsError` and fell
back to fusion order. `hybrid_rerank` and `full` are therefore ~85% measured,
15% degraded — their true scores are somewhat higher than shown.

Two other confounds worth knowing:

- The reranker sees a **fixed 25 candidates at every k**, so `k=1` gets 25×
  over-fetch while `k=10` gets 2.5×. The `k=10` rerank rows partly reflect a
  thinner shortlist rather than the cut-off alone.
- 35 queries is enough to rank configs but not to separate differences of a
  few points.

## Suggested next steps

1. Make the reranker fail loudly under `STRICT_QUERY_REWRITE`, as the query
   parser already does, so a degraded run cannot be mistaken for a result.
2. Re-run the rerank configs on a fresh quota to get clean numbers.
3. Scale `RERANK_CANDIDATES` with k (`max(25, k * 5)`) so over-fetch depth is
   constant across cut-offs.
4. Consider shipping `dense` and keeping hybrid/rerank behind a flag — it is
   currently 5 s/query slower for no measured gain.
