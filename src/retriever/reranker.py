"""
Cross-encoder reranking.

Retrieval scores a query and a chunk separately and compares the two vectors,
which is fast enough to run over the whole corpus but throws away any
word-level interaction between them. A reranker reads the query and the chunk
together and scores the pair directly. That is far more accurate and far more
expensive, so it only ever sees the handful of candidates retrieval already
shortlisted.

Voyage is used when VOYAGE_API_KEY is set, otherwise Cohere. The Cohere trial
key allows 10 rerank calls/minute, which aborted benchmark runs part-way
through - 35 queries across several configs exhausts that in seconds.

Usage:  python -m src.retriever.reranker "your question"
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request

import cohere
from dotenv import load_dotenv

COHERE_MODEL = "rerank-v3.5"
VOYAGE_MODEL = "rerank-2.5"
VOYAGE_URL = "https://api.voyageai.com/v1/rerank"
MAX_RETRIES = 5

# A 429 needs a pause measured in seconds, not the sub-second start of an
# exponential curve. Waiting the limit out is the whole point now that failure
# raises instead of silently handing back retrieval order.
RATE_LIMIT_WAIT = 8

# Minimum gap between calls, enforced before sending rather than after being
# refused. Reacting to 429s was not enough: successful calls fired back to
# back, spent the whole per-minute allowance in a few seconds, and then every
# retry landed inside the same blocked window and the run aborted. Cohere trial
# keys allow 10 rerank calls/minute, so 6.5s keeps a run under the limit
# indefinitely. Voyage without billing is 3 RPM - raise this to ~21s for that.
MIN_INTERVAL = 6.5

_last_call = 0.0


def _throttle():
    """Sleep just long enough to keep calls MIN_INTERVAL apart."""
    global _last_call
    wait = MIN_INTERVAL - (time.monotonic() - _last_call)
    if wait > 0:
        time.sleep(wait)
    _last_call = time.monotonic()

load_dotenv()

_client = None


class _VoyageReranker:
    """Minimal Voyage rerank client - one endpoint, no SDK needed."""

    def __init__(self, api_key, model=VOYAGE_MODEL):
        self.api_key = api_key
        self.model = model

    def rerank(self, query, documents, top_n, **_):
        payload = json.dumps({
            "query": query,
            "documents": documents,
            "model": self.model,
            "top_k": top_n,
        }).encode("utf-8")
        request = urllib.request.Request(
            VOYAGE_URL,
            data=payload,
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            body = json.loads(response.read())

        # Present Cohere's shape so the caller does not care which ran.
        return type("Result", (), {"results": [
            type("Hit", (), {"index": d["index"],
                             "relevance_score": d["relevance_score"]})()
            for d in body["data"]
        ]})()


def get_client():
    """Voyage if its key is present, else Cohere."""
    global _client
    if _client is None:
        voyage_key = os.environ.get("VOYAGE_API_KEY", "")
        if voyage_key.startswith("pa-"):
            _client = _VoyageReranker(voyage_key)
        else:
            key = os.environ.get("COHERE_API_KEY")
            if not key:
                raise SystemExit("no reranker key - set VOYAGE_API_KEY or COHERE_API_KEY")
            _client = cohere.ClientV2(key)
    return _client


def active_model():
    client = get_client()
    return getattr(client, "model", COHERE_MODEL)


def rerank(query, hits, top_n=5, client=None):
    """
    Reorder `hits` by relevance to `query`.

    Each returned hit gets `rerank_score` (0-1) and keeps `fused_score` /
    `dense_rank` / `bm25_rank` so the retrieval path stays inspectable. `score`
    is overwritten with the rerank score, since that is now the ranking signal.

    Raises on failure rather than returning the retrieval order. The two stages
    put incompatible values in `score`: rerank is a 0-1 relevance, while RRF
    sums weight/(60+rank) and tops out near 0.021. Silently handing back RRF
    scores made every hit fall below the generation threshold, so the pipeline
    answered "insufficient information" for questions it had retrieved
    correctly. A loud failure is better than an answer that looks like a
    knowledge gap.
    """
    if not hits:
        return []

    client = client or get_client()
    documents = [hit["text"] for hit in hits]

    for attempt in range(MAX_RETRIES):
        try:
            _throttle()
            response = client.rerank(
                model=getattr(client, "model", COHERE_MODEL),
                query=query,
                documents=documents,
                top_n=min(top_n, len(documents)),
            )
            break
        except Exception as exc:
            if attempt == MAX_RETRIES - 1:
                raise RuntimeError(
                    f"rerank failed after {MAX_RETRIES} attempts: {type(exc).__name__}: {exc}"
                ) from exc
            is_429 = "429" in str(exc) or "TooManyRequests" in type(exc).__name__
            wait = RATE_LIMIT_WAIT if is_429 else 2**attempt
            print(f"  [rerank-retry {attempt + 1}/{MAX_RETRIES}] sleeping {wait}s "
                  f"({type(exc).__name__})")
            time.sleep(wait)

    ranked = []
    for position, result in enumerate(response.results, 1):
        hit = dict(hits[result.index])
        hit["rerank_score"] = float(result.relevance_score)
        hit["rerank_position"] = position
        hit["retrieval_rank"] = result.index + 1
        hit["score"] = hit["rerank_score"]
        ranked.append(hit)
    return ranked


def main():
    from src.retriever.hybrid import hybrid_search

    query = sys.argv[1] if len(sys.argv) > 1 else "What causes cirrhosis?"
    print(f"query: {query}\n")
    for hit in hybrid_search(query, k=5, rerank=True):
        moved = hit["retrieval_rank"] - hit["rerank_position"]
        arrow = f"{moved:+d}" if moved else " ="
        print(f"  {hit['score']:.3f} [{arrow}] {hit['heading'][:52]}")


if __name__ == "__main__":
    main()
