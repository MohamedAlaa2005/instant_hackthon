"""
Hybrid retrieval: BM25 (lexical) fused with Cohere embeddings (semantic).

Neither leg is sufficient alone on this corpus. Embeddings handle paraphrase -
"fatty liver diet" finds "How can my diet help prevent or treat NAFLD?" - but
blur rare tokens, so an exact term like HBsAg or anti-HCV can rank below a
merely topical passage. BM25 is the opposite: it nails the exact token and is
useless when the user's wording differs from the document's.

The two are combined with Reciprocal Rank Fusion. RRF reads only the rank
positions, never the raw scores, which matters because BM25 scores are
unbounded while cosine similarity sits in [0, 1] - there is no sane way to
average them directly, and any normalisation would need retuning whenever the
corpus changes.

Usage:  python -m src.retriever.hybrid "your question"
"""

import json
import re
import sys


from rank_bm25 import BM25Okapi

from src.config import CHUNKS_PATH
from src.retriever.query_parser import ParsedQuery, parse_query
from src.retriever.retriever import dense_search

CANDIDATES = 50  # per leg, before fusion
RRF_K = 60  # damping constant; 60 is the value from the original RRF paper

# Leg weights. Equal weighting measurably hurt paraphrase queries on this
# corpus - "what should I eat if my liver is fatty" lost its NAFLD answer to
# BM25 hits that merely shared the words "eat" and "liver". The dense leg is
# the stronger of the two here, so the lexical leg is kept as a corrective
# rather than an equal partner.
DENSE_WEIGHT = 1.0
SPARSE_WEIGHT = 0.25

# Also search the question as the user typed it, alongside the rewrite, and
# fuse both. Weighted equal to the rewrite: neither wording is reliably better,
# and the point is that a poor rewrite can no longer throw away a good match.
KEEP_RAW_QUERY = True
RAW_WEIGHT = 1.0

# How many fused candidates go to the reranker. More is better for recall but
# costs latency and API quota; the reranker cuts this down to k.
RERANK_CANDIDATES = 50

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_bm25 = None
_chunks = None


def tokenize(text):
    """
    Lowercase alphanumeric tokens.

    Splitting on non-alphanumerics keeps clinical identifiers intact as single
    tokens - "HBsAg" -> "hbsag", "anti-HCV" -> "anti", "hcv" - which is what
    makes the lexical leg worth having.
    """
    return _TOKEN_RE.findall(text.lower())


def _load():
    """Build the BM25 index once per process. 344 docs, so this is instant."""
    global _bm25, _chunks
    if _bm25 is None:
        with open(CHUNKS_PATH, encoding="utf-8") as fh:
            _chunks = [json.loads(line) for line in fh if line.strip()]
        _bm25 = BM25Okapi([tokenize(c["text"]) for c in _chunks])
    return _bm25, _chunks


def _matches(chunk, corpus, topic, section):
    if corpus and chunk.get("corpus") != corpus:
        return False
    if topic and chunk.get("topic") != topic:
        return False
    if section and chunk.get("section") != section:
        return False
    return True


def sparse_search(query, k=CANDIDATES, corpus=None, topic=None, section=None):
    """BM25 search. Filters are applied to the pool before taking the top k."""
    bm25, chunks = _load()
    scores = bm25.get_scores(tokenize(query))

    pool = [
        (score, chunk)
        for score, chunk in zip(scores, chunks)
        if score > 0 and _matches(chunk, corpus, topic, section)
    ]
    pool.sort(key=lambda pair: -pair[0])

    hits = []
    for rank, (score, chunk) in enumerate(pool[:k], 1):
        hit = dict(chunk)
        hit["score"] = float(score)
        hit["bm25_score"] = float(score)
        hit["bm25_rank"] = rank
        hits.append(hit)
    return hits


def hybrid_search(
    query,
    k=5,
    corpus=None,
    topic=None,
    section=None,
    candidates=CANDIDATES,
    dense_weight=DENSE_WEIGHT,
    sparse_weight=SPARSE_WEIGHT,
    rerank=False,
    parse=True,
    keep_raw=KEEP_RAW_QUERY,
    raw_weight=RAW_WEIGHT,
):
    """
    Run both legs and fuse by weighted reciprocal rank.

    Returns up to `k` hits sorted by fused score, each annotated with whichever
    of `dense_rank` / `bm25_rank` it earned, so you can see which leg found it.

    `query` may be a string or an already-built ParsedQuery. With `parse=True`
    a string is run through the query parser first, so each leg receives the
    form it wants; `parse=False` sends the raw string to both.

    With `rerank=True` the fused list is widened to RERANK_CANDIDATES and
    handed to the cross-encoder, which picks the final k. Fusion then only has
    to get the right chunk somewhere into the shortlist rather than at the top,
    so the exact leg weights matter much less.
    """
    if isinstance(query, ParsedQuery):
        parsed = query
    elif parse:
        parsed = parse_query(query)
    else:
        parsed = ParsedQuery(raw=query, dense_query=query, sparse_query=query)

    dense = dense_search(
        parsed.dense_query, k=candidates, corpus=corpus, topic=topic, section=section
    )
    sparse = sparse_search(
        parsed.sparse_query, k=candidates, corpus=corpus, topic=topic, section=section
    )
    legs = [(dense, dense_weight), (sparse, sparse_weight)]

    # Search the user's own wording too, not just the rewrite.
    #
    # The rewrite replaces the question, and on this corpus that loses ground:
    # the headings are already plain patient questions ("How can my diet help
    # prevent or treat NAFLD?"), so "what should i eat if my liver is fatty"
    # starts closer to the answer than "dietary guidelines for NAFLD" does.
    # Measured at k=5, rewriting cost 0.055 MAP against not rewriting, and raw
    # dense search beat every config that touched the query. Keeping the
    # original as a third leg means a bad rewrite can no longer discard the
    # good match - it costs one extra embedding, and BM25 is local.
    if keep_raw and parsed.used_llm and parsed.raw.strip() != parsed.dense_query.strip():
        legs.append((
            dense_search(parsed.raw, k=candidates, corpus=corpus, topic=topic, section=section),
            raw_weight,
        ))
        legs.append((
            sparse_search(parsed.raw, k=candidates, corpus=corpus, topic=topic, section=section),
            raw_weight * sparse_weight,
        ))

    fused = {}
    for hits, weight in legs:
        for rank, hit in enumerate(hits, 1):
            entry = fused.setdefault(hit["id"], dict(hit))
            entry.update({key: value for key, value in hit.items() if key != "score"})
            entry["fused_score"] = entry.get("fused_score", 0.0) + weight / (RRF_K + rank)

    ranked = sorted(fused.values(), key=lambda hit: -hit["fused_score"])
    for hit in ranked:
        hit["score"] = hit["fused_score"]
        hit.setdefault("dense_rank", None)
        hit.setdefault("bm25_rank", None)

    if rerank:
        from src.retriever.reranker import rerank as _rerank

        # Reranked against the user's actual question, not the rewrite. The
        # cross-encoder is the last stage that can correct a bad rewrite, so
        # feeding it the rewrite would compound the error instead.
        return _rerank(parsed.raw, ranked[:RERANK_CANDIDATES], top_n=k)
    return ranked[:k]


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "Is HBsAg testing needed in every pregnancy?"
    print(f"query: {query}\n")
    for hit in hybrid_search(query, k=5):
        found = []
        if hit["dense_rank"]:
            found.append(f"dense#{hit['dense_rank']}")
        if hit["bm25_rank"]:
            found.append(f"bm25#{hit['bm25_rank']}")
        print(f"  {hit['score']:.4f} [{'+'.join(found):<18}] {hit['heading'][:50]}")


if __name__ == "__main__":
    main()
