"""
Retrieval evaluation harness.

Runs a labelled query set through several retrieval configurations and reports
Precision@K, Recall@K, MAP@K and MRR side by side, so a change can be judged on
numbers instead of on how a handful of spot checks happen to look.

Test set format - one JSON object per line in data/eval/queries.jsonl:

    {"query": "what should i eat if my liver is fatty",
     "relevant_ids": ["niddk_liver-0123", "niddk_liver-0124"]}

`relevant_ids` are chunk ids from data/processed/chunks.jsonl. Ids are
validated on load: a typo silently scores zero for that query, which looks
exactly like a retrieval failure, so it is caught up front instead.

IMPORTANT: chunk ids are positional (`niddk_liver-0123`) and are reassigned
every time the parser reruns. Rebuilding the corpus invalidates the labels.

Usage:
    python -m src.evaluation.evaluate                 # compare all configs
    python -m src.evaluation.evaluate --k 10
    python -m src.evaluation.evaluate --config hybrid_rerank
    python -m src.evaluation.evaluate --queries data/eval/my_set.jsonl
"""

import argparse
import json
import os
import sys
import time

from src.config import CHUNKS_PATH
from src.evaluation.metrics import aggregate, evaluate_single

QUERIES_PATH = os.path.join("data", "eval", "queries.jsonl")
DEFAULT_K = 5

# Each config is a retrieval function taking (query, k) and returning hits.
# Kept lazy so importing this module does not spin up Chroma or Cohere.
CONFIGS = {
    "dense": lambda q, k: _dense(q, k),
    "hybrid": lambda q, k: _hybrid(q, k, rerank=False, parse=False),
    "hybrid_parse": lambda q, k: _hybrid(q, k, rerank=False, parse=True),
    "hybrid_rerank": lambda q, k: _hybrid(q, k, rerank=True, parse=False),
    "full": lambda q, k: _hybrid(q, k, rerank=True, parse=True),
    # Same two, but searching only the rewrite - the baseline for whether
    # also searching the user's own wording is worth an extra embedding.
    "parse_norawq": lambda q, k: _hybrid(q, k, rerank=False, parse=True, keep_raw=False),
    "full_norawq": lambda q, k: _hybrid(q, k, rerank=True, parse=True, keep_raw=False),
}


def _dense(query, k):
    from src.retriever.retriever import dense_search

    return dense_search(query, k=k)


def _hybrid(query, k, rerank, parse, **kw):
    from src.retriever.hybrid import hybrid_search

    return hybrid_search(query, k=k, rerank=rerank, parse=parse, **kw)


def load_queries(path=QUERIES_PATH, validate=True):
    """Read the labelled set and check every id exists in the corpus."""
    if not os.path.exists(path):
        raise SystemExit(
            f"No query set at {path}\n\n"
            "Create it with one JSON object per line:\n"
            '  {"query": "...", "relevant_ids": ["niddk_liver-0123"]}\n'
        )

    queries = []
    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("query") or not row.get("relevant_ids"):
                raise SystemExit(f"{path}:{line_no} needs both 'query' and 'relevant_ids'")
            queries.append({"query": row["query"], "relevant_ids": set(row["relevant_ids"])})

    if validate:
        with open(CHUNKS_PATH, encoding="utf-8") as fh:
            known = {json.loads(l)["id"] for l in fh if l.strip()}
        unknown = {i for q in queries for i in q["relevant_ids"]} - known
        if unknown:
            raise SystemExit(
                f"{len(unknown)} label id(s) are not in the corpus, e.g. "
                f"{sorted(unknown)[:3]}\n"
                "Ids shift whenever the parser reruns - relabel against the "
                "current chunks.jsonl."
            )
    return queries


def run_config(name, queries, k):
    """Retrieve for every query and return (aggregate_metrics, per_query_rows)."""
    retriever = CONFIGS[name]
    pairs, rows = [], []

    started = time.time()
    for item in queries:
        hits = retriever(item["query"], k)
        retrieved_ids = [h["id"] for h in hits]
        pairs.append((retrieved_ids, item["relevant_ids"]))
        rows.append({
            "query": item["query"],
            "retrieved": retrieved_ids,
            **evaluate_single(retrieved_ids, item["relevant_ids"], k),
        })
    elapsed = time.time() - started

    summary = aggregate(pairs, k)
    summary["config"] = name
    summary["seconds_per_query"] = elapsed / max(len(queries), 1)
    return summary, rows


def print_table(summaries, k):
    header = f"{'config':<16}{'P@'+str(k):>8}{'R@'+str(k):>9}{'MAP@'+str(k):>10}{'MRR':>8}{'sec/q':>9}"
    print("\n" + header)
    print("-" * len(header))
    for s in summaries:
        print(
            f"{s['config']:<16}{s['precision_at_k']:>8.3f}{s['recall_at_k']:>9.3f}"
            f"{s['map_at_k']:>10.3f}{s['mrr']:>8.3f}{s['seconds_per_query']:>9.2f}"
        )


def print_failures(rows, limit=5):
    """Queries where nothing relevant was retrieved - where to look first."""
    misses = [r for r in rows if r["rr"] == 0.0]
    if not misses:
        print("\nno complete misses")
        return
    print(f"\ncomplete misses ({len(misses)}/{len(rows)}):")
    for row in misses[:limit]:
        print(f"  - {row['query'][:66]}")


def main():
    ap = argparse.ArgumentParser(description="Evaluate retrieval configurations.")
    ap.add_argument("--queries", default=QUERIES_PATH)
    ap.add_argument("--k", type=int, nargs="+", default=[DEFAULT_K],
                help="one or more rank cut-offs, e.g. --k 1 3 5 10")
    ap.add_argument("--allow-degraded", action="store_true",
                    help="permit dictionary-only fallback when the LLM is rate limited")
    ap.add_argument("--config", choices=list(CONFIGS) + ["all"], default="all")
    ap.add_argument("--per-query", action="store_true", help="print every query's scores")
    ap.add_argument("--save", help="write results to this JSON file")
    args = ap.parse_args()

    # A 429 used to drop the parse configs to dictionary-only mid-run and
    # still print a number, which understates exactly the configs being
    # tested. Abort instead, unless the caller opts out.
    if not args.allow_degraded:
        os.environ["STRICT_QUERY_REWRITE"] = "1"

    queries = load_queries(args.queries)
    names = list(CONFIGS) if args.config == "all" else [args.config]
    print(f"{len(queries)} queries, k={args.k}, configs: {', '.join(names)}")

    all_summaries, detail = {}, {}
    for k in args.k:
        summaries = []
        for name in names:
            print(f"  running {name} @k={k} ...", end="", flush=True)
            summary, rows = run_config(name, queries, k)
            summaries.append(summary)
            detail[(name, k)] = rows
            print(" done")
        all_summaries[k] = summaries
        print_table(summaries, k)

    largest = max(args.k)
    best = max(all_summaries[largest], key=lambda s: s["map_at_k"])
    print(f"\nbest by MAP@{largest}: {best['config']} ({best['map_at_k']:.3f})")
    print_failures(detail[(best["config"], largest)])
    summaries = [s for k in args.k for s in all_summaries[k]]

    if args.per_query:
        for name in names:
            print(f"\n--- {name} ---")
            for row in detail[name]:
                print(f"  P={row['precision_at_k']:.2f} R={row['recall_at_k']:.2f} "
                      f"RR={row['rr']:.2f}  {row['query'][:52]}")

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        with open(args.save, "w", encoding="utf-8") as fh:
            json.dump({"summaries": summaries,
                       "detail": {f"{n}@{k}": v for (n, k), v in detail.items()}},
                      fh, indent=2)
        print(f"\nsaved -> {args.save}")

    if len(queries) < 20:
        print(
            f"\nwarning: {len(queries)} queries is too few to separate these "
            "configs - differences this size are noise. Aim for 30+."
        )


if __name__ == "__main__":
    main()
