"""
src/evaluation/runner.py
========================
Orchestrates a full retrieval evaluation run over a benchmark query set.

Data flow:
    queries.jsonl  ──► load_qrels()
                            │
                            ▼  list[{"query": str, "relevant_ids": set[str]}]
                       run_evaluation()
                            │
                            ├── retrieve(query, top_k)          (src.retriever)
                            │       └─► list[dict]  with "id", "score", …
                            │
                            ├── [c["id"] for c in results]      retrieved_ids
                            │
                            └── evaluate_single(retrieved_ids, relevant_ids, k)
                                    └─► per-query metric dict
                                            │
                                       aggregate()
                                            │
                                       print_report()
"""

import json
from pathlib import Path

from src.retriever import retrieve
from src.evaluation.metrics import evaluate_single, mean_average_precision, mean_reciprocal_rank


# ==============================================================================
# § 1  load_qrels — deserialise the benchmark file
# ==============================================================================

def load_qrels(path: str | Path) -> list[dict]:
    """
    Load a JSONL ground-truth file where each line is:

        {"query": "...", "relevant_ids": ["corpus-NNNN", ...]}

    Returns a list of dicts with:
        - "query"        : str        — the natural-language question
        - "relevant_ids" : set[str]   — IDs of chunks known to be relevant

    The relevant_ids set is converted from a JSON list here so that downstream
    callers always work with O(1) membership lookups.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Qrels file not found: {path}\n"
            "Create it manually or run:  python -m src.evaluation.generate_qrels"
        )

    qrels = []
    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Bad JSON on line {line_no} of {path}: {exc}") from exc

            # Normalise: relevant_ids must be a set for O(1) lookups in metrics
            entry["relevant_ids"] = set(entry.get("relevant_ids", []))
            qrels.append(entry)

    return qrels


# ==============================================================================
# § 2  run_evaluation — loop over every query, retrieve, then score
# ==============================================================================

def run_evaluation(
    qrels: list[dict],
    top_k: int = 5,
    verbose: bool = True,
) -> dict:
    """
    Run the full evaluation loop.

    For each entry in qrels:
        1. Call retrieve(query, top_k) from src.retriever.
        2. Extract the ordered list of chunk IDs from the result.
        3. Score against the ground-truth relevant_ids using evaluate_single().

    Args:
        qrels:   Output of load_qrels() — list of {"query", "relevant_ids"}.
        top_k:   Rank cut-off K (same value passed to retrieve).
        verbose: If True, print a per-query row while running.

    Returns:
        A summary dict (see aggregate()) containing per-query results and
        macro-average scores.
    """
    per_query: list[dict] = []

    for entry in qrels:
        query        = entry["query"]
        relevant_ids = entry["relevant_ids"]

        # --- Retrieval ---
        # retrieve() returns list[dict] with fields: id, score, text,
        # section_path, url, source, corpus, topic, heading.
        # We only need the ordered IDs for metric computation.
        results      = retrieve(query, top_k=top_k)
        retrieved_ids: list[str] = [c["id"] for c in results]

        # --- Scoring ---
        scores = evaluate_single(retrieved_ids, relevant_ids, k=top_k)
        scores["query"] = query

        per_query.append(scores)

        if verbose:
            print(
                f"  P@{top_k}={scores['precision_at_k']:.3f}  "
                f"R@{top_k}={scores['recall_at_k']:.3f}  "
                f"AP@{top_k}={scores['ap_at_k']:.3f}  "
                f"RR={scores['rr']:.3f}  |  {query[:60]}"
            )

    return aggregate(per_query, top_k=top_k)


# ==============================================================================
# § 3  aggregate — compute macro-average scores across all queries
# ==============================================================================

def aggregate(per_query: list[dict], top_k: int) -> dict:
    """
    Compute macro-average (mean) of each metric over all queries.

    Uses mean_average_precision and mean_reciprocal_rank from metrics.py to
    ensure the averaging logic stays in one place.

    Returns:
        {
            "k":                int,
            "num_queries":      int,
            "map_at_k":         float,   # Mean Average Precision
            "mrr":              float,   # Mean Reciprocal Rank
            "mean_precision":   float,   # Mean Precision@K
            "mean_recall":      float,   # Mean Recall@K
            "per_query":        list[dict],
        }
    """
    if not per_query:
        return {}

    n = len(per_query)
    mean_p = sum(r["precision_at_k"] for r in per_query) / n
    mean_r = sum(r["recall_at_k"]    for r in per_query) / n
    map_k  = sum(r["ap_at_k"]        for r in per_query) / n
    mrr    = sum(r["rr"]             for r in per_query) / n

    return {
        "k":              top_k,
        "num_queries":    n,
        "mean_precision": mean_p,
        "mean_recall":    mean_r,
        "map_at_k":       map_k,
        "mrr":            mrr,
        "per_query":      per_query,
    }


# ==============================================================================
# § 4  print_report — human-readable console output
# ==============================================================================

def print_report(summary: dict) -> None:
    """
    Print a formatted evaluation report to stdout.

    Layout:
        - Header with K and number of queries evaluated
        - Per-query table (query text truncated to 50 chars, then four scores)
        - Separator line
        - Aggregate / macro-average row
    """
    k   = summary["k"]
    n   = summary["num_queries"]
    col = 52   # Width of the query column

    header = (
        f"\n{'=' * 90}\n"
        f"  RETRIEVAL EVALUATION REPORT   (k={k}, queries={n})\n"
        f"{'=' * 90}\n"
        f"  {'Query':<{col}}  P@{k:<3}  R@{k:<3}  AP@{k:<3}  RR\n"
        f"  {'-' * col}  -----  -----  -----  -----"
    )
    print(header)

    for row in summary["per_query"]:
        q_label = (row["query"][:col - 1] + "…") if len(row["query"]) > col else row["query"]
        print(
            f"  {q_label:<{col}}"
            f"  {row['precision_at_k']:.3f}"
            f"  {row['recall_at_k']:.3f}"
            f"  {row['ap_at_k']:.3f}"
            f"  {row['rr']:.3f}"
        )

    print(f"\n  {'MACRO AVERAGE':<{col}}  {summary['mean_precision']:.3f}  {summary['mean_recall']:.3f}  {summary['map_at_k']:.3f}  {summary['mrr']:.3f}")
    print(f"{'=' * 90}\n")
