"""
src/evaluation/metrics.py
=========================
Pure, stateless IR metric functions.

All functions share the same input contract:
    retrieved_ids : list[str]  — ordered list of chunk["id"] values as returned
                                 by retrieve(), highest-score first.
    relevant_ids  : set[str]   — ground-truth set of chunk IDs that are relevant
                                 for this query (from queries.jsonl).
    k             : int        — rank cut-off to evaluate at.

No network calls, no file I/O — every function takes plain Python scalars /
collections and returns a float.  This makes the module trivially unit-testable.
"""


# ==============================================================================
# § Precision@K
# ==============================================================================

def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """
    Fraction of the top-k retrieved items that are relevant.

    P@K = |{retrieved[:k]} ∩ relevant| / k

    Interpretation: "Of the k results shown to the user, how many are actually
    useful?"  Ranges [0, 1]; penalises irrelevant items in the top-k window.
    """
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / k


# ==============================================================================
# § Recall@K
# ==============================================================================

def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """
    Fraction of all relevant items that appear within the top-k results.

    R@K = |{retrieved[:k]} ∩ relevant| / |relevant|

    Interpretation: "Of everything that exists in the knowledge base that could
    answer this query, how much did we surface in the top-k?"  Ranges [0, 1];
    penalises missing relevant chunks regardless of rank.
    Returns 0.0 when relevant_ids is empty to avoid ZeroDivisionError.
    """
    if not relevant_ids or k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / len(relevant_ids)


# ==============================================================================
# § MAP@K  (Mean Average Precision)
# ==============================================================================

def average_precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """
    Average Precision for a single query at rank cut-off k.

    AP@K = (1 / |relevant|) * Σ_{i=1}^{k}  P@i * rel(i)

    where rel(i) = 1 if the item at rank i is relevant, else 0.

    Interpretation: Rewards systems that rank relevant items higher; a hit at
    rank 1 contributes more than a hit at rank k.  Ranges [0, 1].
    Returns 0.0 when relevant_ids is empty.
    """
    if not relevant_ids or k <= 0:
        return 0.0

    score = 0.0
    hits_so_far = 0

    for i, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in relevant_ids:
            hits_so_far += 1
            # Precision at this rank
            score += hits_so_far / i

    # Normalise by the total number of relevant docs (not just those ≤ k)
    return score / len(relevant_ids)


def mean_average_precision(
    results_list: list[tuple[list[str], set[str]]],
    k: int,
) -> float:
    """
    Mean Average Precision across multiple queries at rank cut-off k.

    MAP@K = (1 / |Q|) * Σ_{q} AP@K(q)

    Args:
        results_list: List of (retrieved_ids, relevant_ids) pairs, one per query.
        k:            Rank cut-off.

    Returns:
        Scalar MAP@K in [0, 1].
    """
    if not results_list:
        return 0.0
    ap_scores = [average_precision_at_k(ret, rel, k) for ret, rel in results_list]
    return sum(ap_scores) / len(ap_scores)


# ==============================================================================
# § MRR  (Mean Reciprocal Rank)
# ==============================================================================

def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """
    Reciprocal Rank for a single query.

    RR = 1 / rank_of_first_relevant_item

    Interpretation: Measures how quickly the system surfaces *any* correct
    answer; if the first hit is at rank 1 → RR=1.0, at rank 2 → RR=0.5, etc.
    Returns 0.0 if no relevant item appears in the retrieved list.
    """
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(
    results_list: list[tuple[list[str], set[str]]],
) -> float:
    """
    Mean Reciprocal Rank across multiple queries.

    MRR = (1 / |Q|) * Σ_{q} RR(q)

    Args:
        results_list: List of (retrieved_ids, relevant_ids) pairs, one per query.

    Returns:
        Scalar MRR in [0, 1].
    """
    if not results_list:
        return 0.0
    rr_scores = [reciprocal_rank(ret, rel) for ret, rel in results_list]
    return sum(rr_scores) / len(rr_scores)


# ==============================================================================
# § Aggregation helper
# ==============================================================================

def evaluate_single(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> dict:
    """
    Compute all four metrics for a single query and return them as a dict.

    Returns:
        {
            "precision_at_k": float,
            "recall_at_k":    float,
            "ap_at_k":        float,   # Average Precision (input to MAP)
            "rr":             float,   # Reciprocal Rank  (input to MRR)
            "k":              int,
        }
    """
    return {
        "precision_at_k": precision_at_k(retrieved_ids, relevant_ids, k),
        "recall_at_k":    recall_at_k(retrieved_ids, relevant_ids, k),
        "ap_at_k":        average_precision_at_k(retrieved_ids, relevant_ids, k),
        "rr":             reciprocal_rank(retrieved_ids, relevant_ids),
        "k":              k,
    }


# ==============================================================================
# § Self-test  (run directly:  python -m src.evaluation.metrics)
# ==============================================================================

if __name__ == "__main__":
    # Synthetic example:
    #   retrieved = [a, b, c, d]    relevant = {a, c}    k = 4
    #
    # Expected results:
    #   P@4  = 2/4  = 0.5
    #   R@4  = 2/2  = 1.0
    #   AP@4 = (P@1*1 + P@3*1) / 2  = (1.0 + 2/3) / 2  ≈ 0.8333
    #   RR   = 1/1  = 1.0
    ret = ["a", "b", "c", "d"]
    rel = {"a", "c"}
    k   = 4

    metrics = evaluate_single(ret, rel, k)
    print("=== Metrics Self-Test ===")
    print(f"  retrieved : {ret}")
    print(f"  relevant  : {rel}")
    print(f"  k         : {k}")
    print()
    print(f"  Precision@{k} : {metrics['precision_at_k']:.4f}  (expected 0.5000)")
    print(f"  Recall@{k}    : {metrics['recall_at_k']:.4f}  (expected 1.0000)")
    print(f"  AP@{k}        : {metrics['ap_at_k']:.4f}  (expected 0.8333)")
    print(f"  RR           : {metrics['rr']:.4f}  (expected 1.0000)")

    assert abs(metrics["precision_at_k"] - 0.5)    < 1e-6
    assert abs(metrics["recall_at_k"]    - 1.0)    < 1e-6
    assert abs(metrics["ap_at_k"]        - 5/6)    < 1e-6
    assert abs(metrics["rr"]             - 1.0)    < 1e-6
    print("\nAll assertions passed ✓")
