"""
src/evaluation/generate_qrels.py
=================================
Auto-generate a synthetic multi-relevance evaluation benchmark (qrels) using
Candidate Pooling + LLM-as-a-Judge.

How it works:
-------------
1. Sample a source chunk from the corpus.
2. Ask Gemini to generate a natural, focused query that this chunk answers.
3. Candidate Pooling: Use the vector retriever to fetch the top-M candidate
   chunks from the entire database for that query.
4. LLM-as-a-Judge: Send the candidate pool to Gemini and ask it to evaluate
   which candidate passages actually contain relevant information answering
   or providing essential context for the query.
5. Record {"query": ..., "relevant_ids": [...], "source_chunk": ...} in
   data/eval/queries.jsonl with MULTIPLE ground-truth relevant IDs.

Usage:
------
    python -m src.evaluation.generate_qrels              # sample 50 chunks, pool 10
    python -m src.evaluation.generate_qrels --n 10 --pool-size 10
    python -m src.evaluation.generate_qrels --all        # every chunk

Output:
-------
    data/eval/queries.jsonl   (created or overwritten)
"""

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from src.config import CHUNKS_PATH, LLM_MODEL
from src.generation.llm import Gemini
from src.retriever import retrieve


# ==============================================================================
# § 1  load_chunks — read the indexed corpus from disk
# ==============================================================================

def load_chunks(path: Path = CHUNKS_PATH) -> list[dict]:
    """
    Load all chunked records from data/processed/chunks.jsonl.

    Each record contains at minimum:
        - "id"   : str   — unique chunk identifier (e.g. "niddk_liver-0042")
        - "text" : str   — the passage text to generate a query for
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {path}\n"
            "Run:  python app.py --index   to build it first."
        )
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ==============================================================================
# § 2  generate_query — ask Gemini to write a natural question for a chunk
# ==============================================================================

_QUERY_GEN_SYSTEM = (
    "You are a medical information retrieval expert. "
    "Given a passage of text, generate exactly ONE precise, natural-language "
    "question that the passage directly answers. Return only the question "
    "and nothing else — no preamble, no explanation, no punctuation beyond "
    "the question mark."
)

def generate_query(chunk: dict, llm: Gemini, max_retries: int = 3) -> Optional[str]:
    """Call Gemini to produce a natural-language question for the given chunk."""
    prompt = f"Passage:\n{chunk['text']}\n\nQuestion:"

    for attempt in range(max_retries):
        try:
            question = llm.generate(
                prompt,
                system_instruction=_QUERY_GEN_SYSTEM,
                temperature=0.3,
                max_output_tokens=80,
            ).strip()
            if question:
                return question
        except Exception as exc:
            wait = 2 ** attempt
            print(f"    [query-retry {attempt + 1}/{max_retries}] {type(exc).__name__}: sleeping {wait}s")
            time.sleep(wait)

    return None


# ==============================================================================
# § 3  Candidate Pooling + LLM-as-a-Judge — identify ALL relevant chunk IDs
# ==============================================================================

class RelevanceJudgement(BaseModel):
    relevant_chunk_ids: list[str] = Field(
        default_factory=list,
        description="List of IDs of candidate passages that contain relevant information answering or providing essential context for the query."
    )


_JUDGE_SYSTEM = (
    "You are an expert relevance evaluator for search systems. "
    "You will be given a user Query and a list of Candidate Passages (each with an ID and Text). "
    "Determine which of the Candidate Passages contain factual information that directly answers, "
    "partially answers, or provides essential context for the Query. "
    "Return only the list of relevant chunk IDs in the structured response."
)

def judge_relevance_pool(
    query: str,
    source_chunk: dict,
    llm: Gemini,
    pool_size: int = 10,
    max_retries: int = 3,
) -> list[str]:
    """
    Retrieve top candidate chunks for the query, and ask Gemini to judge which
    chunks are relevant. Guarantees the source_chunk is included in the ground-truth.
    """
    # 1. Retrieve candidates using existing retriever
    try:
        retrieved_candidates = retrieve(query, top_k=pool_size)
    except Exception as e:
        print(f"    [retriever-warning] {e}")
        return [source_chunk["id"]]

    # 2. Build Candidate Pool (ensure source_chunk is present)
    candidate_dict = {c["id"]: c for c in retrieved_candidates}
    candidate_dict[source_chunk["id"]] = source_chunk
    candidate_pool = list(candidate_dict.values())

    # Format candidates for LLM review
    passages_text = []
    for i, c in enumerate(candidate_pool, 1):
        passages_text.append(f"[{i}] ID: {c['id']}\nText: {c['text']}")
    candidates_str = "\n\n".join(passages_text)

    prompt = f"Query:\n{query}\n\nCandidate Passages:\n{candidates_str}\n\nEvaluate all relevant passage IDs:"

    # 3. LLM Judge call with structured output
    for attempt in range(max_retries):
        try:
            judgement: RelevanceJudgement = llm.generate_structured(
                prompt=prompt,
                response_schema=RelevanceJudgement,
                system_instruction=_JUDGE_SYSTEM,
                temperature=0.1,
            )
            # Ensure source_chunk is always present in the ground truth
            relevant_set = set(judgement.relevant_chunk_ids)
            relevant_set.add(source_chunk["id"])

            # Keep only valid IDs from the candidate pool
            valid_pool_ids = set(candidate_dict.keys())
            final_ids = list(relevant_set.intersection(valid_pool_ids))
            return final_ids
        except Exception as exc:
            wait = 2 ** attempt
            print(f"    [judge-retry {attempt + 1}/{max_retries}] {type(exc).__name__}: sleeping {wait}s")
            time.sleep(wait)

    # Fallback to source chunk if judge fails
    return [source_chunk["id"]]


# ==============================================================================
# § 4  main — orchestrate multi-relevance sampling + generation + write
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Auto-generate a multi-relevance retrieval benchmark from the indexed corpus."
    )
    parser.add_argument(
        "--n", type=int, default=50,
        help="Number of chunks to sample (default: 50). Ignored if --all is set."
    )
    parser.add_argument(
        "--pool-size", type=int, default=10,
        help="Number of top retrieved candidate chunks to evaluate with LLM Judge (default: 10)."
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Use every chunk in the corpus instead of a random sample."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible sampling (default: 42)."
    )
    parser.add_argument(
        "--out", type=str, default="data/eval/queries.jsonl",
        help="Output path for the qrels file (default: data/eval/queries.jsonl)."
    )
    args = parser.parse_args()

    # --- Load corpus ---
    all_chunks = load_chunks()
    print(f"Loaded {len(all_chunks)} chunks from corpus.")

    # --- Sample ---
    if args.all:
        sample = all_chunks
    else:
        random.seed(args.seed)
        sample = random.sample(all_chunks, min(args.n, len(all_chunks)))
    print(f"Generating multi-relevance queries for {len(sample)} chunks (candidate pool: {args.pool_size})…\n")

    # --- Initialise LLM ---
    llm = Gemini(model=LLM_MODEL)

    # --- Generate, Judge, and Write ---
    out_path = Path(args.out)
    os.makedirs(out_path.parent, exist_ok=True)

    written = 0
    skipped = 0

    with open(out_path, "w", encoding="utf-8") as fh:
        for i, chunk in enumerate(sample, start=1):
            print(f"[{i}/{len(sample)}] {chunk['id']}  ", end="", flush=True)

            # Step 1: Generate Query
            question = generate_query(chunk, llm)
            if question is None:
                print("SKIPPED (no question generated)")
                skipped += 1
                continue

            # Step 2: Pool & Judge Multi-Relevance
            relevant_ids = judge_relevance_pool(
                query=question,
                source_chunk=chunk,
                llm=llm,
                pool_size=args.pool_size,
            )

            print(f"-> {len(relevant_ids)} relevant IDs | {question[:70]}")

            record = {
                "query":        question,
                "relevant_ids": relevant_ids,
                "source_chunk": chunk["id"],
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"\nDone. Written {written} multi-relevance queries to {out_path} ({skipped} skipped).")


if __name__ == "__main__":
    main()
