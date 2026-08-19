"""
Metrics Generator for RAG Evaluation using Ragas, Gemini LLM, and Cohere embed-v4.0.
Saves outputs to docs/metrics_llm{number}.md and docs/metrics_retrieval{number}.md.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Ensure root directory is in sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from datasets import Dataset
from langchain_cohere import CohereEmbeddings
from ragas import evaluate
from ragas.metrics import (
    answer_correctness,
    context_precision,
    context_recall,
    faithfulness,
)

from src.config import CHUNKS_PATH, LLM_MODEL
from src.generation.llm import get_llm, get_langchain_llm
from src.retriever import retrieve


# ==============================================================================
# Helper Functions
# ==============================================================================


def load_chunks_map() -> dict[str, str]:
    """Load all chunks and return a dict mapping chunk ID to chunk text."""
    if not os.path.exists(CHUNKS_PATH):
        raise FileNotFoundError(
            f"Chunks file not found at {CHUNKS_PATH}. Please run app.py --index first."
        )

    chunks_map = {}
    with open(CHUNKS_PATH, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                chunk = json.loads(line)
                chunks_map[chunk["id"]] = chunk["text"]
    return chunks_map


def load_queries_jsonl(queries_path: str) -> list[dict]:
    """Load evaluation queries directly from JSONL file."""
    queries = []
    with open(queries_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))
    return queries


def generate_rag_response(query: str, top_k: int = 5) -> tuple[str, list[dict]]:
    """Execute full RAG generation pipeline and return (answer, filtered_chunks)."""
    retrieved_chunks = retrieve(query, top_k=top_k)

    from src.generation.pipeline import ALWAYS_KEEP_TOP, RELEVANCE_THRESHOLD

    if retrieved_chunks and "rerank_score" in retrieved_chunks[0]:
        filtered_chunks = [
            c
            for i, c in enumerate(retrieved_chunks)
            if i < ALWAYS_KEEP_TOP or c["rerank_score"] >= RELEVANCE_THRESHOLD
        ]
    else:
        filtered_chunks = retrieved_chunks

    context_blocks = []
    for i, c in enumerate(filtered_chunks, 1):
        context_blocks.append(
            f"--- Context [{i}] (Score: {c.get('score', 0.0):.3f}) ---\n"
            f"ID: {c.get('id', 'N/A')}\n"
            f"Content:\n{c.get('text', '')}"
        )
    context_str = "\n\n".join(context_blocks)

    system_instruction = (
        "You are a medical AI assistant specialized in liver diseases and hepatology.\n"
        "Answer strictly based on the retrieved context."
    )

    prompt = f"Context:\n{context_str if context_str else 'No relevant context found.'}\n\nUser Question: {query}\n\nAnswer:"
    llm = get_llm(system_instruction)
    response = llm.generate(prompt, system_instruction)
    return response, filtered_chunks


def get_next_number(docs_dir: Path) -> int:
    """Find the next available number for saving evaluation results."""
    existing_files = os.listdir(docs_dir) if docs_dir.exists() else []
    max_num = 0
    for filename in existing_files:
        for prefix in [
            "metrics_llm",
            "metrices_llm",
            "metrics_retrieval",
            "metrices_retrieval",
        ]:
            if filename.startswith(prefix) and filename.endswith(".md"):
                try:
                    num_part = filename[len(prefix) : -3]
                    num = int(num_part)
                    if num > max_num:
                        max_num = num
                except ValueError:
                    continue
    return max_num + 1


# ==============================================================================
# Main Runner
# ==============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RAG pipeline using Gemini LLM and Cohere Embeddings via Ragas."
    )
    parser.add_argument(
        "--queries", default="data/eval/queries.jsonl", help="Path to queries file"
    )
    parser.add_argument(
        "--limit", type=int, default=5, help="Limit number of queries to evaluate"
    )
    parser.add_argument("--k", type=int, default=5, help="Retrieval K parameter")
    args = parser.parse_args()

    print("Loading queries and chunks corpus...")
    queries = load_queries_jsonl(args.queries)
    if args.limit:
        queries = queries[: args.limit]
        print(f"Limited evaluation to top {args.limit} queries.")

    chunks_map = load_chunks_map()

    eval_data = {
        "user_input": [],
        "response": [],
        "retrieved_contexts": [],
        "reference": [],
    }

    print(f"Running RAG pipeline for {len(queries)} queries...")
    for idx, q_item in enumerate(queries, 1):
        query = q_item["query"]
        relevant_ids = q_item.get("relevant_ids", [])

        ground_truth = q_item.get("response")
        if not ground_truth:
            ground_truth = "\n\n".join(
                [chunks_map[rid] for rid in relevant_ids if rid in chunks_map]
            )

        print(f"[{idx}/{len(queries)}] Processing: {query[:60]}...")

        live_answer, retrieved = generate_rag_response(query, top_k=args.k)
        retrieved_texts = [c["text"] for c in retrieved]

        eval_data["user_input"].append(query)
        eval_data["response"].append(live_answer)
        eval_data["retrieved_contexts"].append(retrieved_texts)
        eval_data["reference"].append(ground_truth)

    dataset = Dataset.from_dict(eval_data)

    # Initialize Gemini for LLM evaluation and Cohere embed-v4.0 for embedding evaluation
    evaluator_llm = get_langchain_llm(temperature=0.0)
    evaluator_embeddings = CohereEmbeddings(model="embed-v4.0")

    print("\nEvaluating with Ragas (Gemini + Cohere embed-v4.0)...")
    ragas_result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_correctness,
            context_precision,
            context_recall,
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    df_results = ragas_result.to_pandas()

    # Aggregate metric scores using pandas mean for safety
    summary = df_results.mean(numeric_only=True).to_dict()

    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    num = get_next_number(docs_dir)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    # 1. Write LLM Metrics Report
    llm_markdown = f"""# LLM Evaluation Report (Run #{num})

**LLM Evaluator**: {LLM_MODEL}
**Embedding Model**: embed-v4.0 (Cohere)
**Queries Evaluated**: {len(queries)}
**Evaluation Time**: {timestamp}

## Summary Metrics

| Metric | Average Score | Description |
|---|---|---|
| **Faithfulness** | **{summary.get('faithfulness', 0.0):.3f}** | Adherence of answer to retrieved context |
| **Answer Correctness** | **{summary.get('answer_correctness', 0.0):.3f}** | Accuracy & completeness against reference ground truth |

## Per-Query Detail

"""
    for idx, row in df_results.iterrows():
        llm_markdown += f"""### Query: {row['user_input']}

**Generated Answer**:
{row['response']}

**Reference Answer**:
{row['reference']}

* **Faithfulness Score**: {row.get('faithfulness', 0.0):.3f}
* **Answer Correctness Score**: {row.get('answer_correctness', 0.0):.3f}

---
"""

    # 2. Write Retrieval Metrics Report
    ret_markdown = f"""# Retrieval Evaluation Report (Run #{num})

**K Parameter**: {args.k}
**Embedding Model**: embed-v4.0 (Cohere)
**Queries Evaluated**: {len(queries)}
**Evaluation Time**: {timestamp}

## Summary Metrics

| Metric | Score | Description |
|---|---|---|
| **Context Precision** | **{summary.get('context_precision', 0.0):.3f}** | Signal-to-noise ratio in retrieved context |
| **Context Recall** | **{summary.get('context_recall', 0.0):.3f}** | Ratio of relevant context retrieved |

## Per-Query Detail

| Query | Context Precision | Context Recall |
|---|---|---|
"""
    for idx, row in df_results.iterrows():
        q_trunc = (
            row["user_input"][:60] + "..."
            if len(row["user_input"]) > 60
            else row["user_input"]
        )
        ret_markdown += f"| {q_trunc} | {row.get('context_precision', 0.0):.3f} | {row.get('context_recall', 0.0):.3f} |\n"

    # Save reports
    for prefix in ["metrics_llm", "metrices_llm"]:
        with open(docs_dir / f"{prefix}{num}.md", "w", encoding="utf-8") as f:
            f.write(llm_markdown)

    for prefix in ["metrics_retrieval", "metrices_retrieval"]:
        with open(docs_dir / f"{prefix}{num}.md", "w", encoding="utf-8") as f:
            f.write(ret_markdown)

    print(f"\nSaved LLM metrics report to docs/metrics_llm{num}.md")
    print(f"Saved Retrieval metrics report to docs/metrics_retrieval{num}.md")


if __name__ == "__main__":
    main()