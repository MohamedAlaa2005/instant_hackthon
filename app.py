import argparse
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path(__file__).resolve().parent / ".env")

from src.config import VECTORS_PATH
from src.indexing.parse import main as run_parse_chunk
from src.indexing.embeddings import main as run_embeddings
from src.generation import run_rag
from src.evaluation import run_evaluation, load_qrels
from src.evaluation.runner import print_report


# ==========================================
# 1. Pipeline Indexing Routine
# ==========================================

def index_pipeline():
    """Run end-to-end processing from raw files to embedded vectors."""
    print("=== Step 1: Parsing & Chunking Raw Data ===")
    run_parse_chunk()

    print("\n=== Step 2: Generating Cohere Embeddings ===")
    run_embeddings()
    print("\nIndexing completed successfully!")


# ==========================================
# 2. CLI Entry Point
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="End-to-End RAG System with Gemini & Cohere")
    parser.add_argument("--index", action="store_true", help="Parse data and generate vector index")
    parser.add_argument("--query", type=str, help="Run RAG query against stored knowledge base")
    parser.add_argument("--top-k", type=int, default=4, help="Number of chunks to retrieve (default: 4)")
    parser.add_argument("--eval", type=str, nargs="?", const="data/eval/queries.jsonl",
        metavar="QRELS_PATH",
        help="Run retrieval evaluation against a qrels JSONL file "
             "(default: data/eval/queries.jsonl)"
    )

    args = parser.parse_args()

    if args.index:
        index_pipeline()
    elif args.query:
        run_rag(args.query, top_k=args.top_k)
    elif args.eval:
        qrels  = load_qrels(args.eval)
        summary = run_evaluation(qrels, top_k=args.top_k, verbose=True)
        print_report(summary)
    else:
        # Interactive loop if no flags supplied
        if not os.path.exists(VECTORS_PATH):
            print("No index detected. Building vector index now...")
            index_pipeline()

        print("\n=== Interactive RAG System ready! (Type 'exit' to quit) ===")
        while True:
            try:
                user_input = input("\nAsk a question > ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ("exit", "quit"):
                    break
                run_rag(user_input, top_k=args.top_k)
            except KeyboardInterrupt:
                break


if __name__ == "__main__":
    main()