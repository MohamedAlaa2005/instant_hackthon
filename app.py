import argparse
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path(__file__).resolve().parent / ".env")

from src.vector_db.collections import get_collection
from src.indexing.parse import main as run_parse_chunk
from src.vector_db.collections import build as run_build_index
from src.generation import run_rag


# ==========================================
# 1. Pipeline Indexing Routine
# ==========================================

def index_pipeline():
    """Run end-to-end processing from raw files into Chroma vector store."""
    print("=== Step 1: Parsing & Chunking Raw Data ===")
    run_parse_chunk()

    print("\n=== Step 2: Embedding & Indexing into Chroma ===")
    run_build_index()
    print("\nIndexing completed successfully!")


# ==========================================
# 2. CLI Entry Point
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="End-to-End RAG System with Gemini & Cohere")
    parser.add_argument("--index", action="store_true", help="Parse data and generate vector index")
    parser.add_argument("--query", type=str, help="Run RAG query against stored knowledge base")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve (default: 5)")
    parser.add_argument("--no-verify", action="store_true", help="Skip post-generation verification audit")

    args = parser.parse_args()

    verify_enabled = not args.no_verify

    if args.index:
        index_pipeline()
    elif args.query:
        run_rag(args.query, top_k=args.top_k, verify=verify_enabled)
    else:
        # Interactive loop if no flags supplied
        if get_collection().count() == 0:
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
                run_rag(user_input, top_k=args.top_k, verify=verify_enabled)
            except KeyboardInterrupt:
                break


if __name__ == "__main__":
    main()