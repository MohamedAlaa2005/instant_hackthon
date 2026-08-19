"""
Load the chunked corpus into Chroma and search it.

Embeddings are computed here at index time using Cohere and stored
directly inside Chroma's persistent store (data/chroma/).
No external .npy / .json files are needed.

Usage:  python -m src.vector_db.collections          # build the index
        python -m src.vector_db.collections "query"  # build, then search
"""

import json
import os
import sys
import time

import cohere
import numpy as np
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from src.config import CHUNKS_PATH, EMBEDDING_MODEL, EMBEDDING_DIMENSION
from src.vector_db.client import get_client
from src.vector_db.schemas import (
    contextualize,
    COLLECTION_NAME,
    DISTANCE_METRIC,
    build_where,
    to_metadata,
)

BATCH_SIZE = 96   # Cohere max texts per embed call
ADD_BATCH = 500   # Chroma max docs per add call
MAX_RETRIES = 5


# ---------------------------------------------------------------------------
# Cohere client
# ---------------------------------------------------------------------------

def _get_cohere():
    key = os.environ.get("COHERE_API_KEY")
    if not key:
        raise SystemExit("COHERE_API_KEY not set — add it to .env")
    return cohere.ClientV2(key)


def _embed_batch(client, texts: list[str], input_type: str) -> list[list[float]]:
    """Embed a single batch with exponential backoff for rate limits."""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.embed(
                texts=texts,
                model=EMBEDDING_MODEL,
                input_type=input_type,
                embedding_types=["float"],
                output_dimension=EMBEDDING_DIMENSION,
            )
            return response.embeddings.float_
        except Exception as exc:
            if attempt == MAX_RETRIES - 1:
                raise
            is_429 = "429" in str(exc) or "TooManyRequests" in type(exc).__name__
            wait = (15 * (attempt + 1)) if is_429 else (2 ** attempt)
            print(f"  [embed-retry {attempt+1}/{MAX_RETRIES}] sleeping {wait}s ({type(exc).__name__})")
            time.sleep(wait)


def embed_texts(texts: list[str], input_type: str = "search_document") -> list[list[float]]:
    """
    Embed a list of texts, returning a flat list of vectors.

    """
    client = _get_cohere()
    vectors = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start: start + BATCH_SIZE]
        vectors.extend(_embed_batch(client, batch, input_type))
        print(f"  embedded {min(start + BATCH_SIZE, len(texts))}/{len(texts)}")
    return vectors


_query_cache: dict[str, list[float]] = {}


def embed_query(text: str) -> list[float]:
    """
    Embed a single query string.

    Memoised per process. Embedding is deterministic, and a benchmark runs the
    same query set through five configs at several k values - 35 queries became
    ~700 identical Cohere calls, which exhausted the trial rate limit and
    aborted the run. Serving repeats from memory keeps the call count at one
    per distinct query.
    """
    if text in _query_cache:
        return _query_cache[text]
    client = _get_cohere()
    vector = _embed_batch(client, [text], "search_query")[0]
    _query_cache[text] = vector
    return vector


# ---------------------------------------------------------------------------
# Collection helpers
# ---------------------------------------------------------------------------

def get_collection():
    """Fetch (or create) the Chroma collection."""
    return get_client().get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": DISTANCE_METRIC},
    )


def load_chunks(path=CHUNKS_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def build():
    """
    Rebuild the Chroma collection from data/processed/chunks.jsonl.
    Embeddings are computed on the fly — no .npy files needed.
    """
    chunks = load_chunks()
    print(f"{len(chunks)} chunks -> {EMBEDDING_MODEL} ({EMBEDDING_DIMENSION}d)")

    # Embed the contextualised form so a passage that never names its own
    # subject is still reachable; store the raw text, since the generator
    # should quote the passage rather than the heading glued to it.
    texts = [c["text"] for c in chunks]
    vectors = embed_texts([contextualize(c) for c in chunks],
                          input_type="search_document")

    client = get_client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # first build for this provider
    collection = get_collection()

    ids = [c["id"] for c in chunks]
    for start in range(0, len(chunks), ADD_BATCH):
        stop = min(start + ADD_BATCH, len(chunks))
        collection.add(
            ids=ids[start:stop],
            embeddings=vectors[start:stop],
            documents=texts[start:stop],
            metadatas=[to_metadata(c) for c in chunks[start:stop]],
        )
        print(f"  indexed {stop}/{len(chunks)}")

    print(f"\ncollection '{COLLECTION_NAME}': {collection.count()} vectors")
    return collection


def search(query: str, k: int = 5, corpus=None, topic=None, section=None):
    """
    Semantic search with optional metadata filtering.

    Returns dicts with a `score` in [0, 1] where higher is closer. Chroma
    reports cosine *distance*, so it is converted here.
    """
    collection = get_collection()
    result = collection.query(
        query_embeddings=[embed_query(query)],
        n_results=k,
        where=build_where(corpus, topic, section),
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for chunk_id, doc, meta, dist in zip(
        result["ids"][0],
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        hits.append({"id": chunk_id, "text": doc, "score": 1.0 - dist, **meta})
    return hits


def main():
    build()
    query = sys.argv[1] if len(sys.argv) > 1 else "Who should be screened for hepatitis C?"
    print(f"\nquery: {query}")
    for hit in search(query, k=3):
        print(f"  {hit['score']:.3f} [{hit['corpus']}] {hit['topic'][:34]} > {hit['heading'][:40]}")


if __name__ == "__main__":
    main()
