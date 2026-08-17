import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

# Data Paths
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"
VECTORS_PATH = EMBEDDINGS_DIR / "vectors.npy"
IDS_PATH = EMBEDDINGS_DIR / "ids.json"

# Models
EMBEDDING_MODEL = "embed-v4.0"
EMBEDDING_DIMENSION = 1024
LLM_MODEL = "gemini-3.5-flash-lite"
