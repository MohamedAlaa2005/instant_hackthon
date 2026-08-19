"""
What a chunk looks like inside the vector store.

Chroma metadata values must be scalars (str/int/float/bool), so this is a flat
projection of a chunk record. Only fields listed here are filterable at query
time - the chunk text itself lives in Chroma's `documents`, not in metadata.
"""

from src.config import EMBEDDING_DIMENSION

COLLECTION_NAME = "liver_rag"
DIMENSION = EMBEDDING_DIMENSION

# Fields promoted to filterable metadata.
METADATA_FIELDS = ("corpus", "topic", "section", "heading", "url", "source")

# Cosine, because the Cohere vectors are unit-normalized.
DISTANCE_METRIC = "cosine"


def to_metadata(chunk):
    """Flatten a chunk record into Chroma metadata."""
    return {field: str(chunk.get(field, "")) for field in METADATA_FIELDS}


def build_where(corpus=None, topic=None, section=None):
    """
    Compose a Chroma `where` filter. Returns None when nothing is constrained,
    which Chroma treats as an unfiltered search.

    Chroma needs an explicit $and once there is more than one condition.
    """
    clauses = []
    if corpus:
        clauses.append({"corpus": corpus})
    if topic:
        clauses.append({"topic": topic})
    if section:
        clauses.append({"section": section})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}

# ---------------------------------------------------------------------------
# Contextual retrieval
# ---------------------------------------------------------------------------

# A chunk is embedded and BM25-indexed with its document and section name
# prepended, so a passage that never names its own subject can still be found.
# 52 of 348 chunks contain no disease term at all - pdf_documents-0041 reads in
# full "This recommendation applies to all asymptomatic adults aged 18 to 79
# years", which is unfindable from a hepatitis C query without this.
#
# Anthropic's version generates the context with an LLM, 50-100 tokens per
# chunk. That is 33-65% of a chunk here (median 153 tokens), and since the
# context is near-identical across a document, at that share every chunk in a
# document embeds to nearly the same vector - you find the right document and
# lose the ranking within it. Reusing metadata costs ~8 tokens instead, and no
# API calls.

# Skip the prefix when it would outweigh the passage. 47 chunks are under 60
# tokens; on those even a short prefix can dominate.
MAX_PREFIX_SHARE = 0.35


def contextualize(chunk: dict) -> str:
    """Chunk text with its document/section context prepended, for indexing."""
    text = chunk.get("text", "")
    lowered = text.lower()

    parts = []
    for field in ("topic", "section"):
        value = (chunk.get(field) or "").strip()
        if not value or value.lower() in lowered:
            continue  # already stated in the passage
        if any(value.lower() in p.lower() or p.lower() in value.lower() for p in parts):
            continue  # section repeats the topic, as it does on the PDF title pages
        parts.append(value)

    if not parts:
        return text

    prefix = " - ".join(parts)
    if len(prefix.split()) > MAX_PREFIX_SHARE * max(len(text.split()), 1):
        return text

    return f"{prefix}: {text}"
