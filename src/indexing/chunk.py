import re
import ftfy
import wordninja
from langchain_cohere import CohereEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

# Module-level references — populated lazily on first use so that:
#   a) importing this module never crashes due to a missing API key, and
#   b) clean_text_for_embedding() (no network calls) can be used standalone.
_embeddings = None
_semantic_chunker = None


def _get_chunker():
    """Lazy-initialize Cohere embeddings + SemanticChunker on first call."""
    global _embeddings, _semantic_chunker
    if _semantic_chunker is None:
        # 1. Initialize Cohere Embeddings with embed-v4.0
        # Requires COHERE_API_KEY set in environment variables
        _embeddings = CohereEmbeddings(
            model="embed-v4.0",
            user_agent="semantic-chunking"
        )
        # 2. Initialize the Semantic Chunker
        _semantic_chunker = SemanticChunker(
            embeddings=_embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95.0,  # Adjust lower (e.g. 80-90) for smaller, more frequent chunks
        )
    return _semantic_chunker


def clean_text_for_embedding(text: str) -> str:
    """Sanitize raw text extracted from any PDF without publisher-specific rules."""
    if not text:
        return ""

    # 1. Fix encoding and unicode corruptions
    text = ftfy.fix_text(text)

    # 2. Remove HTML / XML tags (e.g. <sup>1,28</sup>)
    text = re.sub(r"<[^>]+>", "", text)

    # 3. Strip standalone URLs and emails
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"\S+@\S+\.\S+", "", text)

    # 4. Remove inline numeric citation artifacts (e.g., "time.1,28" -> "time.")
    text = re.sub(r"(?<=[a-zA-Z.])\d+(?:[,–-]\d+)*(?=\s|[A-Z]|$)", "", text)
    text = re.sub(r"\[\d+(?:[,–-]\d+)*\]", "", text)  # Bracketed [1, 2]

    # 5. Fix Markdown/Math symbol corruptions (e.g., "_ P _ < .001" -> "P < .001")
    text = re.sub(r"_\s*([A-Za-z])\s*_", r"\1", text)

    # 6. Fix concatenated words from PDF font glues (e.g., 'estimatednumberofinfants')
    words = text.split()
    fixed_words = [
        " ".join(wordninja.split(word)) if len(word) > 18 and word.isalpha() else word
        for word in words
    ]
    text = " ".join(fixed_words)

    # 7. Collapse all multi-whitespace and linebreaks into single spaces
    return re.sub(r"\s+", " ", text).strip()


def chunk_text_semantically(text: str) -> list[str]:
    """Clean text and chunk it based on semantic transitions in Cohere embed-v4.0 vector space."""
    cleaned = clean_text_for_embedding(text)
    if not cleaned:
        return []

    # create_documents returns LangChain Document objects; return page_content strings
    docs = _get_chunker().create_documents([cleaned])
    return [doc.page_content for doc in docs]


