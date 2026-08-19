"""
Query understanding: turn what the user typed into what each retrieval leg wants.

Two stages, deliberately in this order:

1. A clinical synonym dictionary. Deterministic, free, offline. Patients write
   "fatty liver"; the corpus writes "NAFLD". Every target term here was checked
   to actually occur in chunks.jsonl - mapping to a word the corpus never uses
   would be worse than not expanding at all.

2. An LLM rewrite producing two forms of the question, because the legs want
   different things. Dense retrieval reads meaning, so it wants a fluent
   clinical sentence. BM25 counts token overlap, so "what should i eat if my"
   is pure noise to it and it wants keywords only.

The LLM stage is optional. Without GEMINI_API_KEY the parser still expands
terms and strips stopwords, so retrieval never depends on the network.

No metadata filters are produced. A wrong corpus/topic guess silently removes
good chunks - and on a 348-chunk corpus there is nothing to gain by narrowing
the search in the first place. Filters stay available on hybrid_search() for
callers who genuinely know the constraint.

Usage:  python -m src.retriever.query_parser "your question"
"""

import json
import os
import re
import time
import sys
from dataclasses import dataclass, field
from pathlib import Path
from google import genai
from google.genai import types  # Import type
from dotenv import load_dotenv

from src.config import LLM_MODEL

MAX_RETRIES = 4

# Fail loudly instead of degrading to dictionary-only. Benchmarks set this so
# a rate-limited run cannot be mistaken for a real measurement.
STRICT = os.environ.get("STRICT_QUERY_REWRITE") == "1"

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# Colloquial or abbreviated phrasing -> the term the corpus actually uses.
# Expansions are appended rather than substituted, so the user's own wording
# still matches documents that happen to use it.
CLINICAL_TERMS = {
    "fatty liver": "NAFLD nonalcoholic fatty liver disease",
    "liver fat": "NAFLD",
    "hep a": "hepatitis A",
    "hep b": "hepatitis B",
    "hep c": "hepatitis C",
    "hep d": "hepatitis D",
    "hep e": "hepatitis E",
    "hbv": "hepatitis B virus",
    "hcv": "hepatitis C virus",
    "hbsag": "hepatitis B surface antigen",
    "pbc": "primary biliary cholangitis",
    "psc": "primary sclerosing cholangitis",
    "yellow skin": "jaundice",
    "yellow eyes": "jaundice",
    "yellowing": "jaundice",
    "liver scarring": "cirrhosis",
    "scarred liver": "cirrhosis",
    "iron overload": "hemochromatosis",
    "too much iron": "hemochromatosis",
    "copper buildup": "Wilson disease",
    "too much copper": "Wilson disease",
    "fluid in belly": "ascites",
    "swollen belly": "ascites",
    "belly swelling": "ascites",
    "confusion from liver": "hepatic encephalopathy",
    "new liver": "liver transplant",
    "liver operation": "liver transplant",
    "screening guideline": "USPSTF recommendation",
    "task force": "USPSTF",
}

# Dropped from the BM25 form only. BM25 scores by token overlap, so question
# scaffolding actively competes with the terms that matter.
STOPWORDS = {
    "a", "about", "am", "an", "and", "any", "are", "as", "at", "be", "been",
    "by", "can", "could", "do", "does", "for", "from", "get", "getting", "give",
    "had", "has", "have", "how", "i", "if", "in", "into", "is", "it", "its",
    "long", "many", "me", "much", "my", "need", "of", "on", "or", "our", "should",
    "so", "some", "tell", "that", "the", "their", "them", "there", "these",
    "they", "this", "to", "was", "we", "what", "when", "where", "which", "who",
    "why", "will", "with", "would", "you", "your",
}

_CLINICAL_DICT_TEXT = "\n".join(
    f"  - '{k}' -> '{v}'" for k, v in CLINICAL_TERMS.items()
)

_PROMPT = f"""You prepare patient questions about liver disease for a search engine (Dense Vector + BM25 Sparse).

### Canonical Term Mappings (Preferred Corpus Terms):
{_CLINICAL_DICT_TEXT}

### Step 1 — Classify the input into exactly one of three intents.

"chitchat" — no health content at all:
- a greeting or farewell: "hi", "hello", "thanks", "bye"
- small talk: "how are you", "what's up"
- a question about you the assistant: "what can you do"

"vague" — the person reports feeling unwell or raises a health concern, but
names no condition, symptom, test or body system specific enough to search:
- "i feel sick", "something is wrong with me", "i don't feel right"
- "i'm worried about my health", "i've been feeling off lately"
A single named symptom is NOT vague - "my eyes are yellow" is medical.

"medical" — anything answerable from liver-disease material: a condition, a
symptom, a test, a treatment, diet, screening, or prognosis.

Set "needs_retrieval" true only for "medical". For "chitchat" and "vague"
return empty strings for both queries - do NOT invent a medical topic, and do
NOT fall back to "liver disease".

For "vague", also return "clarify": one short question naming two or three
concrete liver-related symptoms the person could confirm or rule out.
Leave "clarify" empty for the other two intents.

### Step 2 — Only when needs_retrieval is true, build the queries.
- "dense_query": a clinical rephrasing of the user's actual question, one sentence.
- "sparse_query": key medical terms only, space-separated, no stopwords, no punctuation.

Rules:
- Rewrite only what the user asked. Never add symptoms, conditions, or
  scope they did not mention.
- Apply the dictionary mappings above whenever a matching colloquial phrase appears.
- For terms not in the dictionary, use standard clinical terminology.

### Output format
Return ONLY a valid JSON object with exactly five keys:
intent, needs_retrieval, dense_query, sparse_query, clarify.

### Examples
Question: hi
JSON Output: {{"intent": "chitchat", "needs_retrieval": false, "dense_query": "", "sparse_query": "", "clarify": ""}}

Question: how are you
JSON Output: {{"intent": "chitchat", "needs_retrieval": false, "dense_query": "", "sparse_query": "", "clarify": ""}}

Question: i feel sick
JSON Output: {{"intent": "vague", "needs_retrieval": false, "dense_query": "", "sparse_query": "", "clarify": "Sorry to hear that. Are you noticing anything specific - yellowing of the skin or eyes, pain or swelling in your abdomen, unusual tiredness, or dark urine?"}}

Question: something is wrong with me
JSON Output: {{"intent": "vague", "needs_retrieval": false, "dense_query": "", "sparse_query": "", "clarify": "I can help with liver-related concerns. Are you experiencing symptoms such as jaundice, abdominal pain or swelling, nausea, or unexplained fatigue?"}}

Question: my eyes went yellow last week
JSON Output: {{"intent": "medical", "needs_retrieval": true, "dense_query": "What causes scleral icterus and jaundice in liver disease?", "sparse_query": "jaundice scleral icterus", "clarify": ""}}

Question: is fatty liver reversible
JSON Output: {{"intent": "medical", "needs_retrieval": true, "dense_query": "Is hepatic steatosis reversible through treatment or lifestyle modification?", "sparse_query": "hepatic steatosis reversibility treatment", "clarify": ""}}

Question: {{query}}
JSON Output:"""

_client = None


@dataclass
class ParsedQuery:
    """What the retrieval legs consume. `raw` is kept for logging and display."""

    raw: str
    dense_query: str
    sparse_query: str
    expansions: list = field(default_factory=list)
    used_llm: bool = False
    # False only when the model explicitly judged the input unsearchable -
    # a greeting, small talk, or a question about the assistant. Callers skip
    # retrieval on it. Defaults True so any path that cannot make that
    # judgement (dictionary fallback, no key) still searches as before.
    needs_retrieval: bool = True
    # "medical" | "vague" | "chitchat". Only "medical" is searchable, but the
    # other two need different replies: a greeting deserves the scope message,
    # while someone saying "i feel sick" is raising a health concern and being
    # handed a menu of topics reads as dismissive. Defaults to "medical" so
    # every path that cannot classify (dictionary fallback, no key) searches
    # as before.
    intent: str = "medical"
    # For "vague" only: one short question naming concrete symptoms to confirm.
    clarify: str = ""


def expand_clinical_terms(query):
    """
    Append the canonical term for any colloquial phrase found.

    Matching is on token presence rather than exact substring, because people
    do not type phrases in dictionary order - "my liver is fatty" and "eyes are
    yellow" both have to hit, and a substring match catches neither. Phrases
    here are two or three specific words, so co-occurrence is a strong enough
    signal.

    Appending rather than replacing is deliberate: substitution can destroy a
    query when a phrase matches inside a larger one, and keeping both forms
    costs nothing in a bag-of-words leg.
    """
    query_tokens = {t.lower() for t in re.findall(r"[A-Za-z0-9]+", query)}
    expanded, applied = query, []

    for phrase, canonical in CLINICAL_TERMS.items():
        phrase_tokens = {t.lower() for t in re.findall(r"[A-Za-z0-9]+", phrase)}
        if not phrase_tokens <= query_tokens:
            continue
        # Skip if the canonical term is already present anyway.
        if re.search(rf"\b{re.escape(canonical.split()[0])}\b", expanded, re.IGNORECASE):
            continue
        expanded = f"{expanded} {canonical}"
        applied.append(f"{phrase} -> {canonical}")
    return expanded, applied


def keywords_only(text):
    """Strip stopwords and punctuation. The deterministic BM25 form."""
    tokens = re.findall(r"[A-Za-z0-9]+", text)
    kept = [t for t in tokens if t.lower() not in STOPWORDS]
    return " ".join(kept) if kept else text


def bm25_query(query: str) -> str:
    """Expand clinical terms then strip stopwords - the BM25 query form."""
    expanded, _ = expand_clinical_terms(query)
    return keywords_only(expanded)


def _get_client():
    global _client
    if _client is None:
        key = os.environ.get("GEMINI_API_KEY")
        if not key or key == "your-key-here":
            return None
        from google import genai

        _client = genai.Client(api_key=key)
    return _client



def _retry_delay(exc, attempt):
    """Seconds to wait. Google reports its own retryDelay on a 429; honour it."""
    match = re.search(r"'retryDelay':\s*'(\d+)s'", str(exc))
    if match:
        return int(match.group(1)) + 1
    return min(2 ** attempt, 60)


def _llm_rewrite(query):
    """
    Ask the LLM for both query forms.

    On the free tier Gemini allows 15 requests/minute and returns 429 beyond
    that. Silently dropping to the dictionary was wrong for evaluation: the
    parse-enabled configs got scored without the rewrite they exist to test,
    so their numbers looked worse than the feature actually is. Rate limits
    are now waited out rather than swallowed.

    STRICT_QUERY_REWRITE=1 turns any remaining failure into an exception, so
    a benchmark aborts instead of quietly reporting a degraded run.
    """
    client = _get_client()
    if client is None:
        if STRICT:
            raise RuntimeError("GEMINI_API_KEY not set and STRICT_QUERY_REWRITE=1")
        return None

    last = None
    for attempt in range(MAX_RETRIES):
        try:
            # .replace, not .format: the prompt is an f-string whose {{ }}
            # already collapsed to literal braces, so a second format() pass
            # reads the JSON examples as placeholders and raises KeyError on
            # '"needs_retrieval"'. That killed every rewrite silently.
            filled = _PROMPT.replace("{query}", query)

            raw = client.models.generate_content(
                    model=LLM_MODEL,
                    contents=filled,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        # Extraction, not composition: one right answer, and
                        # a cached/benchmarked rewrite has to be repeatable.
                        temperature=0.0,
                    ),
                ).text
            data = json.loads(raw)
            dense = str(data.get("dense_query", "")).strip()
            sparse = str(data.get("sparse_query", "")).strip()
            intent = str(data.get("intent", "medical")).strip().lower()
            clarify = str(data.get("clarify", "")).strip()

            # An explicit refusal is a valid answer, not a failure. "hi" is
            # meant to come back with both queries empty; treating that as an
            # error sent it to the dictionary and searched the greeting.
            if data.get("needs_retrieval") is False:
                return "", "", False, intent, clarify
            if dense and sparse:
                return dense, sparse, True, "medical", ""
            last = ValueError("LLM returned empty dense_query/sparse_query")
        except Exception as exc:
            last = exc
            if "RESOURCE_EXHAUSTED" in str(exc) or "429" in str(exc):
                wait = _retry_delay(exc, attempt)
                if attempt < MAX_RETRIES - 1:
                    print(f"  rate limited, waiting {wait}s ({attempt + 1}/{MAX_RETRIES})")
                    time.sleep(wait)
                    continue
            elif attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
                continue
        break

    if STRICT:
        raise RuntimeError(f"query rewrite failed after {MAX_RETRIES} attempts: {last}")
    print(f"  query rewrite unavailable ({type(last).__name__}), using dictionary only")
    return None


_cache = {}


def parse_query(query, use_llm=True):
    """
    Build a ParsedQuery. Never raises and never returns empty fields - if every
    optional stage fails the caller still gets the original question back.

    Results are memoised per process. The rewrite is deterministic for a given
    question, and benchmarks sweep the same query set across several k values,
    so without this a 35-query set costs 140 LLM calls instead of 35 - enough
    to spend most of a run sitting out rate limits.
    """
    query = (query or "").strip()
    if not query:
        return ParsedQuery(raw=query, dense_query=query, sparse_query=query)

    key = (query, use_llm)
    if key in _cache:
        return _cache[key]

    expanded, applied = expand_clinical_terms(query)

    rewritten = _llm_rewrite(expanded) if use_llm else None
    if rewritten:
        dense, sparse, needs_retrieval, intent, clarify = rewritten
        if not needs_retrieval:
            # Nothing worth searching for. Keep the raw text so callers can
            # echo it, and carry the intent so they can tell a greeting apart
            # from someone saying they feel unwell - the two need different
            # replies.
            parsed = ParsedQuery(query, query, query, applied,
                                 used_llm=True, needs_retrieval=False,
                                 intent=intent, clarify=clarify)
        else:
            parsed = ParsedQuery(query, dense, sparse, applied, used_llm=True,
                                 intent="medical")
    else:
        parsed = ParsedQuery(query, expanded, keywords_only(expanded), applied, used_llm=False)

    _cache[key] = parsed
    return parsed


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "what should i eat if my liver is fatty"
    parsed = parse_query(query)
    print(f"raw    : {parsed.raw}")
    print(f"dense  : {parsed.dense_query}")
    print(f"sparse : {parsed.sparse_query}")
    print(f"expands: {parsed.expansions or 'none'}")
    print(f"llm    : {parsed.used_llm}")


if __name__ == "__main__":
    main()
