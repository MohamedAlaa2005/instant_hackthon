import os
from src.config import LLM_MODEL
from src.generation.llm import get_llm
from src.retriever import retrieve
from src.retriever.query_parser import parse_query

# Minimum Cohere rerank relevance for a supporting chunk to reach the model.
# Formally-worded questions score ~0.73, colloquial patient questions
# ~0.50-0.65, and weak or irrelevant matches < 0.45.
RELEVANCE_THRESHOLD = 0.45

# The best chunk is always passed through, whatever it scores.
#
# Some legitimate questions score below the threshold on every chunk - "my
# eyes are yellow what does that mean" tops out at 0.307 - and filtering them
# to nothing answered a real jaundice question with "insufficient
# information". Small talk is rejected by needs_retrieval before retrieval
# runs, so the threshold no longer has to double as a scope check; it only has
# to keep near-misses out of the prompt. The model still declines when the
# passage does not answer the question, and unlike a number it can read it.
ALWAYS_KEEP_TOP = 1


def run_rag(query: str, top_k: int = 5, stream: bool = True):
    """Execute full RAG generation pipeline with query parsing and threshold filtering."""
    print(f"\n[Raw Query]: {query}")

    # 1. Parse and expand query terms
    parsed = parse_query(query)
    print("\n[Query Parsing Breakdown]:")
    print(f"  Dense Query  : {parsed.dense_query}")
    print(f"  Sparse Query : {parsed.sparse_query}")
    print(f"  Expansions   : {parsed.expansions or 'None'}")
    print(f"  Intent       : {parsed.intent}")
    print(f"  LLM Rewritten: {parsed.used_llm}\n")

    # 2. Nothing searchable. Two different reasons need two different replies.
    #    Searching either costs an embedding and a rerank call to return chunks
    #    the model then has to decline anyway.
    if not parsed.needs_retrieval:
        print(f"Intent '{parsed.intent}' - skipping retrieval.")
        print("\n[Answer]:")
        if parsed.intent == "vague" and parsed.clarify:
            # Someone reporting they feel unwell has raised a health concern.
            # Handing them the same capability blurb a greeting gets reads as
            # dismissive, so ask what they are actually experiencing instead.
            print(parsed.clarify)
        else:
            print("I answer questions about liver disease using NIDDK patient "
                  "information and USPSTF screening guidelines. Ask me about "
                  "symptoms, causes, diagnosis, treatment, diet, or screening.")
        return

    # 3. Retrieve using the ParsedQuery object
    print(f"Retrieving top {top_k} contexts...")
    retrieved_chunks = retrieve(parsed, top_k=top_k)

    # 4. Drop weak matches, but never drop the best one - see ALWAYS_KEEP_TOP.
    if retrieved_chunks and "rerank_score" in retrieved_chunks[0]:
        filtered_chunks = [
            c for i, c in enumerate(retrieved_chunks)
            if i < ALWAYS_KEEP_TOP or c.get("rerank_score", 0.0) >= RELEVANCE_THRESHOLD
        ]
        print(f"Retrieved {len(retrieved_chunks)} chunks; {len(filtered_chunks)} remain "
              f"after filtering (rerank_score >= {RELEVANCE_THRESHOLD}, "
              f"top {ALWAYS_KEEP_TOP} always kept).")
    else:
        filtered_chunks = retrieved_chunks
        print(f"Retrieved {len(retrieved_chunks)} chunks; no rerank scores, filter skipped.")

    # If no relevant chunks found after thresholding, return early
    if not filtered_chunks:
        print("\n[Gemini Answer]:")
        print("The provided context does not contain sufficient information to answer this query.\n")
        return

    # 4. Build context block
    context_blocks = []
    for i, c in enumerate(filtered_chunks, 1):
        context_blocks.append(
            f"--- Context [{i}] (Score: {c['score']:.3f}) ---\n"
            f"Source: {c.get('url', 'N/A')} | Path: {c.get('section_path', 'N/A')}\n"
            f"Content:\n{c['text']}"
        )
    context_str = "\n\n".join(context_blocks)

    # 5. System prompt structured using the 4 grounding principles
    system_instruction = (
        "1. ROLE:\n"
        "You are a citation-bound clinical evidence tool specializing in hepatology and liver disease, "
        "not a general advisor. Translate clinical jargon into clear, accessible language.\n\n"
        
        "2. CONTEXT BOUNDARY:\n"
        "Answer ONLY from the provided context passages. Do not use outside medical knowledge, assumptions, "
        "or external facts. State nothing else.\n\n"
        
        "3. OUTPUT FORMAT:\n"
        "Structure every response into these clear sections:\n"
        "- Recommendation / Core Findings: Direct summary answering the query.\n"
        "- Supporting Excerpts & Citations: Key excerpts or quotes from the passages with Context ID/Section.\n\n"
        
        "4. ESCAPE HATCH:\n"
        "If the answer cannot be determined from the provided context chunks, state explicitly: "
        "'The provided context does not contain sufficient information to answer this query.'"
    )

    prompt = f"Context:\n{context_str if context_str else 'No relevant context found.'}\n\nUser Question: {query}\n\nAnswer:"

    llm = get_llm(system_instruction)

    print("\n[Gemini Answer]:")
    if stream:
        for chunk in llm.generate_stream(prompt, system_instruction):
            print(chunk, end="", flush=True)
        print("\n")
    else:
        response = llm.generate(prompt, system_instruction)
        print(response)