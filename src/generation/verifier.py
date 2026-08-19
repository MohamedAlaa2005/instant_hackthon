"""
Post-Generation Verification Module.

Audits candidate responses against retrieved context for:
1. Factual grounding & hallucination prevention.
2. Citation validity (verifying chunk IDs, headings, and sources).
3. Anti-personalization compliance (objective clinical tone vs personalized advice).
4. Deterministic Certainty scoring derived directly from Cohere reranker cross-encoder scores.
"""

from typing import Optional, List
from pydantic import BaseModel, Field

from src.config import LLM_MODEL
from src.generation.llm import Gemini


class CitationCheck(BaseModel):
    cited_id: str = Field(description="The ID of the chunk cited in the answer")
    is_valid: bool = Field(description="True if this chunk ID exists in context and actually contains the cited facts")
    issue: Optional[str] = Field(default=None, description="Explanation if citation is invalid or inaccurate")


class VerificationResult(BaseModel):
    is_grounded: bool = Field(
        description="True if ALL medical claims in the answer and evidence are explicitly supported by the retrieved chunks without hallucination."
    )
    citations_valid: bool = Field(
        description="True if all citations accurately point to retrieved chunks containing the relevant facts."
    )
    no_personalization: bool = Field(
        description="True if the response maintains an impersonal, objective clinical tone without second-person advice (e.g. no 'you should', 'for you')."
    )
    rerank_certainty: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Certainty score derived directly from top Cohere cross-encoder rerank relevance."
    )
    audit_notes: str = Field(
        default="",
        description="Brief summary from the auditor on evidence coverage and grounding."
    )
    citations_audit: List[CitationCheck] = Field(
        default_factory=list,
        description="Audit breakdown of each cited chunk."
    )
    flagged_issues: List[str] = Field(
        default_factory=list,
        description="List of any identified grounding, citation, or personalization violations."
    )
    verdict: str = Field(
        description="'PASSED' if grounded and citations valid; 'FAILED' if hallucinated, ungrounded, or severe violations found."
    )


VERIFIER_SYSTEM_INSTRUCTION = (
    "You are a strict Medical RAG Verification Auditor.\n"
    "Your SOLE purpose is to evaluate and audit a candidate answer against retrieved context.\n"
    "DO NOT generate new answers or new medical knowledge.\n\n"
    "AUDIT CRITERIA:\n"
    "1. GROUNDING & FAITHFULNESS:\n"
    "   - Every claim in the candidate Answer and Evidence must be strictly traceable to the provided context.\n"
    "   - If any fact, medication, statistic, or guideline is not in the context, mark is_grounded = false.\n\n"
    "2. CITATION ACCURACY:\n"
    "   - Check each cited chunk ID. Does that chunk exist in the context? Does it actually support the claim?\n"
    "   - If the citation is fabricated or mismatched, mark citations_valid = false.\n\n"
    "3. ANTI-PERSONALIZATION & TONE:\n"
    "   - The answer must be objective and generalized (e.g., 'Guidelines recommend...', 'Studies indicate...').\n"
    "   - The answer must NOT give direct personal patient instructions (e.g., 'You should take...', 'In your case', 'I recommend you consult...').\n"
    "   - If personalized medical advice is present, mark no_personalization = false.\n\n"
    "4. VERDICT:\n"
    "   - Set verdict = 'PASSED' only if is_grounded is true and citations_valid is true.\n"
    "   - Otherwise set verdict = 'FAILED'."
)


def verify_generation(
    query: str,
    retrieved_chunks: list[dict],
    candidate_output: str,
    llm_model: str = LLM_MODEL,
) -> VerificationResult:
    """
    Run post-generation verification on candidate LLM output and calculate
    deterministic certainty scores from the Cohere reranker.
    """
    # 1. Calculate top rerank score from retrieved chunks
    rerank_scores = [
        float(c.get("rerank_score", c.get("score", 0.0)))
        for c in retrieved_chunks
        if "rerank_score" in c or "score" in c
    ]
    max_rerank = max(rerank_scores, default=0.0)

    # 2. Build context string for verifier
    context_blocks = []
    for i, c in enumerate(retrieved_chunks, 1):
        context_blocks.append(
            f"--- Chunk [{i}] (Rerank Score: {c.get('rerank_score', c.get('score', 0.0)):.3f}) ---\n"
            f"ID: {c.get('id', 'N/A')}\n"
            f"Topic: {c.get('topic', 'N/A')}\n"
            f"Section: {c.get('section', 'N/A')}\n"
            f"Heading: {c.get('heading', 'N/A')}\n"
            f"Source: {c.get('url', 'N/A')} ({c.get('source', 'N/A')})\n"
            f"Section Path: {c.get('section_path', 'N/A')}\n"
            f"Content:\n{c.get('text', '')}"
        )
    context_str = "\n\n".join(context_blocks)

    verification_prompt = (
        f"USER QUERY:\n{query}\n\n"
        f"RETRIEVED CONTEXT CHUNKS:\n{context_str if context_str else 'No context chunks provided.'}\n\n"
        f"CANDIDATE OUTPUT TO VERIFY:\n{candidate_output}\n\n"
        f"Perform strict audit and output JSON conforming to the schema."
    )

    verifier_llm = Gemini(model=llm_model, system_instruction=VERIFIER_SYSTEM_INSTRUCTION)
    result = verifier_llm.generate_structured(
        prompt=verification_prompt,
        response_schema=VerificationResult,
        system_instruction=VERIFIER_SYSTEM_INSTRUCTION,
        temperature=0.0,
    )

    # 3. Deterministically assign certainty based on reranker and audit verdict
    if result.verdict == "PASSED" and result.is_grounded and result.citations_valid:
        result.rerank_certainty = round(max_rerank, 3)
    else:
        result.rerank_certainty = 0.0

    return result
