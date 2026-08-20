"""Prompt templates for constrained, citation-grounded generation."""

INSUFFICIENT_EVIDENCE_MESSAGE = "Insufficient evidence found to answer this question."

SYSTEM_PROMPT = (
    "You are an assistant that answers questions strictly from the numbered "
    "context provided below.\n"
    "Rules:\n"
    "1. Answer ONLY using the provided context chunks. Never use outside or "
    "parametric knowledge.\n"
    "2. Cite every factual claim with the matching bracket marker, e.g. [1], "
    "[2], where [n] refers to chunk n in the context.\n"
    f"3. If the context does not contain the answer, respond with EXACTLY this "
    f"text and nothing else: \"{INSUFFICIENT_EVIDENCE_MESSAGE}\" - do not guess, "
    "paraphrase, or give a partial answer.\n"
    "4. Do not invent facts, names, dates, or numbers that are not present in "
    "the context."
)


def is_refusal(answer_text: str | None) -> bool:
    """True if ``answer_text`` is the canonical insufficient-evidence refusal.

    Matches exactly, or with only case/whitespace differences. Any OTHER
    wording a model emits as a refusal should be investigated as a
    prompt-following regression, not silently whitelisted here.
    """
    if not answer_text:
        return False
    if answer_text == INSUFFICIENT_EVIDENCE_MESSAGE:
        return True
    normalized = " ".join(str(answer_text).split())
    return normalized.casefold() == INSUFFICIENT_EVIDENCE_MESSAGE.casefold()


def build_context_block(chunks: list[dict]) -> str:
    """Number chunks 1..N in order; the numbering is the citation-index contract."""
    return "\n".join(
        f"[{index}] {chunk.get('text', '')}" for index, chunk in enumerate(chunks, start=1)
    )


def build_user_prompt(query: str, context_block: str) -> str:
    return f"Context:\n{context_block}\n\nQuestion: {query}"