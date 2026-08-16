"""Prompt templates for constrained, citation-grounded generation."""

SYSTEM_PROMPT = (
    "You are an assistant that answers questions strictly from the numbered "
    "context provided below.\n"
    "Rules:\n"
    "1. Answer ONLY using the provided context chunks. Never use outside or "
    "parametric knowledge.\n"
    "2. Cite every factual claim with the matching bracket marker, e.g. [1], "
    "[2], where [n] refers to chunk n in the context.\n"
    "3. If the context does not contain the answer, say so explicitly (for "
    "example: 'The context does not contain this information.') and do not "
    "guess or give a partial answer.\n"
    "4. Do not invent facts, names, dates, or numbers that are not present in "
    "the context."
)


def build_context_block(chunks: list[dict]) -> str:
    """Number chunks 1..N in order; the numbering is the citation-index contract."""
    return "\n".join(
        f"[{index}] {chunk.get('text', '')}" for index, chunk in enumerate(chunks, start=1)
    )


def build_user_prompt(query: str, context_block: str) -> str:
    return f"Context:\n{context_block}\n\nQuestion: {query}"