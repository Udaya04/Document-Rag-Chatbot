"""Constrained generation: answers grounded ONLY in the retrieved chunks."""

from __future__ import annotations

from typing import Any

from src.generation import llm_client
from src.generation.prompt_templates import (
    SYSTEM_PROMPT,
    build_context_block,
    build_user_prompt,
)

_EMPTY_CONTEXT_ANSWER = "No context available to answer this question."


def generate_answer(query: str, chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Generate a grounded answer from the given context chunks.

    With no chunks the LLM is never called (nothing to ground on) and a fixed
    answer is returned with ``token_usage`` set to None. Otherwise the caller
    is expected to have already passed confidence-filtered chunks (e.g. the
    output of score_and_filter).
    """
    if not chunks:
        return {
            "answer": _EMPTY_CONTEXT_ANSWER,
            "context_chunks": [],
            "token_usage": None,
        }

    context_block = build_context_block(chunks)
    user_prompt = build_user_prompt(query, context_block)
    response = llm_client.generate(SYSTEM_PROMPT, user_prompt)
    return {
        "answer": response["text"],
        "context_chunks": chunks,
        "token_usage": response["usage"],
    }