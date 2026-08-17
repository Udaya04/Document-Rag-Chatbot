"""Thin wrapper over the Groq chat-completions SDK."""

from __future__ import annotations

from typing import Any

from groq import Groq

from src.config import settings

DEFAULT_MODEL = "openai/gpt-oss-120b"


class GroqAPIKeyError(ValueError):
    """Raised when GROQ_API_KEY is missing/empty before any SDK call."""


def generate(
    system_prompt: str,
    user_prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Call Groq chat completions and return the assistant text plus usage.

    Returns ``{"text": str, "usage": {"prompt_tokens": int,
    "completion_tokens": int, "total_tokens": int}}``. Token counts default to
    0 if ``response.usage`` (or any token field) is unexpectedly absent.

    Raises GroqAPIKeyError immediately if settings.GROQ_API_KEY is empty.
    Genuine SDK errors (rate limit, network) propagate as-is.
    """
    if not settings.GROQ_API_KEY:
        raise GroqAPIKeyError(
            "GROQ_API_KEY is not set. Add it to .env before calling generate()."
        )

    client = Groq(api_key=settings.GROQ_API_KEY)
    response: Any = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    usage = getattr(response, "usage", None)
    if usage is None:
        token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    else:
        token_usage = {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
            "total_tokens": getattr(usage, "total_tokens", 0) or 0,
        }
    return {
        "text": response.choices[0].message.content,
        "usage": token_usage,
    }