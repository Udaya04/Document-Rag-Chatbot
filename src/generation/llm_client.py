"""Thin wrapper over the Groq chat-completions SDK."""

from __future__ import annotations

import logging
import time
from typing import Any

from groq import Groq, RateLimitError

from src.config import settings

DEFAULT_MODEL = "openai/gpt-oss-120b"

MAX_ATTEMPTS = 3
_BACKOFF_SECONDS = (1.0, 2.0)
_MAX_RETRY_AFTER_SECONDS = 60.0

logger = logging.getLogger(__name__)


class GroqAPIKeyError(ValueError):
    """Raised when GROQ_API_KEY is missing/empty before any SDK call."""


def _retry_after_seconds(error: RateLimitError) -> float | None:
    """Read the API's Retry-After hint from the 429 response, clamped to [0, 60]."""
    headers = getattr(error.response, "headers", None)
    if headers is None:
        return None
    if headers.get("retry-after-ms"):
        value = headers.get("retry-after-ms")
        divisor = 1000.0
    else:
        value = headers.get("retry-after")
        divisor = 1.0
    if not value:
        return None
    try:
        seconds = float(value) / divisor
    except (TypeError, ValueError):
        return None
    return min(max(seconds, 0.0), _MAX_RETRY_AFTER_SECONDS)


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
    Rate-limit (429) responses are retried up to ``MAX_ATTEMPTS`` times, waiting
    on the API's Retry-After header when present and otherwise using an
    exponential 1s/2s backoff; the last RateLimitError propagates if it still
    fails. All other SDK/network errors propagate immediately, unchanged.
    """
    if not settings.GROQ_API_KEY:
        raise GroqAPIKeyError(
            "GROQ_API_KEY is not set. Add it to .env before calling generate()."
        )

    client = Groq(api_key=settings.GROQ_API_KEY)

    response: Any = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            break
        except RateLimitError as exc:
            if attempt == MAX_ATTEMPTS - 1:
                raise
            wait = _retry_after_seconds(exc) or _BACKOFF_SECONDS[attempt]
            logger.warning(
                "Groq rate limit on attempt %d/%d; retrying in %.1fs",
                attempt + 1,
                MAX_ATTEMPTS,
                wait,
            )
            time.sleep(wait)

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