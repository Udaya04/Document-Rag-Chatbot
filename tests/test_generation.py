from __future__ import annotations

import httpx
import pytest

from groq import NotFoundError, RateLimitError

from src.config import settings
from src.generation import llm_client
from src.generation.citation_mapper import extract_citations, map_citations
from src.generation.constrained_generator import generate_answer
from src.generation.llm_client import DEFAULT_MODEL, GroqAPIKeyError
from src.generation.prompt_templates import (
    SYSTEM_PROMPT,
    build_context_block,
    build_user_prompt,
)

CHUNKS = [
    {"chunk_id": "c1", "parent_doc_id": "p1", "source": "a.txt", "text": "first"},
    {"chunk_id": "c2", "parent_doc_id": "p2", "source": "b.md", "text": "second"},
    {"chunk_id": "c3", "parent_doc_id": "p3", "source": "c.pdf", "text": "third"},
]


def _groq_available() -> bool:
    return bool(settings.GROQ_API_KEY)


class TestPromptTemplates:
    def test_build_context_block_numbers_chunks(self) -> None:
        block = build_context_block(CHUNKS)
        assert block == "[1] first\n[2] second\n[3] third"

    def test_build_user_prompt_includes_query_and_context(self) -> None:
        block = build_context_block(CHUNKS)
        prompt = build_user_prompt("what is first?", block)
        assert "what is first?" in prompt
        assert "[1] first" in prompt
        assert "[3] third" in prompt

    def test_system_prompt_instructs_grounding(self) -> None:
        assert "ONLY using the provided context" in SYSTEM_PROMPT
        assert "[" in SYSTEM_PROMPT


class TestCitationMapper:
    def test_valid_markers_resolve(self) -> None:
        citations = extract_citations("A says [1]. B says [2] and [1].", CHUNKS)
        assert citations == [
            {"marker": 1, "chunk_id": "c1", "parent_doc_id": "p1", "source": "a.txt"},
            {"marker": 2, "chunk_id": "c2", "parent_doc_id": "p2", "source": "b.md"},
        ]

    def test_out_of_range_marker_dropped(self) -> None:
        citations = extract_citations("Real [2] but bogus [5].", CHUNKS)
        assert [c["marker"] for c in citations] == [2]

    def test_no_markers_returns_empty(self) -> None:
        assert extract_citations("No citations here.", CHUNKS) == []

    def test_map_citations_shape(self) -> None:
        result = map_citations("Answer [3].", CHUNKS)
        assert result["answer"] == "Answer [3]."
        assert [c["marker"] for c in result["citations"]] == [3]

    def test_line_ref_citation_corner_brackets(self) -> None:
        citations = extract_citations("Answer 【1†L1-L3】.", CHUNKS)
        assert [c["marker"] for c in citations] == [1]

    def test_multiple_line_ref_citations_extract_separately(self) -> None:
        citations = extract_citations("A 【1†L1-L3】 and B 【2†L9-L12】.", CHUNKS)
        assert [c["marker"] for c in citations] == [1, 2]

    def test_plain_bracket_styles_still_work(self) -> None:
        assert [c["marker"] for c in extract_citations("X [1] Y 【3】", CHUNKS)] == [1, 3]


class _FakeMessage:
    content = "the generated answer"


class _FakeUsage:
    prompt_tokens = 10
    completion_tokens = 5
    total_tokens = 15


class _FakeChoice:
    message = _FakeMessage()


class _FakeResponse:
    choices = [_FakeChoice()]
    usage = _FakeUsage()


class _FakeResponseNoUsage:
    choices = [_FakeChoice()]


class _FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.response = _FakeResponse()

    def create(self, **kwargs: object) -> _FakeResponse:
        self.calls.append(kwargs)
        return self.response


class _FakeChat:
    def __init__(self) -> None:
        self.completions = _FakeCompletions()


class _FakeGroq:
    def __init__(self, **kwargs: object) -> None:
        self.init_kwargs = kwargs
        self.chat = _FakeChat()


class TestLlmClient:
    def test_generate_calls_groq_and_returns_text(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "GROQ_API_KEY", "test-key")
        fake = _FakeGroq()
        captured: dict = {}
        monkeypatch.setattr(llm_client, "Groq", lambda **kw: captured.update(kw) or fake)

        result = llm_client.generate("sys", "user", model="m", temperature=0.1)

        assert result == {
            "text": "the generated answer",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        assert captured == {"api_key": "test-key"}
        (call,) = fake.chat.completions.calls
        assert call["model"] == "m"
        assert call["temperature"] == 0.1
        assert call["messages"] == [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "user"},
        ]

    def test_missing_usage_defaults_to_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(settings, "GROQ_API_KEY", "test-key")
        fake = _FakeGroq()
        fake.chat.completions.response = _FakeResponseNoUsage()
        monkeypatch.setattr(llm_client, "Groq", lambda **kw: fake)

        result = llm_client.generate("sys", "user")

        assert result["text"] == "the generated answer"
        assert result["usage"] == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def test_default_model_used(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(settings, "GROQ_API_KEY", "test-key")
        fake = _FakeGroq()
        monkeypatch.setattr(llm_client, "Groq", lambda **kw: fake)
        llm_client.generate("sys", "user")
        (call,) = fake.chat.completions.calls
        assert call["model"] == DEFAULT_MODEL
        assert call["temperature"] == 0.0

    def test_empty_key_raises_before_sdk_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(settings, "GROQ_API_KEY", "")
        monkeypatch.setattr(
            llm_client, "Groq", lambda **kw: pytest.fail("Groq should not be constructed")
        )
        with pytest.raises(GroqAPIKeyError, match="GROQ_API_KEY is not set"):
            llm_client.generate("sys", "user")


def _make_rate_limit_error(retry_after: str | None = None) -> RateLimitError:
    headers = {"retry-after": retry_after} if retry_after is not None else {}
    response = httpx.Response(
        429, request=httpx.Request("POST", "http://groq.test"), headers=headers
    )
    return RateLimitError("rate limited", response=response, body=None)


class TestLlmClientRetry:
    def test_generate_retries_rate_limit_then_succeeds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "GROQ_API_KEY", "test-key")
        monkeypatch.setattr(llm_client.time, "sleep", lambda _seconds: None)
        calls: list[int] = []

        class _FlakyCompletions:
            def create(self, **kwargs: object) -> _FakeResponse:
                calls.append(1)
                if len(calls) < 3:
                    raise _make_rate_limit_error("0.01")
                return _FakeResponse()

        class _FlakyChat:
            completions = _FlakyCompletions()

        class _FlakyGroq:
            def __init__(self, **kwargs: object) -> None:
                self.chat = _FlakyChat()

        monkeypatch.setattr(llm_client, "Groq", _FlakyGroq)

        result = llm_client.generate("sys", "user")

        assert len(calls) == 3
        assert result["text"] == "the generated answer"
        assert result["usage"]["total_tokens"] == 15

    def test_generate_propagates_last_rate_limit_after_max_attempts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "GROQ_API_KEY", "test-key")
        monkeypatch.setattr(llm_client.time, "sleep", lambda _seconds: None)
        calls: list[int] = []

        class _AlwaysLimitedCompletions:
            def create(self, **kwargs: object) -> _FakeResponse:
                calls.append(1)
                raise _make_rate_limit_error(None)

        class _AlwaysLimitedChat:
            completions = _AlwaysLimitedCompletions()

        class _AlwaysLimitedGroq:
            def __init__(self, **kwargs: object) -> None:
                self.chat = _AlwaysLimitedChat()

        monkeypatch.setattr(llm_client, "Groq", _AlwaysLimitedGroq)

        with pytest.raises(RateLimitError):
            llm_client.generate("sys", "user")
        assert len(calls) == 3

    def test_generate_does_not_retry_non_rate_limit_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "GROQ_API_KEY", "test-key")
        calls: list[int] = []

        class _BrokenCompletions:
            def create(self, **kwargs: object) -> _FakeResponse:
                calls.append(1)
                response = httpx.Response(
                    404, request=httpx.Request("POST", "http://groq.test")
                )
                raise NotFoundError("model not found", response=response, body=None)

        class _BrokenChat:
            completions = _BrokenCompletions()

        class _BrokenGroq:
            def __init__(self, **kwargs: object) -> None:
                self.chat = _BrokenChat()

        monkeypatch.setattr(llm_client, "Groq", _BrokenGroq)

        with pytest.raises(NotFoundError, match="model not found"):
            llm_client.generate("sys", "user")
        assert len(calls) == 1


class TestConstrainedGenerator:
    def test_empty_chunks_skips_llm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            llm_client, "generate", lambda *a, **k: pytest.fail("LLM must not be called")
        )
        result = generate_answer("any query", [])
        assert result == {
            "answer": "No context available to answer this question.",
            "context_chunks": [],
            "token_usage": None,
        }

    def test_non_empty_chunks_calls_llm_and_returns_shape(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[tuple] = []
        monkeypatch.setattr(
            llm_client,
            "generate",
            lambda sys_p, user_p: calls.append((sys_p, user_p))
            or {
                "text": "grounded [1]",
                "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
            },
        )

        result = generate_answer("about first", CHUNKS)

        assert result == {
            "answer": "grounded [1]",
            "context_chunks": CHUNKS,
            "token_usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
        }
        (sys_p, user_p) = calls[0]
        assert sys_p == SYSTEM_PROMPT
        assert "about first" in user_p
        assert "[1] first" in user_p


@pytest.mark.slow
class TestLiveSmoke:
    def test_generate_and_map_citations_end_to_end(self) -> None:
        if not _groq_available():
            pytest.skip("no GROQ_API_KEY available")
        from src.retrieval.hybrid import hybrid_search
        from src.retrieval.reranker import get_reranker
        from src.scoring.confidence import score_and_filter

        results = hybrid_search("fully managed cloud database", top_k=5, candidates_per_method=10)
        reranked = get_reranker().rerank("fully managed cloud database", results, top_k=5)
        scored = score_and_filter(reranked)

        generated = generate_answer("What is MongoDB Atlas?", scored)
        assert generated["answer"], "expected a non-empty answer"
        mapped = map_citations(generated["answer"], scored)
        assert mapped["citations"], "expected at least one valid citation"