from __future__ import annotations

import pytest

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