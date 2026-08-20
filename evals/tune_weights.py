"""LLM-free sweep over CONFIDENCE_WEIGHTS schemes (Phase 9d).

Purely offline: hybrid_search + rerank + score_and_filter are all local
(embeddings + cross-encoder + pure math). No Groq call, no cache write.

For every query it mirrors answer_query's stage defaults (hybrid_search ->
rerank) but enriches sources ONCE per query via score_and_filter(threshold=0.0,
weights=baseline), then re-evaluates the weighted sum per scheme via
score_chunk(chunk, weights=scheme) -- retrieval/rerank/source-resolution run
once, only the weighted-sum math varies.

The best-case confidence per scheme is the max score over the reranked chunks
(None when retrieval yields nothing at all). The recommended threshold per
scheme is min(genuine best-case) - 0.01. The decision rule picks the LEAST
aggressive scheme that keeps genuine_pass at 30/30 AND catches the Roman
Empire query specifically.

Run:  python evals/tune_weights.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retrieval.hybrid import hybrid_search
from src.retrieval.reranker import get_reranker
from src.scoring.confidence import (
    freshness_score,
    overlap_score,
    relevance_score,
    score_and_filter,
    score_chunk,
    trust_score,
)

MARGIN = 0.01
ROMAN_SUBSTR = "Roman Empire"

SCHEMES: dict[str, dict[str, float]] = {
    "A_baseline": {"freshness": 0.25, "trust": 0.25, "overlap": 0.25, "relevance": 0.25},
    "B_mild": {"freshness": 0.15, "trust": 0.20, "overlap": 0.20, "relevance": 0.45},
    "C_stronger": {"freshness": 0.10, "trust": 0.15, "overlap": 0.20, "relevance": 0.55},
    "D_aggressive": {"freshness": 0.05, "trust": 0.10, "overlap": 0.20, "relevance": 0.65},
}
SCHEME_ORDER = ["A_baseline", "B_mild", "C_stronger", "D_aggressive"]
SELECTION_ORDER = ["B_mild", "C_stronger", "D_aggressive"]


def _load_queries(name: str) -> list[str]:
    path = Path(__file__).resolve().parent / name
    return [
        json.loads(line)["query"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _enriched_chunks(query: str) -> list[dict]:
    """Run retrieval+rerank once and resolve sources once (threshold=0.0 keeps all)."""
    results = hybrid_search(query)
    if not results:
        return []
    reranked = get_reranker().rerank(query, results)
    if not reranked:
        return []
    return score_and_filter(reranked, threshold=0.0, weights=SCHEMES["A_baseline"])


def _best_case(chunks: list[dict], scheme: dict[str, float]) -> float | None:
    if not chunks:
        return None
    return max(score_chunk(c, weights=scheme) for c in chunks)


def _argmax_chunk(chunks: list[dict], scheme: dict[str, float]) -> dict | None:
    if not chunks:
        return None
    return max(chunks, key=lambda c: score_chunk(c, weights=scheme))


def _inputs(chunk: dict) -> dict:
    return {
        "freshness": freshness_score(chunk.get("date")),
        "trust": trust_score(chunk.get("source")),
        "overlap": overlap_score(chunk.get("retrieved_by")),
        "relevance": relevance_score(chunk.get("rerank_score")),
        "rerank_score": chunk.get("rerank_score"),
        "source": chunk.get("source"),
        "retrieved_by": chunk.get("retrieved_by"),
    }


def main() -> None:
    genuine_queries = _load_queries("eval_set.jsonl")
    adversarial_queries = _load_queries("adversarial_queries.jsonl")

    genuine_cache = [(q, _enriched_chunks(q)) for q in genuine_queries]
    adversarial_cache = [(q, _enriched_chunks(q)) for q in adversarial_queries]

    roman_index = next(
        i for i, q in enumerate(adversarial_queries) if ROMAN_SUBSTR in q
    )

    results: dict[str, dict] = {}
    for scheme_name in SCHEME_ORDER:
        scheme = SCHEMES[scheme_name]
        genuine = [(q, _best_case(chunks, scheme)) for q, chunks in genuine_cache]
        adversarial = [(q, _best_case(chunks, scheme)) for q, chunks in adversarial_cache]

        genuine_scores = [s for _, s in genuine if s is not None]
        no_context = sum(1 for _, s in genuine if s is None)
        threshold = (min(genuine_scores) - MARGIN) if genuine_scores else 0.0
        threshold = max(threshold, 0.0)

        genuine_pass = sum(1 for s in genuine_scores if s >= threshold)
        adversarial_gate = sum(1 for _, s in adversarial if s is None or s < threshold)
        roman_conf = adversarial[roman_index][1]
        roman_caught = roman_conf is None or roman_conf < threshold

        results[scheme_name] = {
            "threshold": threshold,
            "genuine_pass": genuine_pass,
            "genuine_total": len(genuine_scores),
            "no_context": no_context,
            "adversarial_gate": adversarial_gate,
            "adversarial_total": len(adversarial_queries),
            "roman_conf": roman_conf,
            "roman_caught": roman_caught,
            "genuine": genuine,
            "adversarial": adversarial,
        }

        print(f"\n=== {scheme_name} {scheme} ===")
        print(f"  threshold (min genuine - {MARGIN}) = {threshold:.4f}")
        print(f"  genuine_pass = {genuine_pass}/{len(genuine_scores)}"
              f"{'  (no-context genuine: %d)' % no_context if no_context else ''}")
        print(f"  adversarial_gate = {adversarial_gate}/{len(adversarial_queries)}")
        print(f"  Roman Empire conf = {roman_conf:.4f}  caught = {roman_caught}")

        bottom = sorted(genuine, key=lambda x: (x[1] is None, x[1] or 0.0))[:6]
        print("  lowest genuine best-case:")
        for q, s in bottom:
            display = f"{s:.4f}" if s is not None else "N/A"
            print(f"    {display:<10} {q}")
        top = sorted(adversarial, key=lambda x: x[1] if x[1] is not None else -1.0, reverse=True)[:6]
        print("  highest adversarial best-case:")
        for q, s in top:
            display = f"{s:.4f}" if s is not None else "N/A"
            print(f"    {display:<10} {q}")

    print("\n\n=== COMPARISON TABLE ===")
    print(f"  {'scheme':<14}{'threshold':<10}{'genuine_pass':<14}{'adv_gate':<12}{'Roman_conf':<12}{'Roman_caught'}")
    for scheme_name in SCHEME_ORDER:
        r = results[scheme_name]
        print(
            f"  {scheme_name:<14}{r['threshold']:<10.4f}"
            f"{r['genuine_pass']}/{r['genuine_total']:<11}"
            f"{r['adversarial_gate']}/{r['adversarial_total']:<9}"
            f"{r['roman_conf']:<12.4f}{r['roman_caught']}"
        )

    roman_chunk = _argmax_chunk(adversarial_cache[roman_index][1], SCHEMES["A_baseline"])
    print("\n=== Roman Empire chunk inputs (baseline argmax) ===")
    if roman_chunk is None:
        print("  (no retrieval)")
    else:
        inputs = _inputs(roman_chunk)
        print(f"  source={inputs['source']!r} rerank={inputs['rerank_score']:.4f} "
              f"retrieved_by={inputs['retrieved_by']}")
        print(f"  freshness={inputs['freshness']:.4f} trust={inputs['trust']:.4f} "
              f"overlap={inputs['overlap']:.4f} relevance={inputs['relevance']:.4f}")

    print("\n=== Lowest 3 genuine queries' chunk inputs (baseline argmax) ===")
    baseline = results["A_baseline"]
    lowest_3 = sorted(baseline["genuine"], key=lambda x: (x[1] is None, x[1] or 0.0))[:3]
    for q, s in lowest_3:
        index = genuine_queries.index(q)
        chunk = _argmax_chunk(genuine_cache[index][1], SCHEMES["A_baseline"])
        inputs = _inputs(chunk) if chunk else {}
        display = f"{s:.4f}" if s is not None else "N/A"
        print(f"  conf={display} | {q}")
        if chunk is not None:
            print(f"    source={inputs['source']!r} rerank={inputs['rerank_score']:.4f} "
                  f"retrieved_by={inputs['retrieved_by']}")
            print(f"    freshness={inputs['freshness']:.4f} trust={inputs['trust']:.4f} "
                  f"overlap={inputs['overlap']:.4f} relevance={inputs['relevance']:.4f}")

    selected = None
    for scheme_name in SELECTION_ORDER:
        r = results[scheme_name]
        if r["genuine_pass"] == r["genuine_total"] and r["roman_caught"]:
            selected = scheme_name
            break

    print("\n=== DECISION ===")
    if selected:
        r = results[selected]
        print(
            f"SELECT {selected}: threshold={r['threshold']:.4f}, "
            f"genuine_pass={r['genuine_pass']}/{r['genuine_total']}, "
            f"adversarial_gate={r['adversarial_gate']}/{r['adversarial_total']}, "
            f"Roman Empire caught."
        )
    else:
        print("STOP: no scheme catches the Roman Empire query while keeping 30/30 genuine.")
        print("Reweighting alone cannot fix this; a different approach (formula-level) is needed.")


if __name__ == "__main__":
    main()