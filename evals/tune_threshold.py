"""Data-driven sweep of the confidence gate threshold (PART A).

Pure retrieval/scoring analysis -- NO LLM calls, no cache writes. For every
query it mirrors answer_query's stage defaults (hybrid_search -> rerank ->
score_and_filter) but scores with threshold=0.0 so every chunk's confidence
is visible. The best-case confidence for a query is the highest
confidence_score among its reranked chunks (None when retrieval yields
nothing at all -- such a genuine query can never pass the gate at any
threshold).

Genuine queries (eval_set.jsonl) must PASS the gate; adversarial queries
(adversarial_queries.jsonl) must TRIP it. The recommended threshold is
min(genuine best-case) - 0.01. The recommendation is only trusted if the
honest check passes: the adversarial gate rate at the recommendation must
exceed 2/20 (otherwise the gate adds no real protection and the threshold
is left unchanged).

Run:  python evals/tune_threshold.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retrieval.hybrid import hybrid_search
from src.retrieval.reranker import get_reranker
from src.scoring.confidence import score_and_filter

SWEEP = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
ADVERSARIAL_MIN_GATED = 3  # must gate >= 3/20 (i.e. > 2/20) to trust the recommendation
MARGIN = 0.01


def _load_queries(name: str) -> list[str]:
    path = Path(__file__).resolve().parent / name
    return [
        json.loads(line)["query"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _best_case_confidence(query: str) -> float | None:
    """Return the max confidence_score answer_query could ever produce, or None."""
    results = hybrid_search(query)
    if not results:
        return None
    reranked = get_reranker().rerank(query, results)
    if not reranked:
        return None
    scored = score_and_filter(reranked, threshold=0.0)
    if not scored:
        return None
    return max(c["confidence_score"] for c in scored)


def _summary(scores: list[tuple[str, float | None]], label: str) -> None:
    print(f"\n{label} ({len(scores)} queries) -- best-case confidence:")
    for query, score in scores:
        display = f"{score:.4f}" if score is not None else "N/A (no retrieval)"
        print(f"  {display:<24} {query}")


def main() -> None:
    genuine_entries = _load_queries("eval_set.jsonl")
    adversarial_entries = _load_queries("adversarial_queries.jsonl")

    genuine: list[tuple[str, float | None]] = []
    adversarial: list[tuple[str, float | None]] = []
    for query in genuine_entries:
        genuine.append((query, _best_case_confidence(query)))
    for query in adversarial_entries:
        adversarial.append((query, _best_case_confidence(query)))

    _summary(genuine, "GENUINE")
    _summary(adversarial, "ADVERSARIAL")

    genuine_scores = [s for _, s in genuine if s is not None]
    no_context = sum(1 for _, s in genuine if s is None)
    print(
        f"\nGenuine with retrievable context: {len(genuine_scores)}/{len(genuine)}"
        f"{' (no-context, permanently gated: %d)' % no_context if no_context else ''}"
    )
    if not genuine_scores:
        print("No genuine query has retrievable context; cannot recommend a threshold.")
        return

    print(f"\nThreshold sweep (genuine_pass_rate, adversarial_gate_rate):")
    print(f"  {'threshold':<10}{'genuine_pass':<16}{'adversarial_gate':<18}")
    sweep_results: list[tuple[float, int, int]] = []
    for threshold in SWEEP:
        genuine_pass = sum(1 for s in genuine_scores if s >= threshold)
        adversarial_gate = sum(
            1 for _, s in adversarial if s is None or s < threshold
        )
        sweep_results.append((threshold, genuine_pass, adversarial_gate))
        print(
            f"  {threshold:<10}{genuine_pass}/{len(genuine_scores):<13}"
            f"{adversarial_gate}/{len(adversarial_entries):<13}"
        )

    recommended = min(genuine_scores) - MARGIN
    recommended = max(recommended, 0.0)
    gated_at_recommended = sum(
        1 for _, s in adversarial if s is None or s < recommended
    )
    print(f"\nRecommended threshold = min(genuine best-case) - {MARGIN} = {recommended:.3f}")
    print(f"At {recommended:.3f}: genuine_pass={sum(1 for s in genuine_scores if s >= recommended)}/"
          f"{len(genuine_scores)}, adversarial_gate={gated_at_recommended}/{len(adversarial_entries)}")

    if gated_at_recommended >= ADVERSARIAL_MIN_GATED:
        print(
            f"\nHONESTY CHECK PASSED: gate catches {gated_at_recommended}/{len(adversarial_entries)} "
            f"adversarial queries (>= {ADVERSARIAL_MIN_GATED}/20) at threshold {recommended:.3f}."
        )
        print(f"APPLY: set src/scoring/config.py CONFIDENCE_THRESHOLD = {recommended:.3f}")
    else:
        print(
            f"\nHONESTY CHECK FAILED: gate catches only {gated_at_recommended}/{len(adversarial_entries)} "
            f"adversarial queries (below {ADVERSARIAL_MIN_GATED}/20) at threshold {recommended:.3f}."
        )
        print("DO NOT change CONFIDENCE_THRESHOLD. The gate adds no real protection.")


if __name__ == "__main__":
    main()