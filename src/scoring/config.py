"""Configuration for source-confidence scoring."""

DEFAULT_TRUST_SCORE = 0.6

# Substring-of-source -> trust override. Wikipedia is a reasonably-trustworthy
# secondary source, distinct from the 0.6 default.
TRUST_OVERRIDES: dict[str, float] = {"wiki_": 0.8}

FRESHNESS_DECAY_DAYS = 365

# Start equal; tune after Phase 9 evals.
CONFIDENCE_WEIGHTS = {
    "freshness": 0.25,
    "trust": 0.25,
    "overlap": 0.25,
    "relevance": 0.25,
}

CONFIDENCE_THRESHOLD = 0.4