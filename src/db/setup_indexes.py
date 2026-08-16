"""Print the exact Atlas Search / Vector Search index configs for the Atlas UI.

M0 clusters do not support index creation via the API, so these indexes are
created manually in the Atlas web UI (Atlas Search -> Create Search Index).
Run:  python -m src.db.setup_indexes
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)

SEARCH_INDEX = {
    "name": "default",
    "searchAnalyzer": "lucene.standard",
    "analyzer": "lucene.standard",
    "mappings": {
        "dynamic": False,
        "fields": {
            "text": {
                "type": "string",
                "analyzer": "lucene.standard",
            }
        },
    },
}

VECTOR_INDEX = {
    "name": "vector_index",
    "type": "vectorSearch",
    "fields": [
        {
            "type": "knnVector",
            "path": "embedding",
            "numDimensions": 384,
            "similarity": "cosine",
        }
    ],
}


def _print_index(title: str, config: dict) -> None:
    print(f"=== {title} ===")
    print(json.dumps(config, indent=2))
    print()


def main() -> None:
    _print_index("Atlas Search Index (BM25) - collection 'chunks'", SEARCH_INDEX)
    _print_index("Atlas Vector Search Index - collection 'chunks'", VECTOR_INDEX)


if __name__ == "__main__":
    main()
