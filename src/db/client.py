from typing import Any

from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection

from src.config import settings

_client: MongoClient[dict[str, Any]] | None = None


def get_client() -> MongoClient[dict[str, Any]]:
    global _client
    if _client is None:
        _client = MongoClient(settings.MONGODB_URI)
    return _client


def get_db() -> Database[dict[str, Any]]:
    return get_client()[settings.MONGODB_DB_NAME]


def get_collection(name: str) -> Collection[dict[str, Any]]:
    return get_db()[name]


if __name__ == "__main__":
    db = get_db()
    print(db.list_collection_names())
