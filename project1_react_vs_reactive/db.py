"""MongoDB access. Two collections in Project 1:
   runs  - one document per agent run (summary + metrics + grading)
   steps - one document per loop iteration (thought / action / observation)
"""
from pymongo import MongoClient
import config

_db = None


def get_db():
    global _db
    if _db is None:
        client = MongoClient(config.MONGODB_URI, serverSelectionTimeoutMS=3000)
        client.admin.command("ping")  # fail fast with a clear error if Mongo is down
        _db = client[config.MONGODB_DB]
        _db.runs.create_index("run_id", unique=True)
        _db.runs.create_index("batch_id")
        _db.steps.create_index([("run_id", 1), ("step_no", 1)])
    return _db
