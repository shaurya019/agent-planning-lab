"""MongoDB access + seed data.

Collections written by the AGENT (the audit trail):
  runs       one doc per run: mode, status, metrics, accuracy
  plans      every plan VERSION (v1 = planner, v2+ = replanner)
  step_runs  one doc per executed/skipped step
  llm_calls  one doc per LLM call (purpose = plan | replan | llm_reason) -> cost breakdown
Collections used by TOOLS (the "world" the agent acts on):
  products   read by db_query          emails   written by send_email (the side effect)
  orders     read by db_query
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
        prepare(_db)
    return _db


def prepare(database):
    database.runs.create_index("run_id", unique=True)
    database.runs.create_index("batch_id")
    database.plans.create_index([("run_id", 1), ("version", 1)], unique=True)
    database.step_runs.create_index([("run_id", 1), ("seq", 1)])
    database.llm_calls.create_index("run_id")
    seed(database)


def seed(database):
    """Idempotent demo data for the db_query tool."""
    database.products.update_one({"name": "Widget A"}, {"$set": {
        "name": "Widget A", "sku": "WA-2026", "price": 450.0, "stock": 40, "active": True}}, upsert=True)
    database.products.update_one({"name": "Widget B"}, {"$set": {
        "name": "Widget B", "sku": "WB-2026", "price": 1200.0, "stock": 12, "active": True}}, upsert=True)
    database.products.update_one({"name": "Widget A (2023)"}, {"$set": {
        "name": "Widget A (2023)", "sku": "WA-2023", "price": 380.0, "stock": 0, "active": False}}, upsert=True)
    database.orders.update_one({"order_id": "1042"}, {"$set": {
        "order_id": "1042", "customer": "ACME Ltd", "status": "pending"}}, upsert=True)
