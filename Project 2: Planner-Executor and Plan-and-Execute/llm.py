"""OpenAI wrapper + a Meter that records every call (tokens, cost, latency) in MongoDB.

Two call types:
  structured(...) -> Pydantic object (plans, replan decisions) via Structured Outputs
  text(...)       -> plain string (used by the llm_reason tool)
`purpose` tags each call so we can answer: "how much of the bill was planning vs executing?"
"""
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from openai import OpenAI

import config

_client = None


@dataclass
class Usage:
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_s: float
    
class Meter:
    def __init__(self, database, run_id: str):
        self.db, self.run_id = database, run_id
        self.by_purpose: dict[str, dict] = {}

    def add(self, purpose: str, u: Usage):
        b = self.by_purpose.setdefault(purpose, {"calls": 0, "prompt_tokens": 0,
                                                 "completion_tokens": 0, "cost_usd": 0.0, "latency_s": 0.0})
        b["calls"] += 1
        b["prompt_tokens"] += u.prompt_tokens
        b["completion_tokens"] += u.completion_tokens
        b["cost_usd"] += u.cost_usd
        b["latency_s"] += u.latency_s
        self.db.llm_calls.insert_one({
            "run_id": self.run_id, "purpose": purpose, "model": config.MODEL,
            "prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens,
            "cost_usd": u.cost_usd, "latency_s": round(u.latency_s, 3),
            "ts": datetime.now(timezone.utc)})

    def totals(self) -> dict:
        t = {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0, "latency_s": 0.0}
        for b in self.by_purpose.values():
            t["llm_calls"] += b["calls"]
            for k in ("prompt_tokens", "completion_tokens", "cost_usd", "latency_s"):
                t[k] += b[k]
        t["total_tokens"] = t["prompt_tokens"] + t["completion_tokens"]
        t["cost_usd"], t["latency_s"] = round(t["cost_usd"], 6), round(t["latency_s"], 3)
        return t
    

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set (see .env.example)")
        _client = OpenAI()  # reads the key from the environment itself
    return _client
        
def _usage(resp, t0) -> Usage:
    u = resp.usage
    return Usage(u.prompt_tokens, u.completion_tokens,
                 config.cost_usd(config.MODEL, u.prompt_tokens, u.completion_tokens),
                 time.perf_counter() - t0)
    
def structured(system: str, user: str, schema, *, purpose: str, meter: Meter):
    t0 = time.perf_counter()
    resp = _get_client().chat.completions.parse(
        model=config.MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format=schema,   # Pydantic class -> strict JSON schema
        temperature=0,            # reasoning models reject this; remove the line if you switch to one
    )
    msg = resp.choices[0].message
    if msg.parsed is None:
        raise RuntimeError(f"No parsed output (refusal={msg.refusal!r})")
    meter.add(purpose, _usage(resp, t0))
    return msg.parsed

   
def text(system: str, user: str, *, purpose: str, meter: Meter) -> str:
    t0 = time.perf_counter()
    resp = _get_client().chat.completions.create(
        model=config.MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0,
    )
    meter.add(purpose, _usage(resp, t0))
    return resp.choices[0].message.content or ""