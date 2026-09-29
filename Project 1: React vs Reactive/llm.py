"""One thin wrapper around OpenAI Structured Outputs. Returns the parsed Pydantic object plus
usage (tokens, cost, latency) so every step can be costed and logged."""
import os
import time
from dataclasses import dataclass

from openai import OpenAI

import config

_client = None

@dataclass
class Usage:
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_s: float
    
def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set (see .env.example)")
        _client = OpenAI()  # reads the key from the environment itself
    return _client

def structured(system,user,schema):
    t0 = time.perf_counter()
    res = _get_client().chat.completions.parse(
        model=config.MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format=schema,
        temperature=0
    )
    latency = time.perf_counter() - t0
    msg = res.choices[0].message
    if msg.parsed is None:
        raise RuntimeError(f"No parsed output (refusal={msg.refusal!r})")
    
    u = res.usage
    usage = Usage(u.prompt_tokens, u.completion_tokens,
                  config.cost_usd(config.MODEL, u.prompt_tokens, u.completion_tokens), latency)
    return msg.parsed, usage
