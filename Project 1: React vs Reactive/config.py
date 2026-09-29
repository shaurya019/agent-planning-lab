"""Central config. Secrets come ONLY from environment variables (.env is loaded for convenience)."""
import os
from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB = os.getenv("MONGODB_DB", "agent_course")

# USD per 1M tokens: (input, output). EXAMPLE VALUES - check current pricing and edit.
# Unknown models fall back to (0, 0) so cost shows as 0 rather than crashing.
PRICING = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
}


def cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    inp, out = PRICING.get(model, (0.0, 0.0))
    return (prompt_tokens * inp + completion_tokens * out) / 1_000_000
