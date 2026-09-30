"""Tool registry. Each tool = Pydantic args model + function + metadata.

The registry does double duty:
  1. the executor dispatches through it
  2. `catalog_text()` renders it into the PLANNER's prompt, so the planner only plans with tools
     that exist. `side_effect=True` marks tools that change the world (used for gating in Project 6).

Tools return strings and raise ToolError on failure; the executor turns that into a failed step.
"""
import ast
import json
import operator as op
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import llm
from schemas import (CalculatorArgs, DbQueryArgs, LlmReasonArgs, ReadFileArgs, SendEmailArgs,
                     WebSearchArgs, WriteFileArgs)

DATA_DIR = (Path(__file__).parent / "data").resolve()
OUT_DIR = DATA_DIR / "out"
ALLOWED_EMAIL_DOMAINS = {"example.com"} 

class ToolError(Exception):
    pass

@dataclass
class ToolContext:
    db: object
    run_id: str
    meter: llm.Meter
    dep_outputs: dict = field(default_factory=dict)   # outputs of the current step's dependencies


# ---------------- web_search (stub with a stale result to catch sloppy agents) ----------------
_SEARCH_INDEX = [
    (("widget a",), "Widget A (2026 model) - current unit price: 450.00 INR, in stock: 40"),
    (("widget a",), "Widget A (2023 model, DISCONTINUED) - old price: 380.00 INR"),
    (("widget b",), "Widget B - current unit price: 1200.00 INR, in stock: 12"),
]


def _web_search(a: WebSearchArgs, ctx) -> str:
    q = a.query.lower()
    hits = [t for keys, t in _SEARCH_INDEX if any(k in q for k in keys)]
    return "\n".join(hits) if hits else "No results."


# ---------------- files: sandboxed to ./data (read) and ./data/out (write) ----------------
def _safe(path: str, root: Path) -> Path:
    p = Path(path)
    if p.parts and p.parts[0] == "data":
        p = Path(*p.parts[1:])
    full = (root / p).resolve()
    if not full.is_relative_to(root):
        raise ToolError("path escapes the sandbox directory")
    return full


def _read_file(a: ReadFileArgs, ctx) -> str:
    try:
        return _safe(a.path, DATA_DIR).read_text()
    except FileNotFoundError:
        raise ToolError(f"file not found: {a.path}")


def _write_file(a: WriteFileArgs, ctx) -> str:          # SIDE EFFECT
    if len(a.content) > 10_000:
        raise ToolError("content too large (max 10,000 chars)")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    target = _safe(a.path, OUT_DIR)
    target.write_text(a.content)
    return f"wrote {len(a.content)} chars to data/out/{target.name}"


# ---------------- calculator: AST-based, never eval() model output ----------------
_OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
        ast.Pow: op.pow, ast.Mod: op.mod, ast.USub: op.neg, ast.UAdd: op.pos}


def _eval(n):
    if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
        return n.value
    if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
        l, r = _eval(n.left), _eval(n.right)
        if isinstance(n.op, ast.Pow) and abs(r) > 100:
            raise ToolError("exponent too large")
        return _OPS[type(n.op)](l, r)
    if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
        return _OPS[type(n.op)](_eval(n.operand))
    raise ToolError("unsupported expression (numbers and + - * / ** % only)")


def _calculator(a: CalculatorArgs, ctx) -> str:
    try:
        result = _eval(ast.parse(a.expression.strip(), mode="eval").body)
    except (SyntaxError, ZeroDivisionError) as exc:
        raise ToolError(f"bad expression {a.expression!r}: {exc}")
    return f"{result:.6f}".rstrip("0").rstrip(".") if isinstance(result, float) else str(result)


# ---------------- db_query: read-only, allow-listed collections and fields ----------------
def _db_query(a: DbQueryArgs, ctx) -> str:
    docs = list(ctx.db[a.collection].find({a.field: a.value}, {"_id": 0}).limit(5))
    return json.dumps(docs) if docs else "[]"


# ---------------- send_email: STUB with a real side effect (a row in `emails`) ----------------
def _send_email(a: SendEmailArgs, ctx) -> str:          # SIDE EFFECT (Project 6 adds rollback)
    domain = a.to.rsplit("@", 1)[-1].lower() if "@" in a.to else ""
    if domain not in ALLOWED_EMAIL_DOMAINS:
        raise ToolError(f"recipient domain {domain!r} not in allow-list {sorted(ALLOWED_EMAIL_DOMAINS)}")
    res = ctx.db.emails.insert_one({"run_id": ctx.run_id, "to": a.to, "subject": a.subject,
                                    "body": a.body, "status": "sent",
                                    "sent_at": datetime.now(timezone.utc)})
    return f"email sent to {a.to} (id={res.inserted_id})"



_REASON_SYSTEM = ("You execute ONE step of a larger plan. Use ONLY the provided INPUTS (outputs of earlier "
                  "steps). Do exactly what the INSTRUCTION says and output ONLY that result - no preamble. "
                  "If asked for an arithmetic expression, output just the expression.")


def _llm_reason(a: LlmReasonArgs, ctx) -> str:
    inputs = "\n\n".join(f"[{k}]\n{v}" for k, v in ctx.dep_outputs.items()) or "(none)"
    return llm.text(_REASON_SYSTEM, f"INPUTS:\n{inputs}\n\nINSTRUCTION:\n{a.instruction}",
                    purpose="llm_reason", meter=ctx.meter).strip()

@dataclass
class Tool:
    name: str
    args_model: type
    fn: Callable
    signature: str
    description: str
    side_effect: bool = False
    
TOOLS: dict[str,Tool] = {t.name: t for t in [
    Tool("web_search", WebSearchArgs, _web_search, "web_search(query)",
         "Search the web for facts such as prices. May return stale or discontinued results."),
    Tool("read_file", ReadFileArgs, _read_file, "read_file(path)",
         "Read a text file, e.g. 'order.txt'."),
    Tool("write_file", WriteFileArgs, _write_file, "write_file(path, content)",
         "Write a text file (saved under data/out/).", side_effect=True),
    Tool("calculator", CalculatorArgs, _calculator, "calculator(expression)",
         "Evaluate ONE arithmetic expression with numbers only, e.g. '(3*450+5*1200)*1.18'."),
    Tool("db_query", DbQueryArgs, _db_query, "db_query(collection, field, value)",
         "Exact-match lookup. collection: products|orders; field: name|sku|order_id. Returns JSON list."),
    Tool("send_email", SendEmailArgs, _send_email, "send_email(to, subject, body)",
         "Send an email (stub). Only @example.com recipients are allowed.", side_effect=True),
    Tool("llm_reason", LlmReasonArgs, _llm_reason, "llm_reason(instruction)",
         "Ask the LLM to extract, compute an expression, or write text FROM the outputs of the steps "
         "listed in depends_on. Use it whenever a step needs a value that is only known at run time."),
]}

def catalog_text() -> str:
    return "\n".join(f"- {t.signature}{' [SIDE EFFECT]' if t.side_effect else ''}: {t.description}"
                     for t in TOOLS.values())