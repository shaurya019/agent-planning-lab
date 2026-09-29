"""Realistic-but-safe tools. Every tool returns a STRING observation; errors are returned as
'ERROR: ...' strings instead of raised, so the agent can see the failure and adapt (this is
the observation half of the loop)."""
import ast
import operator as op
from pathlib import Path

from schemas import Action

DATA_DIR = (Path(__file__).parent / "data").resolve()

# --- web_search stub: canned snippets, including a STALE one to test accuracy ----------
_SEARCH_INDEX = [
    (("widget a",), "Widget A (2026 model) - current unit price: 450.00 INR, in stock: 40"),
    (("widget a",), "Widget A (2023 model, DISCONTINUED) - old price: 380.00 INR"),
    (("widget b",), "Widget B - current unit price: 1200.00 INR, in stock: 12"),
]


def web_search(query: str) -> str:
    q = query.lower()
    hits = [text for keys, text in _SEARCH_INDEX if any(k in q for k in keys)]
    return "\n".join(hits) if hits else "No results."


# --- read_file: sandboxed to ./data (blocks ../ path traversal) -------------------------
def read_file(path: str) -> str:
    p = Path(path)
    if p.parts and p.parts[0] == "data":  # tolerate "data/order.txt"
        p = Path(*p.parts[1:])
    full = (DATA_DIR / p).resolve()
    if not full.is_relative_to(DATA_DIR):
        raise PermissionError("path escapes data directory")
    return full.read_text()


# --- calculator: safe arithmetic via AST (never use eval() on model output) -------------
_OPS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
    ast.Pow: op.pow, ast.Mod: op.mod, ast.USub: op.neg, ast.UAdd: op.pos,
}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("exponent too large")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("unsupported expression")


def calculator(expression: str) -> str:
    result = _eval(ast.parse(expression, mode="eval").body)
    if isinstance(result, float):  # avoid scientific notation like 1.23457e+06
        return f"{result:.6f}".rstrip("0").rstrip(".")
    return str(result)


def _need(value, name):
    if not value:
        raise ValueError(f"missing required argument '{name}'")
    return value


def run_tool(action: Action) -> str:
    """Dispatch one action. Never raises: failures become observations."""
    try:
        if action.tool == "web_search":
            return web_search(_need(action.query, "query"))
        if action.tool == "read_file":
            return read_file(_need(action.path, "path"))
        if action.tool == "calculator":
            return calculator(_need(action.expression, "expression"))
        return f"ERROR: unknown tool {action.tool}"
    except Exception as exc:  # noqa: BLE001 - deliberate: surface every failure to the agent
        return f"ERROR: {exc}"


def describe(action: Action) -> str:
    """Compact human/LLM-readable form of an action, used in prompts and logs."""
    args = {k: v for k, v in (("query", action.query), ("path", action.path),
                              ("expression", action.expression), ("answer", action.answer)) if v}
    return f"{action.tool}({', '.join(f'{k}={v!r}' for k, v in args.items())})"
