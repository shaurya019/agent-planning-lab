"""Reactive vs ReAct: ONE loop, two policies.

              Reactive                               ReAct
  output      action only                            thought -> action
  context     task + past actions/observations       task + past thoughts/actions/observations
              (optionally only the last N: --window)
  planning    none: picks the next move from state   implicit: reasoning is written down, so the
                                                     model can carry sub-goals across steps

The loop itself is identical (this is the "observe -> decide -> act" cycle every later project
builds on):   decide (LLM) -> act (tool) -> observe (string) -> log -> repeat, until `finish`
or the step limit. The step limit is our first safety net against infinite loops.
"""
import uuid
from datetime import datetime, timezone

import config
import db
import llm
import tools
from evaluate import grade
from schemas import ReActOutput, ReactiveOutput, TOOL_DOC

REACTIVE_SYSTEM = f"""You are a REACTIVE agent. Look at the task and the history, then output ONLY the single next action. Do not explain.
{TOOL_DOC}"""

REACT_SYSTEM = f"""You are a ReAct agent. On every step: first write a short Thought (what you know, what is still missing, why the next action), then choose exactly ONE action.
{TOOL_DOC}"""


MODES = {
    "reactive": (REACTIVE_SYSTEM, ReactiveOutput),
    "react": (REACT_SYSTEM, ReActOutput),
}


def _now():
    return datetime.now(timezone.utc)


def _render_history(history: list[dict], mode: str, window: int) -> str:
    """Turn past steps into prompt text. `window`=N keeps only the last N steps (0 = keep all).
    A small window makes the reactive agent forget earlier observations -> it must re-fetch or fail.
    That is the memory/cost trade-off in miniature."""
    items = history[-window:] if window else history
    lines = []
    for h in items:
        if mode == "react":
            lines.append(f"Thought: {h['thought']}")   # the ONLY thing ReAct adds to the context
        lines.append(f"Action: {tools.describe(h['action'])}")
        lines.append(f"Observation: {h['observation']}")
    return "\n".join(lines) or "(nothing yet)"
    
def run_agent(mode: str, task: str, max_steps: int = 8, window: int = 0,
              batch_id: str | None = None, verbose: bool = True) -> dict:
    system, schema = MODES[mode]
    database = db.get_db()
    run_id = uuid.uuid4().hex[:12]
    database.runs.insert_one({
        "run_id": run_id, "batch_id": batch_id, "project": "p1_react_vs_reactive",
        "mode": mode, "task": task, "model": config.MODEL, "window": window,
        "max_steps": max_steps, "status": "running", "started_at": _now(),
    })

    history: list[dict] = []
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0, "latency_s": 0.0}
    status, final_answer, error, step_no = "max_steps_exceeded", None, None, 0
    
    try:
        for step_no in range(1,max_steps):
            prompt = (f"TASK:\n{task}\n\nHISTORY:\n{_render_history(history, mode, window)}\n\n"
                      "What is your next step?")
            out, usage = llm.structured(system, prompt, schema)   # DECIDE
            action, thought = out.action, getattr(out, "thought", None)
            
            observation = None
            if action.tool == "finish":
                if action.answer:
                    final_answer, status = action.answer, "completed"
                else:
                    observation = "ERROR: finish requires an answer"
            else:
                observation = tools.run_tool(action)
                
            database.steps.insert_one({
                "run_id": run_id, "step_no": step_no, "thought": thought,
                "action": action.model_dump(), "observation": observation,
                "usage": {"prompt_tokens": usage.prompt_tokens,
                          "completion_tokens": usage.completion_tokens},
                "cost_usd": usage.cost_usd, "latency_s": round(usage.latency_s, 3), "ts": _now(),
            })
            totals["prompt_tokens"] += usage.prompt_tokens
            totals["completion_tokens"] += usage.completion_tokens
            totals["cost_usd"] += usage.cost_usd
            totals["latency_s"] += usage.latency_s

            if verbose:
                if thought:
                    print(f"  [{step_no}] Thought: {thought}")
                print(f"  [{step_no}] Action: {tools.describe(action)}")
                if observation is not None:
                    print(f"  [{step_no}] Observation: {observation[:120].replace(chr(10), ' | ')}")

            if status == "completed":
                break
            history.append({"thought": thought, "action": action, "observation": observation})
        
    except Exception as exc:  # noqa: BLE001 - record any failure in the run document
        status, error = "error", repr(exc)
        
    accuracy = grade(final_answer)
    metrics = {
        "steps": step_no, "tool_calls": len(history),
        "prompt_tokens": totals["prompt_tokens"], "completion_tokens": totals["completion_tokens"],
        "total_tokens": totals["prompt_tokens"] + totals["completion_tokens"],
        "cost_usd": round(totals["cost_usd"], 6), "latency_s": round(totals["latency_s"], 3),
    }
    database.runs.update_one({"run_id": run_id}, {"$set": {
        "status": status, "final_answer": final_answer, "error": error,
        "metrics": metrics, "accuracy": accuracy, "finished_at": _now(),
    }})
    return {"run_id": run_id, "status": status, "final_answer": final_answer,
            "metrics": metrics, "accuracy": accuracy, "error": error}