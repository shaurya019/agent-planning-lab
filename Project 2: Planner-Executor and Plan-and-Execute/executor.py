"""The EXECUTOR: no LLM decisions here. It only (1) orders steps by dependency, (2) fills in
{{sN}} placeholders with earlier outputs, (3) validates args, (4) calls the tool, (5) logs.

That separation is the point of Planner-Executor: thinking happens in the planner, doing happens
here. (Project 4 turns `check_plan` into a full Plan Validator with risk/cost scoring.)
"""
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from pydantic import ValidationError

from schemas import Step
from tools import TOOLS, ToolContext, ToolError

PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z0-9_]+)\s*\}\}")


class PlanError(Exception):
    """The plan is not executable. Message is fed back to the planner on retry."""


@dataclass
class StepResult:
    step_id: str
    status: str            # ok | failed | skipped
    output: str = ""
    error: str | None = None


def _refs(step: Step) -> set[str]:
    return {m for v in step.args.non_null().values() for m in PLACEHOLDER.findall(v)}


def check_args(step: Step):
    """Does the step supply exactly the arguments its tool needs? (Pydantic layer 2)"""
    tool = TOOLS.get(step.tool)
    if tool is None:
        raise PlanError(f"{step.id}: unknown tool {step.tool!r}")
    try:
        return tool.args_model(**step.args.non_null())
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(map(str, e['loc'])) or 'args'}: {e['msg']}" for e in exc.errors())
        raise PlanError(f"{step.id} ({step.tool}) has invalid args: {problems}")


def order_steps(steps: list[Step], known_ids: set[str], answer_step: str) -> list[Step]:
    """Validate + topologically sort (Kahn's algorithm, stable w.r.t. the planner's order).
    `known_ids` = steps already executed in earlier plan versions (their outputs exist)."""
    ids = [s.id for s in steps]
    if len(set(ids)) != len(ids):
        raise PlanError("duplicate step ids in plan")
    clash = set(ids) & known_ids
    if clash:
        raise PlanError(f"new step ids clash with already-executed steps: {sorted(clash)}")
    if answer_step not in set(ids) | known_ids:
        raise PlanError(f"answer_step {answer_step!r} is not a step in the plan")
    valid = set(ids) | known_ids
    for s in steps:
        missing = [d for d in s.depends_on if d not in valid]
        if missing:
            raise PlanError(f"{s.id} depends on unknown steps {missing}")
        undeclared = _refs(s) - set(s.depends_on)
        if undeclared:
            raise PlanError(f"{s.id} uses {{{{{sorted(undeclared)[0]}}}}} but does not list it in depends_on")
        check_args(s)

    ordered, done, pending = [], set(known_ids), list(steps)
    while pending:
        ready = [s for s in pending if all(d in done for d in s.depends_on)]
        if not ready:
            raise PlanError(f"dependency cycle among steps {[s.id for s in pending]}")
        nxt = ready[0]
        ordered.append(nxt)
        done.add(nxt.id)
        pending.remove(nxt)
    return ordered


def _fill(text: str, outputs: dict[str, str]) -> str:
    return PLACEHOLDER.sub(lambda m: outputs.get(m.group(1), m.group(0)), text)


def execute_step(step: Step, results: dict[str, StepResult], ctx: ToolContext,
                 seq: int, plan_version: int) -> StepResult:
    """Run one step and log it. Never raises: failure is data (status='failed')."""
    t0 = time.perf_counter()
    bad_deps = [d for d in step.depends_on if d not in results or results[d].status != "ok"]
    resolved, res = None, None
    if bad_deps:
        res = StepResult(step.id, "skipped", error=f"dependencies not ok: {bad_deps}")
    else:
        outputs = {d: results[d].output for d in step.depends_on}
        ctx.dep_outputs = outputs
        try:
            # PLACEHOLDER FILLING: the planner wrote "{{s2}}" because it could not know the value
            # at planning time; now that s2 has run, substitute its real output.
            resolved = {k: _fill(v, outputs) for k, v in step.args.non_null().items()}
            args = TOOLS[step.tool].args_model(**resolved)    # re-validate the concrete values
            out = TOOLS[step.tool].fn(args, ctx)
            res = StepResult(step.id, "ok", output=str(out))
        except (ToolError, ValidationError) as exc:
            res = StepResult(step.id, "failed", error=str(exc))
        except Exception as exc:  # noqa: BLE001 - any tool crash becomes a failed step
            res = StepResult(step.id, "failed", error=f"{type(exc).__name__}: {exc}")

    ctx.db.step_runs.insert_one({
        "run_id": ctx.run_id, "seq": seq, "plan_version": plan_version, "step_id": step.id,
        "tool": step.tool, "description": step.description, "depends_on": step.depends_on,
        "side_effect": TOOLS[step.tool].side_effect, "resolved_args": resolved,
        "status": res.status, "output": res.output, "error": res.error,
        "latency_s": round(time.perf_counter() - t0, 3), "ts": datetime.now(timezone.utc)})
    return res
