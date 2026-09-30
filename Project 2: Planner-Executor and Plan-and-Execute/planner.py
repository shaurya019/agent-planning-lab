"""The PLANNER (and REPLANNER). One LLM call turns a task into a full Plan (JSON, Pydantic-typed).

Planning principles baked into the prompt (they matter more than any code here):
  * SMALLEST plan that solves the task      -> guards against OVERPLANNING
  * every needed action is a step           -> guards against UNDERPLANNING
  * values unknown until run time are passed via {{sN}} placeholders, computed by llm_reason steps
"""
import json

import llm
from executor import PlanError, StepResult, order_steps
from schemas import Plan, ReplanDecision, Step
from tools import catalog_text

PLAN_SYSTEM = f"""You are the PLANNER of an automation agent. Produce a complete, executable plan as JSON.

TOOLS (use only these):
{catalog_text()}

RULES
1. Use the SMALLEST plan that fully solves the task (typically 4-8 steps). No decorative steps.
2. Step ids are s1, s2, s3, ... Each step has exactly one tool.
3. depends_on lists the ids whose OUTPUT the step needs. A step may only run after its dependencies.
4. You do NOT know tool outputs while planning. To use an earlier output inside an argument, write the
   placeholder {{{{sN}}}} (e.g. expression "{{{{s5}}}}") AND list sN in depends_on.
5. llm_reason gets the outputs of its depends_on steps automatically - do NOT use placeholders in its instruction.
   Use llm_reason to extract facts, build a calculator expression from data, or write the final answer/summary.
6. calculator takes ONE numeric expression only. Have an llm_reason step produce it, then pass it via a placeholder.
7. Set unused args fields to null. Only the fields the tool needs may be non-null.
8. answer_step = id of the step whose output is the final user-facing answer (usually the last llm_reason).
9. If a tool has [SIDE EFFECT], use it only when the task asks for that effect, and only once."""

REVIEW_SYSTEM = f"""You are the REPLANNER of an automation agent. A plan is being executed one step at a time.
After each step you see the task, what has run so far, and the steps still remaining. Decide:

- continue: the remaining steps are still correct and sufficient (DEFAULT - do not change a good plan)
- revise:   remaining steps are wrong/insufficient given what you now know, or a step failed and a
            different approach can work. Return the COMPLETE new list of remaining steps (new ids that do
            not reuse executed ids; they may depend on executed steps) and the answer_step.
- finish:   the task is already fully answered. Return final_answer.

TOOLS: {catalog_text()}
Follow the same planning rules: smallest plan, {{{{sN}}}} placeholders + depends_on, unused args null."""


def make_plan(task: str, meter, feedback: str | None = None) -> Plan:
    user = f"TASK:\n{task}"
    if feedback:
        user += f"\n\nYOUR PREVIOUS PLAN WAS REJECTED: {feedback}\nFix it and return a corrected full plan."
    return llm.structured(PLAN_SYSTEM, user, Plan, purpose="plan", meter=meter)


def plan_with_retry(task: str, meter, retries: int = 2) -> Plan:
    """Plan -> sanity-check -> on failure, feed the error back and re-plan (self-correcting loop)."""
    feedback = None
    for _ in range(retries + 1):
        plan = make_plan(task, meter, feedback)
        try:
            order_steps(plan.steps, set(), plan.answer_step)
            return plan
        except PlanError as exc:
            feedback = str(exc)
    raise PlanError(f"planner failed to produce an executable plan after {retries + 1} tries: {feedback}")


def review(task: str, remaining: list[Step], all_steps: dict[str, Step],
           results: dict[str, StepResult], meter) -> ReplanDecision:
    done = [{"id": sid, "tool": all_steps[sid].tool, "description": all_steps[sid].description,
             "status": r.status, "output": r.output[:500], "error": r.error}
            for sid, r in results.items()]
    user = (f"TASK:\n{task}\n\nEXECUTED SO FAR:\n{json.dumps(done, indent=1)}\n\n"
            f"REMAINING STEPS:\n{json.dumps([s.model_dump() for s in remaining], indent=1)}")
    return llm.structured(REVIEW_SYSTEM, user, ReplanDecision, purpose="replan", meter=meter)
