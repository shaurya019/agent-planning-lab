"""CLI
  python main.py tasks
  python main.py run --mode planner_executor --task order
  python main.py run --mode plan_and_execute --task discover --replan always
  python main.py run --mode planner_executor --task order --plan-only
  python main.py compare --task discover --trials 3
  python main.py show <run_id>
"""
import argparse
import uuid

import db
import pipeline
from evaluate import TASKS

MODES = ["planner_executor", "plan_and_execute"]


def cmd_tasks(_):
    for name, text in TASKS.items():
        print(f"{name:9s} {text[:110]}...")


def cmd_run(a):
    r = pipeline.run(a.mode, a.task, replan=a.replan, max_steps=a.max_steps,
                     max_replans=a.max_replans, plan_only=a.plan_only)
    print(f"\nstatus={r['status']}  run_id={r['run_id']}")
    print(f"answer: {r['answer']}")
    print(f"metrics: {r['metrics']}")
    print(f"accuracy: {r['accuracy']}")
    if r["error"]:
        print(f"error: {r['error']}")


def cmd_compare(a):
    batch = uuid.uuid4().hex[:8]
    variants = [("planner_executor", "always"), ("plan_and_execute", "on_failure"), ("plan_and_execute", "always")]
    for t in range(1, a.trials + 1):
        for mode, replan in variants:
            r = pipeline.run(mode, a.task, replan=replan, max_steps=a.max_steps, batch_id=batch, verbose=False)
            print(f"trial {t} {mode:17s} replan={replan:10s} status={r['status']:18s} passed={r['accuracy'].get('passed')}")
    rows = db.get_db().runs.aggregate([
        {"$match": {"batch_id": batch}},
        {"$group": {"_id": {"mode": "$mode", "replan": "$replan_policy"}, "runs": {"$sum": 1},
                    "passed": {"$sum": {"$cond": ["$accuracy.passed", 1, 0]}},
                    "llm_calls": {"$avg": "$metrics.llm_calls"}, "steps": {"$avg": "$metrics.steps_executed"},
                    "replans": {"$avg": "$metrics.replans"}, "tokens": {"$avg": "$metrics.total_tokens"},
                    "cost": {"$avg": "$metrics.cost_usd"}, "latency": {"$avg": "$metrics.latency_s"}}},
        {"$sort": {"_id.mode": 1, "_id.replan": 1}}])
    print(f"\nBatch {batch}  task={a.task}")
    print(f"{'mode':18}{'replan':11}{'runs':>5}{'pass':>5}{'llm':>5}{'steps':>6}{'repl':>5}{'tokens':>8}{'cost($)':>10}{'lat(s)':>8}")
    for r in rows:
        k = r["_id"]
        rp = k["replan"] if k["mode"] == "plan_and_execute" else "-"
        print(f"{k['mode']:18}{rp:11}{r['runs']:>5}{r['passed']:>5}{r['llm_calls']:>5.1f}{r['steps']:>6.1f}"
              f"{r['replans']:>5.1f}{r['tokens']:>8.0f}{r['cost']:>10.5f}{r['latency']:>8.1f}")


def cmd_show(a):
    d = db.get_db()
    run = d.runs.find_one({"run_id": a.run_id}, {"_id": 0})
    if not run:
        raise SystemExit("run not found")
    print({k: run.get(k) for k in ("mode", "task_name", "status", "final_answer", "accuracy")})
    print("metrics:", run.get("metrics"))
    for p in d.plans.find({"run_id": a.run_id}).sort("version"):
        print(f"\nPLAN v{p['version']} by {p['created_by']} ({p['reason']})")
        for s in p["plan"]["steps"]:
            print(f"  {s['id']}: {s['tool']} {({k: v for k, v in s['args'].items() if v is not None})} <- {s['depends_on']}")
    print("\nEXECUTION")
    for s in d.step_runs.find({"run_id": a.run_id}).sort("seq"):
        print(f"  #{s['seq']} v{s['plan_version']} {s['step_id']} {s['tool']:10s} {s['status']:8s} "
              f"{(s['output'] or s['error'] or '')[:80]!r}")


def main():
    p = argparse.ArgumentParser(description="Project 2: Planner-Executor vs Plan-and-Execute")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("tasks").set_defaults(fn=cmd_tasks)
    for name, fn in (("run", cmd_run), ("compare", cmd_compare)):
        sp = sub.add_parser(name)
        sp.add_argument("--task", choices=list(TASKS), default="order")
        sp.add_argument("--max-steps", type=int, default=12)
        if name == "run":
            sp.add_argument("--mode", choices=MODES, default="planner_executor")
            sp.add_argument("--replan", choices=["always", "on_failure"], default="always")
            sp.add_argument("--max-replans", type=int, default=3)
            sp.add_argument("--plan-only", action="store_true", help="make + store the plan, execute nothing")
        else:
            sp.add_argument("--trials", type=int, default=3)
        sp.set_defaults(fn=fn)
    sp = sub.add_parser("show")
    sp.add_argument("run_id")
    sp.set_defaults(fn=cmd_show)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
