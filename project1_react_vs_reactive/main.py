"""CLI:
  python main.py run --mode react
  python main.py run --mode reactive --window 1
  python main.py compare --trials 3
  python main.py show <run_id>
"""
import argparse
import uuid

import db
from agents import run_agent
from evaluate import DEFAULT_TASK


def cmd_run(args):
    print(f"Running {args.mode} agent...")
    r = run_agent(args.mode, args.task, args.max_steps, args.window)
    print(f"\nstatus={r['status']}  run_id={r['run_id']}")
    print(f"answer: {r['final_answer']}")
    print(f"metrics: {r['metrics']}")
    print(f"accuracy: {r['accuracy']}")
    if r["error"]:
        print(f"error: {r['error']}")


def cmd_compare(args):
    batch = uuid.uuid4().hex[:8]
    for t in range(1, args.trials + 1):
        for mode in ("reactive", "react"):
            r = run_agent(mode, args.task, args.max_steps, args.window, batch_id=batch, verbose=False)
            print(f"trial {t} {mode:9s} status={r['status']:18s} passed={r['accuracy']['passed']} "
                  f"steps={r['metrics']['steps']}")
    # Aggregation done by MongoDB, not Python: this is the payoff of logging structured runs.
    rows = db.get_db().runs.aggregate([
        {"$match": {"batch_id": batch}},
        {"$group": {"_id": "$mode", "runs": {"$sum": 1},
                    "passed": {"$sum": {"$cond": ["$accuracy.passed", 1, 0]}},
                    "avg_steps": {"$avg": "$metrics.steps"},
                    "avg_tokens": {"$avg": "$metrics.total_tokens"},
                    "avg_cost": {"$avg": "$metrics.cost_usd"},
                    "avg_latency": {"$avg": "$metrics.latency_s"}}},
        {"$sort": {"_id": 1}},
    ])
    print(f"\nBatch {batch}")
    print(f"{'mode':10}{'runs':>5}{'passed':>8}{'steps':>7}{'tokens':>9}{'cost($)':>10}{'latency(s)':>12}")
    for r in rows:
        print(f"{r['_id']:10}{r['runs']:>5}{r['passed']:>8}{r['avg_steps']:>7.1f}"
              f"{r['avg_tokens']:>9.0f}{r['avg_cost']:>10.5f}{r['avg_latency']:>12.2f}")


def cmd_show(args):
    d = db.get_db()
    run = d.runs.find_one({"run_id": args.run_id}, {"_id": 0})
    if not run:
        raise SystemExit("run not found")
    print({k: run[k] for k in ("mode", "status", "final_answer", "metrics", "accuracy") if k in run})
    for s in d.steps.find({"run_id": args.run_id}).sort("step_no"):
        print(f"\n#{s['step_no']} thought={s.get('thought')}\n   action={s['action']}\n   obs={s['observation']}")


def main():
    p = argparse.ArgumentParser(description="Project 1: Reactive vs ReAct")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("run", cmd_run), ("compare", cmd_compare)):
        sp = sub.add_parser(name)
        sp.add_argument("--task", default=DEFAULT_TASK)
        sp.add_argument("--max-steps", type=int, default=8)
        sp.add_argument("--window", type=int, default=0, help="history window for prompts; 0 = full")
        if name == "run":
            sp.add_argument("--mode", choices=["reactive", "react"], default="react")
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
