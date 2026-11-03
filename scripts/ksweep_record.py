#!/usr/bin/env python3
"""K-sweep bookkeeping: file a finished run under results/ksweep/ and log it.

    # new run: move the CSV and append a manifest row
    python3 scripts/ksweep_record.py --csv results/tca_results_real_X.csv \
        --K 4 --seed 42 --elapsed 2544.5

    # existing baseline: copy (original stays where make_paper_tables.py expects it)
    python3 scripts/ksweep_record.py --csv results/tca_results_real_Y.csv \
        --K 2 --seed 42 --elapsed 2544.5 --copy

Metrics (condition B rows only):
  cost                 run total, USD
  tasks                distinct task_ids
  inj_tokens_per_task  memory_tokens_injected summed over a task's nodes, mean over tasks
  accuracy             answer_correct of each task's deepest node, mean over tasks
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections import defaultdict
from pathlib import Path

OUT = Path("results/ksweep")
MANIFEST = OUT / "manifest.csv"
FIELDS = ["K", "seed", "file", "elapsed_s", "cost", "tasks", "inj_tokens_per_task", "accuracy"]


def run_metrics(path: Path, condition: str = "B") -> dict:
    by_task: dict[str, list[dict]] = defaultdict(list)
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["condition"] == condition:
                by_task[r["task_id"]].append(r)
    if not by_task:
        sys.exit(f"{path}: no condition {condition} rows")
    rows = [r for task_rows in by_task.values() for r in task_rows]
    deepest = [max(t, key=lambda r: int(r["workflow_depth"])) for t in by_task.values()]
    n = len(by_task)
    return {
        "tasks": n,
        "cost": sum(float(r["cost_total"]) for r in rows),
        "inj_tokens_per_task": sum(int(float(r["memory_tokens_injected"])) for r in rows) / n,
        "accuracy": sum(r["answer_correct"] == "True" for r in deepest) / n,
    }


def logged(K: int, seed: int) -> bool:
    if not MANIFEST.exists():
        return False
    with open(MANIFEST, newline="") as fh:
        return any(r["K"] == str(K) and r["seed"] == str(seed) for r in csv.DictReader(fh))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--elapsed", type=float, default=None, help="wall-clock seconds, from the run's output")
    ap.add_argument("--copy", action="store_true", help="copy instead of move")
    ap.add_argument("--name", default=None, help="destination file name (default K<K>_seed<s>.csv)")
    ap.add_argument("--no-manifest", action="store_true", help="file the CSV but do not log it")
    ap.add_argument("--expect-tasks", type=int, default=200)
    args = ap.parse_args()

    m = run_metrics(args.csv)
    if m["tasks"] != args.expect_tasks:
        sys.exit(f"{args.csv}: {m['tasks']} tasks, expected {args.expect_tasks}; nothing filed")
    if not args.no_manifest and logged(args.K, args.seed):
        sys.exit(f"K={args.K} seed={args.seed} is already in {MANIFEST}; nothing filed")

    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / (args.name or f"K{args.K}_seed{args.seed}.csv")
    if dest.exists():
        sys.exit(f"{dest} already exists; nothing filed")
    (shutil.copy2 if args.copy else shutil.move)(args.csv, dest)

    if not args.no_manifest:
        new = not MANIFEST.exists()
        with open(MANIFEST, "a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(FIELDS)
            w.writerow([
                args.K, args.seed, dest.name,
                "" if args.elapsed is None else f"{args.elapsed:.1f}",
                f"{m['cost']:.4f}", m["tasks"],
                f"{m['inj_tokens_per_task']:.1f}", f"{m['accuracy']:.3f}",
            ])
    print(f"{dest}  tasks={m['tasks']}  cost=${m['cost']:.4f}  "
          f"inj/task={m['inj_tokens_per_task']:.1f}  acc={m['accuracy']:.3f}"
          + ("  (not logged)" if args.no_manifest else ""))


if __name__ == "__main__":
    main()
