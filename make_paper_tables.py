#!/usr/bin/env python3
"""
make_paper_tables.py — regenerate every results table in the IEEE Access
manuscript "Beyond Inference: Measuring and Optimizing Memory Injection
Cost in LLM Agent Workflows" directly from raw result CSVs.

Usage:
    python3 make_paper_tables.py [--results-dir results]

Every number printed by this script is computed from the node-level CSVs in
RUNS below. Per-task cost = sum of cost_total over all node rows of the
task. End-to-end accuracy = answer_correct of the deepest node of the task.
95% CIs use the t distribution (df = n_seeds - 1).

Blocks are printed in manuscript order: Tables II-VII and the Fig. 3 depth
curves. Table I is the benchmark composition (experiment design, not a
measured result) and is therefore not regenerated here.

After new experiment runs, update the filenames in RUNS.
"""
from __future__ import annotations
import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

# ── Manifest: which CSV backs which table ────────────────────────────────────
# Update filenames here after re-running experiments.
RUNS = {
    # Mid-tier (sonnet) baseline, condition A, one file per seed
    "A_mid": {
        42: "tca_results_real_20260613_014418.csv",
        7:  "tca_results_real_20260723_085114.csv",
        99: "tca_results_real_20260723_230302.csv",
    },
    # Mid-tier full system, condition H, one file per seed
    "H_mid": {
        42: "tca_results_real_20260613_175240.csv",
        7:  "tca_results_real_20260613_213817.csv",
        99: "tca_results_real_20260613_223022.csv",
    },
    # Small-tier (haiku) ablation. Conditions A-F share the June 12 run;
    # G and H are fresh full 200-task runs.
    "ablation_small": {
        "A": ("tca_results_real_20260612_200554.csv", "A"),
        "B": ("tca_results_real_20260612_200554.csv", "B"),
        "C": ("tca_results_real_20260612_200554.csv", "C"),
        "D": ("tca_results_real_20260612_200554.csv", "D"),
        "E": ("tca_results_real_20260612_200554.csv", "E"),
        "F": ("tca_results_real_20260612_200554.csv", "F"),
        "G": ("tca_results_real_20260723_111909.csv", "G"),  # fresh full 200-task run
        "H": ("tca_results_real_20260723_163118.csv", "H"),  # fresh, opus-disabled fix
    },
    # Capacity sensitivity (miss-path / eviction experiment)
    "capacity": {
        32: ("tca_results_real_20260612_200554.csv", "B"),
        2:  ("tca_results_real_20260723_172014.csv", "B"),  # warm-capacity 2
    },
}

T_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}


def load(path: Path) -> list[dict]:
    with open(path) as fh:
        return list(csv.DictReader(fh))


def tasks_of(rows: list[dict], condition: str | None = None) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if condition is None or r["condition"] == condition:
            out[r["task_id"]].append(r)
    return out


def per_task_cost(task_rows: list[dict]) -> float:
    return sum(float(r["cost_total"]) for r in task_rows)


def e2e_correct(task_rows: list[dict]) -> float:
    deepest = max(task_rows, key=lambda r: int(r["workflow_depth"]))
    return 1.0 if str(deepest["answer_correct"]).lower() in ("true", "1", "yes") else 0.0


def summarize(rows: list[dict], condition: str) -> dict:
    tasks = tasks_of(rows, condition)
    if not tasks:
        return {}
    costs = [per_task_cost(v) for v in tasks.values()]
    accs = [e2e_correct(v) for v in tasks.values()]
    comp = defaultdict(float)
    inj_tok = 0
    for v in tasks.values():
        for r in v:
            comp["inference"] += float(r["cost_inference"])
            comp["injection"] += float(r["cost_injection"])
            comp["miss"] += float(r["cost_miss_penalty"])
            comp["accum"] += float(r["cost_accum"])
            inj_tok += int(r["memory_tokens_injected"])
    n = len(tasks)
    total = sum(costs)
    return {
        "n": n,
        "cost": statistics.mean(costs),
        "acc": statistics.mean(accs),
        "mem_pct": 100.0 * comp["injection"] / total if total else 0.0,
        "components": {k: v / n for k, v in comp.items()},
        "inj_tokens_per_task": inj_tok / n,
        "fallbacks": sum(
            1 for v in tasks.values() for r in v
            if str(r["fallback_used"]).lower() in ("true", "1")
        ),
        "retrieval_ms": statistics.mean(
            float(r["retrieval_latency_ms"]) for v in tasks.values() for r in v
        ),
    }


def ci95(values: list[float]) -> tuple[float, float, float]:
    n = len(values)
    m = statistics.mean(values)
    if n < 2:
        return m, m, m
    sd = statistics.stdev(values)
    h = T_95.get(n - 1, 1.96) * sd / math.sqrt(n)
    return m, m - h, m + h


def header(title: str) -> None:
    print("=" * 72)
    print(title)
    print("=" * 72)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()
    rd = Path(args.results_dir)
    missing: list[str] = []

    # Mid-tier per-seed statistics, loaded once and reused by Tables III-VI
    # and the Fig. 3 depth curves.
    seed_stats: dict[str, dict[int, dict]] = {"A": {}, "H": {}}
    mid_rows: dict[str, dict[int, list[dict]]] = {"A": {}, "H": {}}
    for cond, key in (("A", "A_mid"), ("H", "H_mid")):
        for seed, fname in RUNS[key].items():
            if fname is None:
                missing.append(f"{key} seed {seed}")
                continue
            rows = load(rd / fname)
            mid_rows[cond][seed] = rows
            seed_stats[cond][seed] = summarize(rows, cond)

    # ── Table II: small-tier ablation ────────────────────────────────────
    header("TABLE II — per-task TCA, memory fraction, and accuracy by ablation "
           "condition\n            (small tier, seed 42, 200 tasks each)")
    abl = {}
    for cond, (fname, c) in RUNS["ablation_small"].items():
        if fname is None:
            missing.append(f"ablation {cond}")
            continue
        s = summarize(load(rd / fname), c)
        abl[cond] = s
        print(f"  {cond}  n={s['n']:3d}  cost/task ${s['cost']:.6f}  "
              f"mem {s['mem_pct']:4.1f}%  acc {s['acc']:.3f}")
    if "A" in abl and "H" in abl:
        print(f"  => small-tier cost reduction "
              f"{(1 - abl['H']['cost'] / abl['A']['cost']) * 100:.1f}%")

    # ── Table III: production-scale projection ───────────────────────────
    # Projected from the published six-decimal three-seed mean per-task costs
    # (the means printed under Table IV below), so these figures reproduce
    # Table III of the manuscript exactly. Carrying full float precision
    # instead shifts the 100,000/day row by about 0.002 percent.
    print()
    header("TABLE III — projected annual cost at enterprise query volumes "
           "(mid tier)")
    if seed_stats["A"] and seed_stats["H"]:
        a6 = round(statistics.mean(s["cost"] for s in seed_stats["A"].values()), 6)
        h6 = round(statistics.mean(s["cost"] for s in seed_stats["H"].values()), 6)
        for daily in (1_000, 10_000, 100_000):
            ay = a6 * daily * 365
            hy = h6 * daily * 365
            print(f"  {daily:>7,} tasks/day  baseline ${ay:>10,.0f}/yr  "
                  f"TCA ${hy:>9,.0f}/yr  savings ${ay - hy:>9,.0f}/yr")

    # ── Table IV: per-seed mid-tier ──────────────────────────────────────
    print()
    header("TABLE IV — per-seed cost and end-to-end accuracy, mid tier")
    for cond, key in (("A", "A_mid"), ("H", "H_mid")):
        for seed in RUNS[key]:
            s = seed_stats[cond].get(seed)
            if not s:
                continue
            print(f"  {cond}  seed {seed:>2}  n={s['n']:3d}  "
                  f"cost/task ${s['cost']:.6f}  acc {s['acc']:.3f}")
    for cond in ("A", "H"):
        st = seed_stats[cond]
        if len(st) >= 2:
            cm, cl, ch = ci95([s["cost"] for s in st.values()])
            am, al, ah = ci95([s["acc"] for s in st.values()])
            print(f"  {cond}  {len(st)}-seed mean: cost ${cm:.6f} "
                  f"[95% CI {cl:.6f}-{ch:.6f}]  acc {am:.3f} [{al:.3f}-{ah:.3f}]")
    if seed_stats["A"] and seed_stats["H"]:
        a = statistics.mean(s["cost"] for s in seed_stats["A"].values())
        h = statistics.mean(s["cost"] for s in seed_stats["H"].values())
        aa = statistics.mean(s["acc"] for s in seed_stats["A"].values())
        ha = statistics.mean(s["acc"] for s in seed_stats["H"].values())
        print(f"  => mid-tier cost reduction {(1 - h / a) * 100:.1f}%   "
              f"accuracy delta {ha - aa:+.3f}")

    # ── Table V: per-category, seed 42 ───────────────────────────────────
    print()
    header("TABLE V — per-category cost and accuracy (mid tier, seed 42, "
           "40 tasks per category)")
    for cond in ("A", "H"):
        rows = mid_rows[cond].get(42)
        if rows is None:
            continue
        bycat: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
        for r in rows:
            if r["condition"] == cond:
                bycat[r["workflow_category"]][r["task_id"]].append(r)
        for cat in sorted(bycat):
            costs = [per_task_cost(v) for v in bycat[cat].values()]
            accs = [e2e_correct(v) for v in bycat[cat].values()]
            print(f"  {cond}  {cat:<12} n={len(bycat[cat]):3d}  "
                  f"cost ${statistics.mean(costs):.4f}  acc {statistics.mean(accs):.3f}")

    # ── Table VI: TCA decomposition, seed 42 ─────────────────────────────
    print()
    header("TABLE VI — TCA cost decomposition, per task (mid tier, seed 42)")
    for cond in ("A", "H"):
        s = seed_stats[cond].get(42)
        if not s:
            continue
        c = s["components"]
        print(f"  {cond}: inference ${c['inference']:.6f}  "
              f"injection ${c['injection']:.6f}  miss ${c['miss']:.6f}  "
              f"accum ${c['accum']:.6f}  total ${s['cost']:.6f}")

    # ── Table VII: capacity sensitivity ──────────────────────────────────
    print()
    header("TABLE VII — memory capacity sensitivity (condition B, small tier, "
           "seed 42)")
    for k, (fname, c) in sorted(RUNS["capacity"].items(), reverse=True):
        if fname is None:
            missing.append(f"capacity K={k}")
            continue
        s = summarize(load(rd / fname), c)
        print(f"  K={k:<3} n={s['n']:3d}  cost/task ${s['cost']:.6f}  "
              f"acc {s['acc']:.3f}  inj tokens/task {s['inj_tokens_per_task']:.0f}  "
              f"fallback events {s['fallbacks']}  "
              f"retrieval {s['retrieval_ms']:.1f} ms")

    # ── Fig. 3 data series: depth curve, baseline A, mid tier, seed 42 ───
    print()
    header("FIG. 3 — injection share and injected tokens by depth "
           "(condition A, mid tier, seed 42)")
    rows = mid_rows["A"].get(42)
    if rows is not None:
        rows = [r for r in rows if r["condition"] == "A"]
        byd_cost = defaultdict(lambda: [0.0, 0.0])
        byd_tok = defaultdict(list)
        for r in rows:
            d = int(r["workflow_depth"])
            byd_cost[d][0] += float(r["cost_injection"])
            byd_cost[d][1] += float(r["cost_total"])
            byd_tok[d].append(int(r["memory_tokens_injected"]))
        for d in sorted(byd_cost):
            i, t = byd_cost[d]
            print(f"  depth {d}: injection {100 * i / t:5.1f}% of node cost | "
                  f"mean injected tokens {statistics.mean(byd_tok[d]):.0f}")

    if missing:
        print()
        print("!! MISSING RUNS (update RUNS manifest after running them):")
        for m in missing:
            print(f"   - {m}")
        sys.exit(1)


if __name__ == "__main__":
    main()
