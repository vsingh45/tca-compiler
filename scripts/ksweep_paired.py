#!/usr/bin/env python3
"""K-sweep paired per-task accuracy tests: results/ksweep/K<K>_seed<s>.csv -> paired_tests.md.

    python3 scripts/ksweep_paired.py            # writes results/ksweep/paired_tests.md and prints it

Standard library only. A task counts as correct when answer_correct is true on its deepest node
(condition B), the definition behind manifest.csv; every run covers the same 200 tasks, for K in 1..5
at seeds 7, 42 and 99.
  (a) McNemar within each seed, K=1 vs K=5 and K=2 vs K=5 on the 200 paired tasks: discordant counts,
      exact two-sided binomial p, and the continuity-corrected chi-square p.
  (b) Pooled over the three seeds (600 task-seed pairs per K): accuracy with a Wilson 95% CI (pairs
      treated as independent) and a task-clustered bootstrap 95% CI (the 200 tasks resampled, each
      carrying its three seeds). One-way comparison across K: Cochran's Q over the 600 task-seed
      blocks, Pearson chi-square homogeneity on the 5x2 table, and an exact run-level randomization
      test (the K labels permuted among the five runs of each seed).
Counts, intervals and p-values only; nothing is interpreted. The output is deterministic.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import math
import random
import sys
from pathlib import Path

KS = (1, 2, 3, 4, 5)
SEEDS = (7, 42, 99)
REF = 5                  # reference capacity of the paired tests
CONTRASTS = (1, 2)       # each compared with REF within every seed
N_TASKS = 200
Z95 = 1.959963984540054


# ------------------------------------------------------------------------------------------ data
def load_run(path: Path) -> dict[str, bool]:
    """task_id -> answer_correct of the task's deepest node, condition B rows (same rule as ksweep_record.py)."""
    deepest: dict[str, tuple[int, bool]] = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["condition"] != "B":
                continue
            depth = int(r["workflow_depth"])
            if r["task_id"] not in deepest or depth > deepest[r["task_id"]][0]:
                deepest[r["task_id"]] = (depth, r["answer_correct"] == "True")
    return {t: ok for t, (_, ok) in deepest.items()}


def load_runs(d: Path) -> dict[tuple[int, int], list[bool]]:
    """{(K, seed): correctness vector over the shared, sorted task ids}; exits on a missing file or task-set mismatch."""
    ids: list[str] | None = None
    runs: dict[tuple[int, int], list[bool]] = {}
    for K in KS:
        for s in SEEDS:
            path = d / f"K{K}_seed{s}.csv"
            if not path.exists():
                sys.exit(f"{path}: missing")
            run = load_run(path)
            if len(run) != N_TASKS:
                sys.exit(f"{path}: {len(run)} tasks, expected {N_TASKS}")
            if ids is None:
                ids = sorted(run)
            elif sorted(run) != ids:
                sys.exit(f"{path}: task ids differ from K={KS[0]} seed={SEEDS[0]}")
            runs[(K, s)] = [run[t] for t in ids]
    return runs


def check_against_manifest(d: Path, runs: dict[tuple[int, int], list[bool]]) -> None:
    """Exit if a run's accuracy here differs from manifest.csv (guards the deepest-node definition)."""
    with open(d / "manifest.csv", newline="") as fh:
        logged = {(int(r["K"]), int(r["seed"])): float(r["accuracy"]) for r in csv.DictReader(fh)}
    for key, v in runs.items():
        acc = sum(v) / len(v)
        if key not in logged or abs(logged[key] - acc) > 5e-4:
            sys.exit(f"K={key[0]} seed={key[1]}: accuracy {acc:.4f} here vs manifest {logged.get(key)}")


# ------------------------------------------------------------------------------------------ tests
def chi2_sf(x: float, df: int) -> float:
    """Upper tail of the chi-square distribution for df = 1 or even df (closed forms, no scipy)."""
    if df == 1:
        return math.erfc(math.sqrt(x / 2))
    if df % 2 == 0:
        h = x / 2
        return math.exp(-h) * sum(h ** i / math.factorial(i) for i in range(df // 2))
    raise ValueError(f"df={df} not supported")


def mcnemar(x: list[bool], y: list[bool]) -> dict:
    """Paired 2x2 of outcomes x (lower K) and y (K=5) on the same tasks."""
    both = sum(a and b for a, b in zip(x, y))
    only_x = sum(a and not b for a, b in zip(x, y))
    only_y = sum(b and not a for a, b in zip(x, y))
    n = only_x + only_y
    if n:
        k = min(only_x, only_y)
        p_exact = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
        chi2 = max(0.0, abs(only_x - only_y) - 1) ** 2 / n      # continuity-corrected
        p_chi2 = chi2_sf(chi2, 1)
    else:
        p_exact = p_chi2 = 1.0
    return {"both": both, "only_x": only_x, "only_y": only_y, "neither": len(x) - both - n,
            "discordant": n, "p_exact": p_exact, "p_chi2": p_chi2}


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def percentile(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolated percentile (the numpy default); q in [0, 1]."""
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (pos - lo) * (sorted_vals[hi] - sorted_vals[lo])


def task_bootstrap(per_task: dict[int, list[int]], n_boot: int, seed: int) -> dict[int, tuple[float, float]]:
    """95% percentile CI of the pooled accuracy. per_task[K][i] = seeds (of len(SEEDS)) in which task i is correct;
    the same resampled task set is used for every K."""
    rng = random.Random(seed)
    n = len(next(iter(per_task.values())))
    draws: dict[int, list[float]] = {K: [] for K in per_task}
    for _ in range(n_boot):
        idx = [int(rng.random() * n) for _ in range(n)]
        for K, c in per_task.items():
            draws[K].append(sum(c[i] for i in idx) / (n * len(SEEDS)))
    out = {}
    for K, v in draws.items():
        v.sort()
        out[K] = (percentile(v, 0.025), percentile(v, 0.975))
    return out


def cochran_q(blocks: list[list[bool]]) -> tuple[float, int, int]:
    """blocks[i] = outcomes of block i under each of the k treatments. Returns (Q, df, informative blocks)."""
    k = len(blocks[0])
    col = [sum(b[j] for b in blocks) for j in range(k)]
    row = [sum(b) for b in blocks]
    total = sum(col)
    q = (k - 1) * (k * sum(c * c for c in col) - total ** 2) / (k * total - sum(r * r for r in row))
    return q, k - 1, sum(0 < r < k for r in row)


def pearson_homogeneity(correct: list[int], n: int) -> tuple[float, int]:
    """Pearson chi-square of the (len(correct) x 2) table of correct / incorrect counts, n trials per row."""
    pbar = sum(correct) / (n * len(correct))
    return sum((c - n * pbar) ** 2 for c in correct) / (n * pbar * (1 - pbar)), len(correct) - 1


def run_level_randomization(counts: list[list[int]]) -> tuple[int, int, int]:
    """counts[s][j] = tasks correct in the run of seed s at the j-th K. Exact test that the K labels are
    exchangeable within a seed: every assignment of the labels to the runs of each seed is enumerated.
    Statistic = sum over labels of (k*T_j - S)^2, with T_j the label's correct count summed over seeds and
    S = sum of T_j (an integer, symmetric in the labels, so the first seed's assignment can be held fixed:
    each class of label permutations is then counted once). Returns (observed statistic,
    assignments with statistic >= observed, assignments enumerated)."""
    k = len(counts[0])
    S = sum(sum(c) for c in counts)
    stat = lambda T: sum((k * t - S) ** 2 for t in T)
    obs = stat([sum(c[j] for c in counts) for j in range(k)])
    perms = list(itertools.permutations(range(k)))
    per_seed = [[tuple(c[p[j]] for j in range(k)) for p in perms] for c in counts[1:]]
    ge = n = 0
    for combo in itertools.product(*per_seed):
        ge += stat([counts[0][j] + sum(v[j] for v in combo) for j in range(k)]) >= obs
        n += 1
    return obs, ge, n


# ------------------------------------------------------------------------------------------ report
def fmt_p(p: float) -> str:
    return "<0.0001" if p < 1e-4 else f"{p:.4f}"


def build_report(runs: dict[tuple[int, int], list[bool]], n_boot: int, boot_seed: int) -> str:
    out: list[str] = []
    w = out.append
    seeds_txt = ", ".join(map(str, SEEDS))
    n_pool = len(SEEDS) * N_TASKS

    w("# K-sweep: paired per-task accuracy tests\n")
    w(f"Produced by `python3 scripts/ksweep_paired.py` from `results/ksweep/K<K>_seed<s>.csv` (condition B, "
      f"haiku tier; K = {KS[0]}-{KS[-1]}; seeds {seeds_txt}). Every run covers the same {N_TASKS} tasks; a task is "
      f"correct when `answer_correct` is true on its deepest node (the definition behind `manifest.csv`). "
      f"Counts, intervals and p-values only.\n")

    # ---- (a) McNemar
    w(f"## (a) McNemar tests within each seed ({N_TASKS} paired tasks)\n")
    w(f"`K only`: correct at K, wrong at K={REF}. `K={REF} only`: wrong at K, correct at K={REF}. "
      f"`Discordant` is their sum. `p (exact)` is the two-sided binomial test on the discordant pairs; "
      f"`p (chi2, cc)` is the continuity-corrected chi-square (df = 1). "
      f"{len(SEEDS) * len(CONTRASTS)} tests, no multiplicity adjustment.\n")
    w(f"| Seed | Comparison | Acc (K) | Acc (K={REF}) | Both correct | K only | K={REF} only | Both wrong "
      f"| Discordant | p (exact) | p (chi2, cc) |")
    w("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s in SEEDS:
        for K in CONTRASTS:
            x, y = runs[(K, s)], runs[(REF, s)]
            m = mcnemar(x, y)
            w(f"| {s} | K={K} vs K={REF} | {sum(x) / N_TASKS:.3f} | {sum(y) / N_TASKS:.3f} | {m['both']} "
              f"| {m['only_x']} | {m['only_y']} | {m['neither']} | {m['discordant']} "
              f"| {fmt_p(m['p_exact'])} | {fmt_p(m['p_chi2'])} |")

    # ---- (b) pooled accuracy
    w(f"\n## (b) Pooled over seeds {seeds_txt} ({n_pool} task-seed pairs per K)\n")
    correct = {K: [sum(runs[(K, s)]) for s in SEEDS] for K in KS}
    per_task = {K: [sum(runs[(K, s)][i] for s in SEEDS) for i in range(N_TASKS)] for K in KS}
    ci_boot = task_bootstrap(per_task, n_boot, boot_seed)
    w("| K | Correct per seed (" + " / ".join(map(str, SEEDS)) + f") | Correct / {n_pool} | Accuracy "
      "| 95% CI, Wilson | 95% CI, task bootstrap |")
    w("|---:|---|---:|---:|---|---|")
    for K in KS:
        k_tot = sum(correct[K])
        lo, hi = wilson(k_tot, n_pool)
        blo, bhi = ci_boot[K]
        w(f"| {K} | {' / '.join(map(str, correct[K]))} | {k_tot} | {k_tot / n_pool:.3f} "
          f"| [{lo:.3f}, {hi:.3f}] | [{blo:.3f}, {bhi:.3f}] |")

    # ---- (b) one-way comparison across K
    blocks = [[runs[(K, s)][i] for K in KS] for s in SEEDS for i in range(N_TASKS)]
    q, q_df, informative = cochran_q(blocks)
    chi2, chi2_df = pearson_homogeneity([sum(correct[K]) for K in KS], n_pool)
    obs, ge, n_assign = run_level_randomization([[correct[K][si] for K in KS] for si in range(len(SEEDS))])
    ss_acc = obs / (len(KS) * n_pool) ** 2
    w(f"\nOne-way comparison across K = {KS[0]}-{KS[-1]}:\n")
    w("| Test | Unit of analysis | Statistic | df | p |")
    w("|---|---|---|---:|---:|")
    w(f"| Cochran's Q | {n_pool} task-seed blocks x {len(KS)} K ({informative} blocks not all-equal) "
      f"| Q = {q:.3f} | {q_df} | {fmt_p(chi2_sf(q, q_df))} |")
    w(f"| Pearson chi-square, homogeneity | {len(KS)} x 2 table, {n_pool} pairs per K treated as independent "
      f"| chi2 = {chi2:.3f} | {chi2_df} | {fmt_p(chi2_sf(chi2, chi2_df))} |")
    w(f"| Exact randomization, run level | {len(SEEDS)} runs per K; K labels permuted among the {len(KS)} runs "
      f"of each seed ({n_assign:,} assignments) | SS of K means = {ss_acc:.6f} | n/a | {fmt_p(ge / n_assign)} |")

    # ---- method
    w("\n## Method\n")
    w("- McNemar: discordant pairs b (K only) and c (K=5 only); exact p = min(1, 2 * P(Bin(b + c, 1/2) <= min(b, c))); "
      "chi-square = (max(0, |b - c| - 1))^2 / (b + c), df = 1.")
    w(f"- Wilson interval: the {n_pool} pairs are treated as independent Bernoulli trials. "
      f"Task bootstrap: the {N_TASKS} tasks are resampled with replacement, each carrying its {len(SEEDS)} seeds; "
      f"{n_boot:,} resamples, percentile interval, `random.Random({boot_seed})`.")
    w(f"- Cochran's Q: matched binary outcomes of the {len(KS)} K values on the same task-seed pair; "
      f"chi-square reference with df = {len(KS) - 1}. Pearson chi-square ignores the pairing.")
    w("- Exact randomization: unit = a whole run; the statistic is the sum of squared deviations of the K-level "
      "mean accuracies from their grand mean (K means pooled over seeds); p = share of label assignments with "
      "a statistic at least as large as the observed one (the observed assignment included).")
    w("- Reproduce: `python3 scripts/ksweep_paired.py`.\n")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dir", type=Path, default=Path("results/ksweep"), help="folder with K<K>_seed<s>.csv and manifest.csv")
    ap.add_argument("--out", type=Path, default=None, help="default: <dir>/paired_tests.md")
    ap.add_argument("--boot", type=int, default=10000, help="bootstrap resamples (default 10000)")
    ap.add_argument("--boot-seed", type=int, default=0, help="RNG seed of the bootstrap (default 0)")
    args = ap.parse_args()

    runs = load_runs(args.dir)
    check_against_manifest(args.dir, runs)
    report = build_report(runs, args.boot, args.boot_seed)
    out = args.out or args.dir / "paired_tests.md"
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
