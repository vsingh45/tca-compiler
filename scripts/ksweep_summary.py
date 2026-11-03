#!/usr/bin/env python3
"""K-sweep summary: aggregate results/ksweep/manifest.csv over seeds -> summary.csv + fig_ksweep.png.

    python3 scripts/ksweep_summary.py              # summary.csv and fig_ksweep.png
    python3 scripts/ksweep_summary.py --no-fig     # summary.csv only (needs no matplotlib)

One row per warm-memory capacity K (condition B, haiku, 200 tasks per run). Each manifest row is one
(K, seed) run; *_mean and *_std are the mean and the sample standard deviation (n-1) of the per-run
values across the seeds present, and *_std is blank for a K that ran at a single seed (K=32).
  inj_tokens  memory tokens injected, summed over a task's nodes, mean over the 200 tasks
  acc         answer_correct of each task's deepest node, mean over the 200 tasks
  cost        run total, USD

Figure: K on a log axis, tokens/task and accuracy as two panels (different scales, so no shared
axis). Line and filled marker = mean across seeds, bars = +-1 s.d., small dots = the individual
seeds (dodged left to right as seed 7, 42, 99).
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

# (summary column prefix, manifest column, decimals)
METRICS = [("inj_tokens", "inj_tokens_per_task", 1), ("acc", "accuracy", 4), ("cost", "cost", 4)]
SUMMARY_FIELDS = ["K", "n_seeds", "seeds"] + [f"{m}_{s}" for m, _, _ in METRICS for s in ("mean", "std")]

SURFACE, INK, INK2 = "#ffffff", "#0b0b0b", "#52514e"
GRID, AXIS, SERIES = "#e1e0d9", "#c3c2b7", "#2a78d6"
DODGE = {7: 0.93, 42: 1.0, 99: 1.07}   # multiplicative offset on the log-K axis


def read_runs(manifest: Path) -> dict[int, list[dict]]:
    runs: dict[int, list[dict]] = defaultdict(list)
    seen: set[tuple[int, int]] = set()
    with open(manifest, newline="") as fh:
        for r in csv.DictReader(fh):
            key = (int(r["K"]), int(r["seed"]))
            if key in seen:
                raise SystemExit(f"{manifest}: duplicate run K={key[0]} seed={key[1]}")
            seen.add(key)
            runs[key[0]].append({"seed": key[1], **{col: float(r[col]) for _, col, _ in METRICS}})
    return dict(sorted(runs.items()))


def summarize(runs: dict[int, list[dict]]) -> list[dict]:
    rows = []
    for K, rs in runs.items():
        rs = sorted(rs, key=lambda r: r["seed"])
        row = {"K": K, "n_seeds": len(rs), "seeds": " ".join(str(r["seed"]) for r in rs)}
        for name, col, _ in METRICS:
            vals = [r[col] for r in rs]
            row[f"{name}_mean"] = statistics.fmean(vals)
            row[f"{name}_std"] = statistics.stdev(vals) if len(vals) > 1 else None
        rows.append(row)
    return rows


def write_summary(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(SUMMARY_FIELDS)
        for row in rows:
            out = [row["K"], row["n_seeds"], row["seeds"]]
            for name, _, nd in METRICS:
                for stat in ("mean", "std"):
                    v = row[f"{name}_{stat}"]
                    out.append("" if v is None else f"{v:.{nd}f}")
            w.writerow(out)


def plot(runs: dict[int, list[dict]], rows: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, MultipleLocator, NullLocator

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.major.size": 3, "ytick.major.size": 3,
    })
    Ks = [r["K"] for r in rows]
    nmax = max(r["n_seeds"] for r in rows)
    panels = [
        ("inj_tokens", "inj_tokens_per_task", "(a) Injected memory tokens per task", "Tokens per task", "{:.0f}"),
        ("acc", "accuracy", "(b) Task accuracy", "Accuracy", "{:.3f}"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.9), facecolor=SURFACE, layout="constrained")
    for ax, (name, col, title, ylabel, fmt) in zip(axes, panels):
        means = [r[f"{name}_mean"] for r in rows]
        stds = [r[f"{name}_std"] for r in rows]
        ax.set_facecolor(SURFACE)
        ax.plot(Ks, means, color=SERIES, lw=1.5, solid_capstyle="round", solid_joinstyle="round", zorder=2)
        for K, m, s in zip(Ks, means, stds):
            if s is not None:
                ax.errorbar(K, m, yerr=s, fmt="none", color=SERIES, lw=1.0, capsize=2.5, capthick=1.0, zorder=3)
        for K, rs in runs.items():
            ax.scatter([K * DODGE.get(r["seed"], 1.0) for r in rs], [r[col] for r in rs],
                       s=9, color=SERIES, alpha=0.45, linewidths=0, zorder=3)
        ax.scatter(Ks, means, s=34, color=SERIES, edgecolors=SURFACE, linewidths=1.2, zorder=4)

        # label the extremes only (the axis carries the rest), clear of the bars and seed dots
        ink = {K: [r[col] for r in rs] + [m - (s or 0), m + (s or 0)]
               for (K, rs), m, s in zip(runs.items(), means, stds)}
        pad = {"boxstyle": "square,pad=0.15", "fc": SURFACE, "ec": "none"}   # keep gridlines off the text
        for i in (0, len(Ks) - 1):
            ax.annotate(fmt.format(means[i]), (Ks[i], max(ink[Ks[i]])), textcoords="offset points",
                        xytext=(0, 5), ha="center", va="bottom", fontsize=7.5, color=INK2, bbox=pad)
        for K, r in zip(Ks, rows):
            if r["n_seeds"] < nmax:
                ax.annotate(f"n = {r['n_seeds']}", (K, min(ink[K])), textcoords="offset points",
                            xytext=(0, -6), ha="center", va="top", fontsize=7, color=INK2, bbox=pad)

        ax.set_xscale("log")
        ax.set_xlim(0.8, 42)
        ax.xaxis.set_major_locator(FixedLocator(Ks))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xlabel("Warm-memory capacity K (entries per workflow)")
        if name == "inj_tokens":
            top = max(r[col] for rs in runs.values() for r in rs)
            ax.set_ylim(0, top * 1.2)
            ax.yaxis.set_major_locator(MaxNLocator(6, steps=[1, 2, 2.5, 5, 10], integer=True))
        else:
            vals = [r[col] for rs in runs.values() for r in rs]
            vals += [m - (s or 0) for m, s in zip(means, stds)] + [m + (s or 0) for m, s in zip(means, stds)]
            ax.set_ylim(max(0.0, math.floor((min(vals) - 0.04) / 0.05) * 0.05),
                        min(1.0, math.ceil((max(vals) + 0.04) / 0.05) * 0.05))
            ax.yaxis.set_major_locator(MultipleLocator(0.05))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontsize=8.5, color=INK, fontweight="bold")
        ax.grid(axis="y", color=GRID, lw=0.6, ls="-")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    key = [
        Line2D([0], [0], color=SERIES, lw=1.5, marker="o", ms=5.5, markeredgecolor=SURFACE,
               markeredgewidth=1.0, label="mean ± s.d. across seeds"),
        Line2D([0], [0], color=SERIES, lw=0, marker="o", ms=3, alpha=0.45, markeredgewidth=0,
               label="individual seed"),
    ]
    fig.legend(handles=key, loc="outside lower center", ncol=2, frameon=False, fontsize=7.5,
               labelcolor=INK2, handletextpad=0.4, columnspacing=1.8)
    fig.savefig(path, dpi=300, facecolor=SURFACE)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=Path("results/ksweep/manifest.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/ksweep"))
    ap.add_argument("--no-fig", action="store_true", help="write summary.csv only (no matplotlib needed)")
    args = ap.parse_args()

    runs = read_runs(args.manifest)
    rows = summarize(runs)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_summary(rows, args.out_dir / "summary.csv")

    sd = lambda v, nd: "   n/a" if v is None else f"{v:.{nd}f}"
    print(f"{'K':>3} {'seeds':>9} {'tokens/task':>11} {'sd':>6} {'accuracy':>9} {'sd':>6} {'cost $':>8} {'sd':>6}")
    for r in rows:
        print(f"{r['K']:>3} {r['seeds']:>9} {r['inj_tokens_mean']:>11.1f} {sd(r['inj_tokens_std'], 1):>6} "
              f"{r['acc_mean']:>9.4f} {sd(r['acc_std'], 4):>6} {r['cost_mean']:>8.4f} {sd(r['cost_std'], 4):>6}")
    print(f"wrote {args.out_dir / 'summary.csv'}")
    if not args.no_fig:
        plot(runs, rows, args.out_dir / "fig_ksweep.png")
        print(f"wrote {args.out_dir / 'fig_ksweep.png'}")


if __name__ == "__main__":
    main()
