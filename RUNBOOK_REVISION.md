# Revision Experiment Runbook — IEEE Access resubmission (Access-2026-30987)

Goal: make every results table in the manuscript reproducible from CSVs in
`results/` via `make_paper_tables.py`. Five runs are missing; this runbook
produces them.

All commands run from the repo root on your Mac. Costs are estimated from
your own previous runs' measured per-task costs.

## 0. Prerequisites (one time)

```bash
cd ~/Documents/All-Proj/tca-compiler
export ANTHROPIC_API_KEY=sk-ant-...   # your key
python3 make_paper_tables.py          # should print tables + list 5 missing runs
```

Two source files were patched for this revision (new CLI flags
`--warm-capacity`, `--haiku-budget`, `--sonnet-budget`, `--ceiling`):
`benchmark/run_experiments.py` and `benchmark/nodes/specialists.py`.
Verify with: `python3 benchmark/run_experiments.py --help`

## 1. CLEAN phase (~$15 total)

Run these one at a time. Each writes a new timestamped CSV into `results/`.
**Note the filename each run prints at the end — you'll paste it into the
manifest in step 2.**

```bash
# 1a. Baseline A, sonnet, seed 7           (~$4.8, ~200 tasks x 4 nodes)
python3 benchmark/run_experiments.py --mode real --tier sonnet --conditions A --seed 7

# 1b. Baseline A, sonnet, seed 99          (~$4.8)
python3 benchmark/run_experiments.py --mode real --tier sonnet --conditions A --seed 99

# 1c. Ablation G, haiku, seed 42           (~$1.5; previous G run stopped at 147/200)
python3 benchmark/run_experiments.py --mode real --tier haiku --conditions G --seed 42 --haiku-budget 2.5

# 1d. Full system H, haiku setting, seed 42 (~$1.5)
python3 benchmark/run_experiments.py --mode real --tier haiku --conditions H --seed 42 --haiku-budget 2.5

# 1e. Capacity-sensitivity: B with warm window K=2 (~$1.5)
python3 benchmark/run_experiments.py --mode real --tier haiku --conditions B --seed 42 --warm-capacity 2 --haiku-budget 2.5
```

If a run dies with a budget error, re-run with a slightly higher
`--haiku-budget` / `--sonnet-budget` / `--ceiling`.

## 2. Update the manifest and rebuild tables

Open `make_paper_tables.py`, find the `RUNS` dict at the top, and replace
each `None` with the filename of the corresponding new CSV (just the file
name, not the path). Then:

```bash
python3 make_paper_tables.py
```

Exit code 0 = every table in the paper is now backed by a CSV on disk.
Send the full output back to Claude — the manuscript text and tables get
rewritten from exactly this output.

## 3. GOLD phase (optional, ~$12 on top)

One single-batch run of the full small-tier ablation, so Table I comes from
one internally consistent execution (no mixed-batch footnote):

```bash
python3 benchmark/run_experiments.py --mode real --tier haiku \
  --conditions A B C D E F G H --seed 42 --haiku-budget 14 --ceiling 20
```

Then point every `ablation_small` entry in `RUNS` at that one file and
re-run `make_paper_tables.py`.

## Sanity expectations (from the verified existing data)

- A sonnet per-task cost should land near $0.0236 (seed 42 measured
  $0.023628; the June 24 partial replication matched per-depth).
- H mid-tier is already banked at all 3 seeds ($0.009985 / $0.009997 /
  $0.009985) — do not re-run.
- Haiku ablation conditions differ by <1% in cost; that is the real result.
- The K=2 run should show fewer injected tokens per task than K=32 (1,074)
  and possibly an accuracy drop — that is the capacity-sensitivity result.

## Do not

- Do not delete or rename anything already in `results/`.
- Do not run condition A at sonnet more than once per seed (it's the
  expensive one, ~$4.8/run).
