# TCA-Compiler: Measuring and Optimizing Memory Injection Cost in LLM Agent Workflows

Reference implementation and benchmark for the paper:

> **Beyond Inference: Measuring and Optimizing Memory Injection Cost in LLM Agent Workflows**
> Vivek Kumar Singh

---

## System Architecture

[![TCA-Compiler Architecture](https://raw.githubusercontent.com/vsingh45/tca-compiler/main/docs/tca_architecture.svg)](https://raw.githubusercontent.com/vsingh45/tca-compiler/main/docs/tca_architecture.svg)

---

## What this repo contains

| Path | Description |
|---|---|
| `tca_compiler/` | Core TCA-Compiler library (CostProfiler, GraphRewriter, TierAssigner, MemoryInjectionEstimator) |
| `tca_compiler/budget.py` | Hard-ceiling budget guard with persistent spend tracking |
| `tca_memory/` | TCA-Memory two-tier memory backend with cost-aware eviction |
| `benchmark/` | 200-task enterprise benchmark runner and task definitions |
| `benchmark/run_experiments.py` | Main experiment runner (conditions A–H) |
| `results/` | Raw per-node CSV output from all experiment runs |

---

### Key Components

| Component | Purpose | Input | Output |
|-----------|---------|-------|--------|
| **CostProfiler** | Maintains learned cost priors per (node_class, tier) | Execution history | Updated averages |
| **MemoryInjectionEstimator** | Predicts memory cost by depth and strategy | Node depth, strategy, profiler | Estimated injection cost |
| **GraphRewriter** | Applies graph optimizations (T1/T2/T3) | Workflow DAG | Rewritten DAG + metadata |
| **TierAssigner** | Selects optimal (tier, strategy) per node | Profiler, estimator, accuracy SLO | Routing table |
| **TCA-Memory** | Two-tier warm/durable backend with cost-aware eviction | Workflow context | Injected context, metrics |
| **BudgetGuard** | Hard ceiling on cumulative spend | Tier, cost | Accept/Reject execution |

---

## Reproducing the paper results

### 1. Prerequisites

```bash
# Python 3.11+
python --version

# Clone the repo
git clone https://github.com/vsingh45/tca-compiler.git
cd tca-compiler

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Set your Anthropic API key

```bash
export ANTHROPIC_API_KEY=sk-ant-...
# or create a .env file:
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
```

### 3. Set a budget ceiling (important — real API calls cost money)

Before running any experiments, open `benchmark/run_experiments.py` and set
`ceiling_usd` to a value you're comfortable spending. The guard will hard-stop
the run if the ceiling is reached:

```python
guard = BudgetGuard(
    ceiling_usd=5.00,   # <-- set this to your budget
    per_tier_limits={},
    state_path=Path(".budget_state.json"),
)
```

The budget state persists across runs. To reset it:

```bash
rm -f .budget_state.json
```

### 4. Run a single condition (recommended first test)

Test with 5 tasks before committing to a full run:

```bash
python benchmark/run_experiments.py \
  --mode real \
  --tier sonnet \
  --conditions H \
  --seed 42 \
  --max-tasks 5
```

Expected output: a CSV in `results/` with 5 task records, spend logged to
`.budget_state.json`.

### 5. Run the full benchmark

The paper reports results for conditions A (no optimization) and H (full
TCA-Compiler) at the sonnet tier across three random seeds (42, 7, 99).

**Estimated cost per full run (200 tasks):**
- Condition A: ~$0.033/task × 200 = ~$6.60
- Condition H: ~$0.010/task × 200 = ~$2.00

Run each seed separately, checking budget between runs:

```bash
# Condition A — baseline (no optimization)
python benchmark/run_experiments.py --mode real --tier sonnet --conditions A --seed 42
python benchmark/run_experiments.py --mode real --tier sonnet --conditions A --seed 7
python benchmark/run_experiments.py --mode real --tier sonnet --conditions A --seed 99

# Condition H — full TCA-Compiler
python benchmark/run_experiments.py --mode real --tier sonnet --conditions H --seed 42
python benchmark/run_experiments.py --mode real --tier sonnet --conditions H --seed 7
python benchmark/run_experiments.py --mode real --tier sonnet --conditions H --seed 99
```

Use `caffeinate -di` on macOS to prevent sleep during long runs:

```bash
caffeinate -di python benchmark/run_experiments.py --mode real --tier sonnet --conditions H --seed 42
```

### 6. All ablation conditions

| Condition | Description |
|---|---|
| A | No optimization (baseline) |
| B | Memory optimization only |
| C | Tier optimization only |
| D | T1 node fusion only |
| E | T2 injection-aware reordering only |
| F | T3 shared namespace promotion only |
| G | All three GraphRewriter transforms (no tier assignment) |
| H | Full TCA-Compiler (all transforms + tier assignment) |

To run all ablation conditions at once (small tier, single seed — cheaper):

```bash
python benchmark/run_experiments.py \
  --mode real \
  --tier haiku \
  --conditions A B C D E F G H \
  --seed 42
```

---

## Extracting and computing the paper's numbers

All results are in CSV files under `results/`. Each row is one **node execution**
within a workflow. To get per-task costs and end-to-end accuracy, aggregate by
`task_id`.

### Extract headline numbers (cost reduction + accuracy)

```python
import csv, statistics, math
from pathlib import Path
from collections import defaultdict

# Point these at your specific result files
files = {
    ('A', '42'): 'results/tca_results_real_YYYYMMDD_HHMMSS.csv',  # condition A seed 42
    ('H', '42'): 'results/tca_results_real_YYYYMMDD_HHMMSS.csv',  # condition H seed 42
    ('H', '7'):  'results/tca_results_real_YYYYMMDD_HHMMSS.csv',  # condition H seed 7
    ('H', '99'): 'results/tca_results_real_YYYYMMDD_HHMMSS.csv',  # condition H seed 99
}

def per_task_stats(filepath):
    """Return per-task cost and end-to-end accuracy from a result CSV."""
    rows = list(csv.DictReader(open(filepath)))
    tasks = defaultdict(lambda: {'cost': 0.0, 'correct': False, 'depth': -1})
    for r in rows:
        tid = r['task_id']
        tasks[tid]['cost'] += float(r.get('cost_total', 0))
        depth = int(r.get('workflow_depth', 0))
        # Terminal node (deepest) determines end-to-end correctness
        if depth > tasks[tid]['depth']:
            tasks[tid]['depth'] = depth
            tasks[tid]['correct'] = r.get('answer_correct') in ('True', '1', 'true')
    costs = [v['cost'] for v in tasks.values()]
    accs  = [1 if v['correct'] else 0 for v in tasks.values()]
    return costs, accs

# Collect per-seed means
seed_data = {}
for (cond, seed), fpath in files.items():
    costs, accs = per_task_stats(fpath)
    seed_data[(cond, seed)] = {
        'avg_cost': statistics.mean(costs),
        'avg_acc':  statistics.mean(accs),
        'n': len(costs),
    }
    print(f"({cond}, seed {seed}): n={len(costs)}  "
          f"avg_cost=${statistics.mean(costs):.6f}  "
          f"e2e_accuracy={statistics.mean(accs):.3f}")

# 95% CI across H seeds (t-distribution, df=2 for 3 seeds)
def ci95(vals):
    n, m = len(vals), statistics.mean(vals)
    if n < 2:
        return m, None, None
    se = statistics.stdev(vals) / math.sqrt(n)
    t  = 4.303  # t critical for df=2, 95% two-tailed
    return m, m - t * se, m + t * se

h_seeds = [s for (c, s) in seed_data if c == 'H']
h_costs = [seed_data[('H', s)]['avg_cost'] for s in h_seeds]
h_accs  = [seed_data[('H', s)]['avg_acc']  for s in h_seeds]

cm, clo, chi = ci95(h_costs)
am, alo, ahi = ci95(h_accs)

a_cost = seed_data[('A', '42')]['avg_cost']
a_acc  = seed_data[('A', '42')]['avg_acc']

print(f"\n=== PAPER HEADLINE NUMBERS ===")
print(f"Baseline (A, seed 42): ${a_cost:.6f}/task, accuracy={a_acc:.3f}")
print(f"TCA-Compiler (H, {len(h_seeds)} seeds): ${cm:.6f}/task "
      f"(95% CI: ${clo:.6f}–${chi:.6f})")
print(f"Accuracy: {am:.3f} (95% CI: {alo:.3f}–{ahi:.3f})")
print(f"Cost reduction: {(1 - cm/a_cost)*100:.1f}%")
print(f"Accuracy delta: {(am - a_acc)*100:+.1f} pp")
```

### Extract memory injection fraction by depth

```python
import csv, statistics
from pathlib import Path
from collections import defaultdict

# Use condition A (baseline) to measure injection growth
filepath = 'results/tca_results_real_YYYYMMDD_HHMMSS.csv'  # condition A result file
rows = list(csv.DictReader(open(filepath)))

by_depth = defaultdict(lambda: {'inject': [], 'total': []})
for r in rows:
    depth = int(r.get('workflow_depth', 0))
    total = float(r.get('cost_total', 0))
    inject = float(r.get('cost_injection', 0))
    if total > 0:
        by_depth[depth]['inject'].append(inject)
        by_depth[depth]['total'].append(total)

print("Depth | Avg injection tokens | Injection % of total cost")
print("------|---------------------|---------------------------")
for depth in sorted(by_depth.keys()):
    inject_costs = by_depth[depth]['inject']
    total_costs  = by_depth[depth]['total']
    avg_frac = statistics.mean(i/t for i,t in zip(inject_costs, total_costs) if t > 0)
    avg_inject = statistics.mean(inject_costs)
    print(f"  {depth}   | ${avg_inject:.6f}           | {avg_frac*100:.1f}%")
```

### Quick check: verify your results match the paper

```bash
python3 -c "
import csv, statistics
from pathlib import Path
from collections import defaultdict

# Paste your H seed 42 result file path here
f = 'results/YOUR_H_SEED42_FILE.csv'
rows = list(csv.DictReader(open(f)))
tasks = defaultdict(float)
for r in rows:
    tasks[r['task_id']] += float(r.get('cost_total', 0))
costs = list(tasks.values())
print(f'H seed 42: {len(costs)} tasks, avg cost \${statistics.mean(costs):.6f}')
print(f'Expected from paper: ~\$0.009985/task (H, seed 42)')
"
```

---

## CSV column reference

Each row in a result CSV is one **node execution**:

| Column | Description |
|---|---|
| `task_id` | Task identifier (T001–T200) |
| `workflow_id` | Unique workflow run ID |
| `node_id` | Node identifier within the workflow (e.g. `extract`, `sql-gen`) |
| `node_class` | Node type |
| `seed` | Random seed used for this run |
| `tier` | Model tier used for this node (`haiku`, `sonnet`, `opus`) |
| `memory_strategy` | Memory strategy (`warm-isolated`, `warm-shared`, `full-history`) |
| `condition` | Experiment condition (A–H) |
| `workflow_depth` | Depth of this node in the workflow DAG |
| `workflow_category` | Task category (Billing, SAM, IAM, CrossDomain, Policy) |
| `base_input_tokens` | Input tokens excluding memory injection |
| `memory_tokens_injected` | Tokens injected from memory into this node's prompt |
| `output_tokens` | Output tokens generated |
| `cost_inference` | Inference cost ($) |
| `cost_injection` | Memory injection cost ($) |
| `cost_miss_penalty` | Miss-penalty cost ($) |
| `cost_accum` | Accumulation cost ($) |
| `cost_total` | Total node cost ($) |
| `warm_hit` | Whether warm memory tier was hit |
| `fallback_used` | Whether durable-tier fallback was used |
| `retrieval_latency_ms` | Memory retrieval latency |
| `answer_correct` | Whether this node's output was correct |
| `required_topics_recall` | Topic recall for this node |
| `end_to_end_ms` | End-to-end node latency |
| `model_id` | Exact model version used |

**Important:** `answer_correct` is **node-level correctness**, not end-to-end task
accuracy. To compute end-to-end task accuracy, take the `answer_correct` value
from the **deepest node** (highest `workflow_depth`) for each `task_id`. See the
extraction script above.

---

## Paper results summary

Computed from three random seeds (42, 7, 99) for BOTH conditions using
`make_paper_tables.py` (single command reproduces every table in the paper):

| Metric | Condition A (baseline) | Condition H (TCA-Compiler) |
|---|---|---|
| Avg cost/task | $0.023605 (3-seed mean) | $0.009989 (3-seed mean) |
| 95% CI on cost | $0.023555-$0.023656 | $0.009971-$0.010007 |
| End-to-end accuracy | 0.582 (3-seed mean) | 0.627 (3-seed mean) |
| 95% CI on accuracy | 0.535-0.629 | 0.583-0.670 |
| **Cost reduction (mid tier)** | | **57.7%** |
| **Accuracy delta** | | **+4.5 pp** |

At the small tier the optimizations are cost-neutral (conditions A-G within
1%; H slightly higher because it escalates accuracy-sensitive nodes to
sonnet) - there is no cheaper tier than the price floor. See
PAPER_RESULTS.md for the full corrected tables and per-CSV provenance.

Memory injection as fraction of total cost (Condition A, sonnet tier):

| Workflow depth | Injection fraction |
|---|---|
| 1 | 0.0% |
| 2 | 8.4% |
| 3 | 14.5% |
| 4 | 19.1% |
| 5 | 22.7% |
| 6 | 27.6% |

## Budget guard

The `BudgetGuard` in `tca_compiler/budget.py` is a hard kill-switch that prevents
runaway API spend. It persists cumulative spend to `.budget_state.json` across
runs — it does **not** reset between runs by design.

```python
from tca_compiler.budget import BudgetGuard, BudgetExceededError

guard = BudgetGuard(
    ceiling_usd=5.00,              # hard ceiling across all runs
    per_tier_limits={},            # optional per-tier sub-limits
    state_path=Path(".budget_state.json"),
)

# Check and record a cost — raises BudgetExceededError if ceiling breached
guard.check_and_record(tier="sonnet", cost_usd=0.0042)

# Check remaining budget
print(f"Remaining: ${guard.remaining():.4f}")

# Reset ONLY when starting a completely fresh experiment series
guard.reset()
```

---

## Citation

If you use this code or benchmark, please cite:

```bibtex
@misc{singh2026tca,
  title   = {Beyond Inference: Measuring and Optimizing Memory Injection Cost
             in LLM Agent Workflows},
  author  = {Singh, Vivek Kumar},
  year    = {2026},
  url     = {https://github.com/vsingh45/tca-compiler}
}
```

---

## License

MIT License. See `LICENSE` for details.
