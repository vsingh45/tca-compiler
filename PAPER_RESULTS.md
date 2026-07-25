# Beyond Inference: Paper Results (Revised)

**Status:** ✓ Complete — all values below are regenerated from raw CSVs in `results/` by `make_paper_tables.py`. Run `python3 make_paper_tables.py` to reproduce every table and figure data series in the manuscript.

> **Note on revision history:** an earlier version of this file (and of the original
> IEEE Access submission) reported baseline costs computed over a depth-biased
> subset of tasks, which inflated the baseline and the headline reduction. All
> values below are computed over the full 200-task benchmark and were
> independently replicated across seeds. This file is the single source of
> truth and matches the revised manuscript exactly.

---

## Headline results (mid tier, 3-seed means)

| Metric | Baseline (A) | TCA-Compiler (H) |
|---|---|---|
| Cost/task | $0.023605 [95% CI 0.023555–0.023656] | $0.009989 [95% CI 0.009971–0.010007] |
| End-to-end accuracy | 0.582 [95% CI 0.535–0.629] | 0.627 [95% CI 0.583–0.670] |
| **Cost reduction** | | **57.7%** |
| **Accuracy delta** | | **+4.5 pp** |

Per-seed (both conditions, 200 tasks each):

| Cond | Seed | Cost/task | Accuracy | CSV |
|---|---|---|---|---|
| A | 42 | $0.023628 | 0.590 | `tca_results_real_20260613_014418.csv` |
| A | 7  | $0.023599 | 0.560 | `tca_results_real_20260723_085114.csv` |
| A | 99 | $0.023589 | 0.595 | `tca_results_real_20260723_230302.csv` |
| H | 42 | $0.009985 | 0.625 | `tca_results_real_20260613_175240.csv` |
| H | 7  | $0.009997 | 0.610 | `tca_results_real_20260613_213817.csv` |
| H | 99 | $0.009985 | 0.645 | `tca_results_real_20260613_223022.csv` |

Mechanism: condition H routes 86% of nodes to the haiku tier (688 of 800 node
executions at seed 42) and reserves sonnet for accuracy-sensitive nodes. The
opus tier is disabled by budget configuration; the router never escalates to it.

## Hidden-cost depth curve (condition A, injection share of total TCA)

| Depth | Mid tier (sonnet) | Small tier (haiku) |
|---|---|---|
| 1 | 0.0% | 0.0% |
| 2 | 8.4% | 8.4% |
| 3 | 14.5% | 14.3% |
| 4 | 19.1% | 19.2% |
| 5 | 22.7% | 23.0% |
| 6 | **27.6%** | **28.0%** |

The two curves nearly coincide because both tiers price input:output at the
same 1:5 ratio; the absolute dollar cost of injection is 3× at sonnet.
Injection tokens by depth (sonnet A, seed 42): 0, 143, 324, 466, 602, 754 —
linear fit R² = 0.998.

## Small-tier ablation (seed 42, 200 tasks per condition)

| Cond | Description | Cost/task | Mem% | Acc | Source CSV |
|---|---|---|---|---|---|
| A | No optimization | $0.007422 | 14.4% | 0.635 | `..._20260612_200554.csv` |
| B | Memory only | $0.007436 | 14.4% | 0.600 | `..._20260612_200554.csv` |
| C | Tier only | $0.007415 | 14.5% | 0.645 | `..._20260612_200554.csv` |
| D | T1 fusion only | $0.007439 | 14.5% | 0.630 | `..._20260612_200554.csv` |
| E | T2 reorder only | $0.007434 | 14.5% | 0.600 | `..._20260612_200554.csv` |
| F | T3 namespace only | $0.007398 | 14.5% | 0.575 | `..._20260612_200554.csv` |
| G | Rewrite (all) | $0.007443 | 14.5% | 0.610 | `..._20260723_111909.csv` |
| H | TCA-Compiler full | $0.009983 | 14.8% | 0.590 | `..._20260723_163118.csv` |

**Honest finding:** at the small tier the optimizations are cost-neutral
(A–G within 1%), and H is *more* expensive because it escalates some nodes to
sonnet for accuracy. There is no cheaper tier than the price floor. The
saving is a property of routing an expensive tier downward — the mid-tier
scenario above. The dominant lever is tier reassignment, not graph rewriting.

## Capacity sensitivity (condition B, haiku, seed 42)

| Warm capacity K | Inj tokens/task | Cost/task | Accuracy | Source CSV |
|---|---|---|---|---|
| 32 | 1,074 | $0.007436 | 0.600 | `..._20260612_200554.csv` |
| 2  | 766   | $0.006939 | 0.570 | `..._20260723_172014.csv` |

Reducing the warm window forces cost-aware eviction: −29% injected tokens,
−6.7% cost, −0.030 accuracy. Fallback events are logged; the dollar miss
penalty is zero by construction because the durable store is in-process and
unbilled.

## TCA decomposition (per task, mid tier, seed 42)

| Component | Baseline (A) | TCA-Compiler (H) |
|---|---|---|
| Inference | $0.020408 | $0.008493 |
| Memory injection | $0.003220 | $0.001492 |
| Miss penalty | $0.000000 | $0.000000 |
| Accumulation | $0.000000 | $0.000000 |
| **Total** | **$0.023628** | **$0.009985** |

## Production projection (mid tier, from 3-seed means)

| Daily tasks | Baseline/yr | TCA/yr | Savings/yr |
|---|---|---|---|
| 1,000 | $8,616 | $3,646 | $4,970 |
| 10,000 | $86,158 | $36,460 | $49,698 |
| 100,000 | $861,582 | $364,598 | $496,984 |

## Models and pricing (pinned)

- Small tier: `claude-haiku-4-5-20251001` — $1 / $5 per MTok (in/out)
- Mid tier: `claude-sonnet-4-6` — $3 / $15 per MTok
- Frontier (opus): disabled by budget configuration in all experiments

## Reproduce

```bash
python3 make_paper_tables.py            # regenerates every table above
```

Exact model IDs, seeds, tiers, per-node token counts, and cost components are
recorded per row in the CSVs under `results/`.
