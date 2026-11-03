# K-sweep: paired per-task accuracy tests

Produced by `python3 scripts/ksweep_paired.py` from `results/ksweep/K<K>_seed<s>.csv` (condition B, haiku tier; K = 1-5; seeds 7, 42, 99). Every run covers the same 200 tasks; a task is correct when `answer_correct` is true on its deepest node (the definition behind `manifest.csv`). Counts, intervals and p-values only.

## (a) McNemar tests within each seed (200 paired tasks)

`K only`: correct at K, wrong at K=5. `K=5 only`: wrong at K, correct at K=5. `Discordant` is their sum. `p (exact)` is the two-sided binomial test on the discordant pairs; `p (chi2, cc)` is the continuity-corrected chi-square (df = 1). 6 tests, no multiplicity adjustment.

| Seed | Comparison | Acc (K) | Acc (K=5) | Both correct | K only | K=5 only | Both wrong | Discordant | p (exact) | p (chi2, cc) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | K=1 vs K=5 | 0.580 | 0.570 | 87 | 29 | 27 | 57 | 56 | 0.8939 | 0.8937 |
| 7 | K=2 vs K=5 | 0.565 | 0.570 | 93 | 20 | 21 | 66 | 41 | 1.0000 | 1.0000 |
| 42 | K=1 vs K=5 | 0.560 | 0.600 | 87 | 25 | 33 | 55 | 58 | 0.3581 | 0.3580 |
| 42 | K=2 vs K=5 | 0.570 | 0.600 | 90 | 24 | 30 | 56 | 54 | 0.4966 | 0.4962 |
| 99 | K=1 vs K=5 | 0.575 | 0.585 | 91 | 24 | 26 | 59 | 50 | 0.8877 | 0.8875 |
| 99 | K=2 vs K=5 | 0.560 | 0.585 | 87 | 25 | 30 | 58 | 55 | 0.5901 | 0.5896 |

## (b) Pooled over seeds 7, 42, 99 (600 task-seed pairs per K)

| K | Correct per seed (7 / 42 / 99) | Correct / 600 | Accuracy | 95% CI, Wilson | 95% CI, task bootstrap |
|---:|---|---:|---:|---|---|
| 1 | 116 / 112 / 115 | 343 | 0.572 | [0.532, 0.611] | [0.517, 0.627] |
| 2 | 113 / 114 / 112 | 339 | 0.565 | [0.525, 0.604] | [0.510, 0.622] |
| 3 | 127 / 106 / 111 | 344 | 0.573 | [0.533, 0.612] | [0.518, 0.628] |
| 4 | 124 / 121 / 122 | 367 | 0.612 | [0.572, 0.650] | [0.555, 0.668] |
| 5 | 114 / 120 / 117 | 351 | 0.585 | [0.545, 0.624] | [0.527, 0.642] |

One-way comparison across K = 1-5:

| Test | Unit of analysis | Statistic | df | p |
|---|---|---|---:|---:|
| Cochran's Q | 600 task-seed blocks x 5 K (305 blocks not all-equal) | Q = 6.517 | 4 | 0.1637 |
| Pearson chi-square, homogeneity | 5 x 2 table, 600 pairs per K treated as independent | chi2 = 3.347 | 4 | 0.5015 |
| Exact randomization, run level | 3 runs per K; K labels permuted among the 5 runs of each seed (14,400 assignments) | SS of K means = 0.001358 | n/a | 0.3117 |

## Method

- McNemar: discordant pairs b (K only) and c (K=5 only); exact p = min(1, 2 * P(Bin(b + c, 1/2) <= min(b, c))); chi-square = (max(0, |b - c| - 1))^2 / (b + c), df = 1.
- Wilson interval: the 600 pairs are treated as independent Bernoulli trials. Task bootstrap: the 200 tasks are resampled with replacement, each carrying its 3 seeds; 10,000 resamples, percentile interval, `random.Random(0)`.
- Cochran's Q: matched binary outcomes of the 5 K values on the same task-seed pair; chi-square reference with df = 4. Pearson chi-square ignores the pairing.
- Exact randomization: unit = a whole run; the statistic is the sum of squared deviations of the K-level mean accuracies from their grand mean (K means pooled over seeds); p = share of label assignments with a statistic at least as large as the observed one (the observed assignment included).
- Reproduce: `python3 scripts/ksweep_paired.py`.
