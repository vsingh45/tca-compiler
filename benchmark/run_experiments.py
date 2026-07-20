"""
TCA-Compiler Experiment Runner
==============================
Runs all 200 tasks across 9 ablation conditions and records TCARecords.

Usage:
    # Dry run (no API calls, estimate only):
    python benchmark/run_experiments.py --mode dry --tier haiku

    # Real run - Phase 1 (Haiku only, ~$20-30):
    python benchmark/run_experiments.py --mode real --tier haiku --conditions A B C H

    # Real run - Full (all conditions, ~$80 total):
    python benchmark/run_experiments.py --mode real --tier haiku --conditions all
    python benchmark/run_experiments.py --mode real --tier sonnet --conditions all

    # Single task test:
    python benchmark/run_experiments.py --mode real --tier haiku --task T001

Conditions:
    A = No optimization (AllFrontier + full-history)
    B = Memory only (AllFrontier + TCA-Memory)
    C = Tier only (TierAssigner + full-history)
    D = T1 only (node fusion + AllFrontier + full-history)
    E = T2 only (reorder + AllFrontier + full-history)
    F = T3 only (namespace + AllFrontier + full-history)
    G = Rewrite only (T1+T2+T3 + AllFrontier + full-history)
    H = TCA-Compiler full (T1+T2+T3 + TierAssigner + TCA-Memory)
    I = Oracle (enumerate all, upper bound)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from dotenv import load_dotenv
import anthropic

load_dotenv()

from tca_compiler.budget import BudgetGuard, BudgetExceededError
from tca_compiler.compiler import TCACompiler
from tca_compiler.cost_profiler import CostProfiler
from tca_compiler.pricing import TIER_ORDER
from tca_compiler.record import TCARecord
from tca_memory.store import SharedNamespaceStore
from benchmark.nodes.specialists import make_node


# ── Condition definitions ─────────────────────────────────────────────────────

CONDITIONS = {
    "A": {
        "label":    "No optimization",
        "tier":     "sonnet",      # AllSonnet baseline (opus too expensive for Phase 1)
        "strategy": "full-history",
        "rewrite":  False,
        "assign":   False,
    },
    "B": {
        "label":    "Memory only",
        "tier":     "sonnet",
        "strategy": "warm-isolated",
        "rewrite":  False,
        "assign":   False,
    },
    "C": {
        "label":    "Tier only",
        "tier":     None,          # TierAssigner
        "strategy": "full-history",
        "rewrite":  False,
        "assign":   True,
    },
    "D": {
        "label":    "T1 only",
        "tier":     "opus",
        "strategy": "full-history",
        "rewrite":  True,
        "t1": True, "t2": False, "t3": False,
        "assign":   False,
    },
    "E": {
        "label":    "T2 only",
        "tier":     "opus",
        "strategy": "full-history",
        "rewrite":  True,
        "t1": False, "t2": True, "t3": False,
        "assign":   False,
    },
    "F": {
        "label":    "T3 only",
        "tier":     "opus",
        "strategy": "full-history",
        "rewrite":  True,
        "t1": False, "t2": False, "t3": True,
        "assign":   False,
    },
    "G": {
        "label":    "Rewrite only",
        "tier":     "opus",
        "strategy": "full-history",
        "rewrite":  True,
        "t1": True, "t2": True, "t3": True,
        "assign":   False,
    },
    "H": {
        "label":    "TCA-Compiler full",
        "tier":     None,          # TierAssigner
        "strategy": None,          # GraphRewriter decides
        "rewrite":  True,
        "t1": True, "t2": True, "t3": True,
        "assign":   True,
    },
    "I": {
        "label":    "Oracle (upper bound)",
        "tier":     None,          # Enumerate all
        "strategy": None,
        "rewrite":  False,
        "assign":   "oracle",
    },
}

# Budget per tier (USD) — Phase 1: Haiku only
TIER_BUDGETS = {
    "haiku":  2.00,
    "sonnet": 8.00,
    "opus":   0.00,
}

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


# ── Routing helpers ───────────────────────────────────────────────────────────

def get_routing(
    condition: str,
    task: dict,
    compiler: TCACompiler,
    override_tier: Optional[str] = None,
) -> dict[str, dict]:
    """
    Get (tier, strategy) per node for a given condition.
    Returns dict: node_class -> {tier, strategy}
    """
    cfg = CONDITIONS[condition]
    workflow = task["workflow"]
    nodes_spec = [
        {
            "id": nc,
            "class": nc,
            "depth": i + 1,
            "depends_on": [workflow[i-1]] if i > 0 else [],
            "required_topics": task.get("required_topics", []),
        }
        for i, nc in enumerate(workflow)
    ]

    if cfg.get("assign") == "oracle":
        # Oracle: try all (strategy, tier) combinations
        # For simplicity, use the best known combo from prior runs
        # In full implementation, enumerate and pick minimum TCA
        return {
            nc: {"tier": "haiku", "strategy": "warm-shared"}
            for nc in workflow
        }

    if cfg.get("assign"):
        # TierAssigner — compile to get routing
        result = compiler.compile(
            workflow=nodes_spec,
            verbose=False,
        )
        routing = {}
        # T3 enabled: force warm-shared so nodes actually share memory
        force_shared = cfg.get("t3", False)
        for node_id, decision in result.routing.items():
            nc = node_id.split("_")[0] if "_" in node_id else node_id
            strategy = decision.strategy
            if force_shared and strategy in ("warm-isolated", "vector-only"):
                strategy = "warm-shared"
            elif cfg.get("strategy"):
                strategy = cfg["strategy"]
            routing[nc] = {
                "tier":     decision.tier,
                "strategy": strategy,
            }
        return routing

    # Fixed tier + strategy for all nodes
    tier = override_tier or cfg.get("tier") or "sonnet"
    strategy = cfg.get("strategy") or "warm-isolated"
    return {nc: {"tier": tier, "strategy": strategy} for nc in workflow}


# ── Single task runner ────────────────────────────────────────────────────────

def run_task(
    task: dict,
    condition: str,
    client: anthropic.Anthropic,
    compiler: TCACompiler,
    profiler: CostProfiler,
    guard: BudgetGuard,
    override_tier: Optional[str] = None,
    seed: int = 42,
    dry: bool = False,
    warm_capacity: int = 32,
) -> list[TCARecord]:
    """Run a single task through its full workflow for one condition."""

    workflow_id = f"{task['task_id']}-{condition}-{seed}-{uuid.uuid4().hex[:6]}"
    SharedNamespaceStore.clear_workflow(workflow_id)

    routing = get_routing(condition, task, compiler, override_tier)
    records = []
    # Workflow-level context accumulator for full-history strategy
    # Each node's output gets appended here so downstream nodes can inject it
    workflow_context: list[str] = []

    for depth, node_class in enumerate(task["workflow"], start=1):
        node_routing = routing.get(node_class, {"tier": "haiku", "strategy": "warm-isolated"})
        tier     = node_routing["tier"]
        strategy = node_routing["strategy"]

        if dry:
            # Dry run: estimate cost without API call
            from tca_compiler.estimator import MemoryInjectionEstimator
            from tca_compiler.pricing import TIER_PRICING, TIER_MODELS, cost_inference
            estimator = MemoryInjectionEstimator(profiler)
            est = estimator.estimate(node_class, tier, strategy, depth)
            p_in  = TIER_PRICING[tier]["input"]
            p_out = TIER_PRICING[tier]["output"]
            profile = profiler.get(node_class, tier)
            inf_cost = profile.avg_input_base * p_in + profile.avg_output * p_out
            inj_cost = est.estimated_inject_cost_usd
            total    = inf_cost + inj_cost

            record = TCARecord(
                task_id=task["task_id"],
                workflow_id=workflow_id,
                node_id=node_class,
                node_class=node_class,
                seed=seed,
                tier=tier,
                memory_strategy=strategy,
                condition=condition,
                workflow_depth=depth,
                workflow_category=task["category"],
                base_input_tokens=int(profile.avg_input_base),
                memory_tokens_injected=est.estimated_inject_tokens,
                output_tokens=int(profile.avg_output),
                cache_read_tokens=0,
                cache_creation_tokens=0,
                cost_inference=inf_cost,
                cost_injection=inj_cost,
                cost_miss_penalty=0.0,
                cost_accum=0.0,
                cost_total=total,
                warm_hit=strategy in ("warm-isolated", "warm-shared"),
                fallback_used=False,
                retrieval_latency_ms=1.0,
                answer_correct=True,
                required_topics_recall=0.8,
                end_to_end_ms=500.0,
                model_id=TIER_MODELS[tier],
            )
            records.append(record)
            continue

        # Real run
        try:
            guard.check_and_record(tier, 0.0)  # pre-check budget
        except BudgetExceededError as e:
            print(f"    [BUDGET] {e}")
            raise

        node = make_node(
            node_class=node_class,
            client=client,
            profiler=profiler,
            workflow_id=workflow_id,
            node_id=node_class,
            tier=tier,
            memory_strategy=strategy,
            condition=condition,
            shared_capacity=warm_capacity,
        )

        try:
            result = node.execute(task=task, workflow_depth=depth, seed=seed)
            guard.check_and_record(tier, result.record.cost_total)
            records.append(result.record)
            # Append this node output to workflow context for downstream nodes
            if result.answer:
                workflow_context.append(
                    f"{node_class}: {result.answer[:300]}"
                )
        except BudgetExceededError:
            raise
        except Exception as e:
            print(f"    [ERROR] {node_class}: {e}")
            records.append(node._error_result(task, depth, seed, str(e)).record)

    SharedNamespaceStore.clear_workflow(workflow_id)
    return records


# ── Results writer ────────────────────────────────────────────────────────────

def write_results(records: list[TCARecord], output_path: Path) -> None:
    """Append TCARecords to CSV file."""
    if not records:
        return

    fieldnames = list(asdict(records[0]).keys())
    write_header = not output_path.exists()

    with open(output_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


# ── Main runner ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="TCA-Compiler Experiment Runner")
    parser.add_argument("--mode", choices=["dry", "real"], default="dry")
    parser.add_argument("--tier", choices=["haiku", "sonnet", "opus", "all"], default="haiku")
    parser.add_argument("--conditions", nargs="+", default=["H"],
                        help="Conditions to run: A B C D E F G H I all")
    parser.add_argument("--task", type=str, default=None,
                        help="Run single task by ID (e.g. T001)")
    parser.add_argument("--max-tasks", type=int, default=None,
                        help="Limit number of tasks (for testing)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warm-capacity", type=int, default=32,
                        help="Warm-tier capacity K per namespace (reduce, e.g. to 2, to force warm misses)")
    parser.add_argument("--haiku-budget", type=float, default=None,
                        help="Override haiku tier budget (USD) for this run")
    parser.add_argument("--sonnet-budget", type=float, default=None,
                        help="Override sonnet tier budget (USD) for this run")
    parser.add_argument("--ceiling", type=float, default=10.00,
                        help="Overall budget ceiling (USD) for this run")
    args = parser.parse_args()

    if args.haiku_budget is not None:
        TIER_BUDGETS["haiku"] = args.haiku_budget
    if args.sonnet_budget is not None:
        TIER_BUDGETS["sonnet"] = args.sonnet_budget

    # Resolve conditions
    run_conditions = (
        list(CONDITIONS.keys())
        if "all" in args.conditions
        else args.conditions
    )

    # Resolve tiers
    run_tiers = (
        [t for t in TIER_ORDER if TIER_BUDGETS.get(t, 0) > 0]
        if args.tier == "all"
        else [args.tier]
    )

    # Load tasks
    tasks_path = Path("benchmark/tasks/tasks.json")
    all_tasks  = json.loads(tasks_path.read_text())

    if args.task:
        tasks = [t for t in all_tasks if t["task_id"] == args.task]
        if not tasks:
            print(f"Task {args.task} not found")
            return
    elif args.max_tasks:
        tasks = all_tasks[:args.max_tasks]
    else:
        tasks = all_tasks

    print(f"Mode:       {args.mode}")
    print(f"Tasks:      {len(tasks)}")
    print(f"Conditions: {run_conditions}")
    print(f"Tiers:      {run_tiers}")
    print(f"Seed:       {args.seed}")
    print()

    # Setup
    client = None
    if args.mode == "real":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        client = anthropic.Anthropic(api_key=api_key)

    profiler = CostProfiler()
    guard = BudgetGuard(
        ceiling_usd=args.ceiling,
        per_tier_limits=TIER_BUDGETS,
        state_path=Path(".budget_state.json"),
    )
    guard.reset()  # reset for each new experiment run

    compiler = TCACompiler(
        accuracy_slo=0.40,
        budget_ceiling=sum(TIER_BUDGETS.values()),
    )

    # Output file
    timestamp  = time.strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"tca_results_{args.mode}_{timestamp}.csv"

    # Run experiments
    total_records = 0
    total_cost    = 0.0
    t_start       = time.perf_counter()

    for condition in run_conditions:
        for tier in run_tiers:
            print(f"\n=== Condition {condition} ({CONDITIONS[condition]['label']}) | Tier: {tier} ===")

            for i, task in enumerate(tasks):
                try:
                    records = run_task(
                        task=task,
                        condition=condition,
                        client=client,
                        compiler=compiler,
                        profiler=profiler,
                        guard=guard,
                        override_tier=tier if not CONDITIONS[condition].get("assign") else None,
                        seed=args.seed,
                        dry=(args.mode == "dry"),
                        warm_capacity=args.warm_capacity,
                    )

                    write_results(records, output_path)
                    total_records += len(records)
                    task_cost = sum(r.cost_total for r in records)
                    total_cost += task_cost

                    # Progress
                    if (i + 1) % 10 == 0 or args.task:
                        elapsed = time.perf_counter() - t_start
                        print(
                            f"  [{i+1:3d}/{len(tasks)}] "
                            f"{task['task_id']} ({task['category']}, depth={task['depth']}) "
                            f"cost=${task_cost:.4f} "
                            f"budget_left=${guard.remaining():.2f}"
                        )

                except BudgetExceededError as e:
                    print(f"\n[KILL SWITCH] {e}")
                    print(f"Stopping. Results saved to {output_path}")
                    return

    elapsed = time.perf_counter() - t_start
    print(f"\n{'='*60}")
    print(f"Complete.")
    print(f"  Records:      {total_records}")
    print(f"  Total cost:   ${total_cost:.4f}")
    print(f"  Elapsed:      {elapsed:.1f}s")
    print(f"  Output:       {output_path}")
    print(f"  Budget left:  ${guard.remaining():.2f}")


if __name__ == "__main__":
    main()
