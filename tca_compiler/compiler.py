"""
TCACompiler: chains all components into a single compile() call.

This is the public API of TCA-Compiler. Takes a workflow description
and returns routing decisions for every node.

Usage:
    from tca_compiler import TCACompiler

    compiler = TCACompiler(accuracy_slo=0.85, budget_ceiling=80.00)

    workflow = [
        {"id": "extract",       "class": "extract",       "depth": 1,
         "depends_on": [],      "required_topics": ["invoice","vendor"]},
        {"id": "sql_gen",       "class": "sql-gen",        "depth": 2,
         "depends_on": ["extract"], "required_topics": ["invoice","amount"]},
        {"id": "billing_recon", "class": "billing-recon",  "depth": 3,
         "depends_on": ["sql_gen"], "required_topics": ["invoice","vendor","discrepancy"]},
        {"id": "policy_check",  "class": "policy-check",   "depth": 4,
         "depends_on": ["billing_recon"], "required_topics": ["invoice","compliance"]},
    ]

    result = compiler.compile(workflow)
    print(result.summary())
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .pricing import Tier, TIER_ORDER
from .budget import BudgetGuard
from .cost_profiler import CostProfiler
from .estimator import MemoryInjectionEstimator
from .graph_rewriter import GraphRewriter, WorkflowNode, RewriteLog
from .tier_assigner import TierAssigner, RoutingDecision


@dataclass
class CompileResult:
    """
    Output of TCACompiler.compile().
    Contains routing decisions + rewrite log + cost estimates.
    """
    routing:        dict[str, RoutingDecision]
    rewrite_log:    RewriteLog
    rewritten_nodes: list[WorkflowNode]
    original_depth: int
    rewritten_depth: int
    compilation_ms: float

    def summary(self) -> dict:
        decisions = list(self.routing.values())
        total_tca    = sum(d.estimated_tca for d in decisions)
        total_inject = sum(d.estimated_inject for d in decisions)
        total_inf    = sum(d.estimated_inf for d in decisions)

        tier_dist = {"haiku": 0, "sonnet": 0, "opus": 0}
        strategy_dist: dict[str, int] = {}
        for d in decisions:
            tier_dist[d.tier] += 1
            strategy_dist[d.strategy] = strategy_dist.get(d.strategy, 0) + 1

        return {
            "n_nodes":            len(decisions),
            "original_depth":     self.original_depth,
            "rewritten_depth":    self.rewritten_depth,
            "depth_reduced_by":   self.original_depth - self.rewritten_depth,
            "total_est_tca_usd":  round(total_tca, 6),
            "total_est_inject_usd": round(total_inject, 6),
            "total_est_inf_usd":  round(total_inf, 6),
            "mem_pct_estimated":  round(
                total_inject / total_tca * 100 if total_tca > 0 else 0, 1
            ),
            "tier_distribution":      tier_dist,
            "strategy_distribution":  strategy_dist,
            "rewrite_log":        self.rewrite_log.summary(),
            "compilation_ms":     round(self.compilation_ms, 2),
            "cold_profiles":      sum(1 for d in decisions if d.profile_is_cold),
        }

    def routing_table(self) -> str:
        """Pretty-print routing decisions for logging."""
        lines = [
            f"{'Node':<25} {'Tier':<10} {'Strategy':<18} "
            f"{'Est TCA':>12} {'Mem%':>8} {'Cold':>6}"
        ]
        lines.append("-" * 80)
        for node_id, d in self.routing.items():
            lines.append(
                f"{node_id:<25} {d.tier:<10} {d.strategy:<18} "
                f"${d.estimated_tca:>10.6f} {d.mem_pct_estimated:>7.1f}% "
                f"{'yes' if d.profile_is_cold else 'no':>6}"
            )
        lines.append("-" * 80)
        total_tca = sum(d.estimated_tca for d in self.routing.values())
        total_inj = sum(d.estimated_inject for d in self.routing.values())
        lines.append(
            f"{'TOTAL':<25} {'':<10} {'':<18} "
            f"${total_tca:>10.6f} "
            f"{total_inj/total_tca*100 if total_tca>0 else 0:>7.1f}%"
        )
        return "\n".join(lines)


class TCACompiler:
    """
    Public API: compile a workflow to routing decisions.

    Chains: GraphRewriter → TierAssigner
    Backed by: CostProfiler (persistent learning) + BudgetGuard (safety)
    """

    def __init__(
        self,
        accuracy_slo:   float = 0.85,
        budget_ceiling: float = 80.00,
        per_tier_limits: Optional[dict] = None,
        profile_db:     str = "~/.tca/cost_profiles.db",
        budget_state:   str = ".budget_state.json",
        apply_t1:       bool = True,
        apply_t2:       bool = True,
        apply_t3:       bool = True,
        available_tiers: Optional[tuple] = None,
    ) -> None:
        self.accuracy_slo  = accuracy_slo
        self.apply_t1      = apply_t1
        self.apply_t2      = apply_t2
        self.apply_t3      = apply_t3

        # Core components
        self.profiler   = CostProfiler(db_path=profile_db)
        self.estimator  = MemoryInjectionEstimator(self.profiler)
        self.rewriter   = GraphRewriter(self.profiler)
        self.assigner   = TierAssigner(
            profiler=self.profiler,
            estimator=self.estimator,
            accuracy_slo=accuracy_slo,
            available_tiers=(
                available_tiers if available_tiers is not None
                else tuple(TIER_ORDER)
            ),
        )
        self.budget = BudgetGuard(
            ceiling_usd=budget_ceiling,
            per_tier_limits=per_tier_limits or {},
            state_path=Path(budget_state),
        )

    def compile(
        self,
        workflow: list[dict],
        verbose: bool = False,
    ) -> CompileResult:
        """
        Compile a workflow to routing decisions.

        Args:
            workflow: list of node dicts with keys:
                - id: str
                - class: str
                - depth: int
                - depends_on: list[str]
                - required_topics: list[str]
                - upstream_avg_tokens: float (optional)
            verbose: print routing table after compilation

        Returns:
            CompileResult with routing decisions and metadata
        """
        import time
        t_start = time.perf_counter()

        # Convert dicts to WorkflowNode objects
        nodes = [
            WorkflowNode(
                id=n["id"],
                node_class=n["class"],
                depth=n["depth"],
                depends_on=n.get("depends_on", []),
                required_topics=n.get("required_topics", []),
            )
            for n in workflow
        ]

        original_depth = max(n.depth for n in nodes)

        # Step 1: GraphRewriter — restructure DAG
        rewritten_nodes, rewrite_log = self.rewriter.transform(
            nodes,
            apply_t1=self.apply_t1,
            apply_t2=self.apply_t2,
            apply_t3=self.apply_t3,
        )
        rewritten_depth = max(n.depth for n in rewritten_nodes)

        # Step 2: TierAssigner — joint (strategy, tier) assignment
        # Convert rewritten nodes back to dicts for assigner
        assigner_nodes = [
            {
                "id":    n.id,
                "class": n.node_class,
                "depth": n.depth,
            }
            for n in rewritten_nodes
        ]
        routing = self.assigner.assign(assigner_nodes, verbose=False)

        compilation_ms = (time.perf_counter() - t_start) * 1000

        result = CompileResult(
            routing=routing,
            rewrite_log=rewrite_log,
            rewritten_nodes=rewritten_nodes,
            original_depth=original_depth,
            rewritten_depth=rewritten_depth,
            compilation_ms=compilation_ms,
        )

        if verbose:
            print(result.routing_table())

        return result

    def update_profiles(
        self,
        node_class: str,
        tier: Tier,
        observed_input_base: int,
        observed_output: int,
        observed_inject: int,
        observed_accuracy: float,
        cost_usd: float,
    ) -> None:
        """
        Update cost profiles after a node executes.
        Call this from your LangGraph execution callback.
        Also checks budget — raises BudgetExceededError if ceiling hit.
        """
        self.budget.check_and_record(tier=tier, cost_usd=cost_usd)
        self.profiler.update(
            node_class=node_class,
            tier=tier,
            observed_input_base=observed_input_base,
            observed_output=observed_output,
            observed_inject=observed_inject,
            observed_accuracy=observed_accuracy,
        )

    def budget_summary(self) -> dict:
        return self.budget.summary()

    def profile_summary(self) -> dict:
        return self.profiler.summary()
