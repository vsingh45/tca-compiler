"""
TierAssigner: joint (memory_strategy, model_tier) assignment per node.

The core routing engine of TCA-Compiler. For each node in the workflow
DAG (processed in topological order), TierAssigner:

1. Retrieves the node's cost profile from CostProfiler
2. Estimates injection cost for all 12 (strategy, tier) combinations
3. Filters out combinations that violate the accuracy SLO
4. Selects the minimum-TCA combination
5. Propagates context estimates to downstream nodes

This is what makes TCA-Compiler different from inference-only routing:
it accounts for C_inject (memory injection cost) in the routing decision,
not just C_inf (inference cost).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .pricing import Tier, TIER_PRICING, TIER_ORDER, TIER_MODELS
from .cost_profiler import CostProfiler, NodeProfile
from .estimator import MemoryInjectionEstimator, WARM_CAPACITY


STRATEGIES = list(WARM_CAPACITY.keys())


@dataclass(slots=True)
class RoutingDecision:
    """Routing decision for one node."""
    node_id:          str
    node_class:       str
    tier:             Tier
    strategy:         str
    estimated_tca:    float   # total estimated cost for this node (USD)
    estimated_inject: float   # injection component only (USD)
    estimated_inf:    float   # inference component only (USD)
    accuracy_slo_met: bool
    profile_is_cold:  bool    # True if using bundled priors
    model_id:         str     # concrete Anthropic model string

    @property
    def mem_pct_estimated(self) -> float:
        """Estimated memory injection as % of total node cost."""
        if self.estimated_tca == 0:
            return 0.0
        return self.estimated_inject / self.estimated_tca * 100


@dataclass
class TierAssigner:
    """
    Joint (strategy, tier) assignment for all nodes in a workflow.

    Processes nodes in topological order so upstream routing decisions
    can inform downstream injection cost estimates.
    """
    profiler:      CostProfiler
    estimator:     MemoryInjectionEstimator
    accuracy_slo:  float = 0.85    # minimum acceptable accuracy per node
    frontier_fallback: bool = True  # if no candidate meets SLO, use frontier

    def assign(
        self,
        nodes: list[dict],
        verbose: bool = False,
    ) -> dict[str, RoutingDecision]:
        """
        Assign (strategy, tier) to every node in the workflow.

        Args:
            nodes: list of dicts with keys:
                   - id: str (node identifier)
                   - class: str (node class for profile lookup)
                   - depth: int (1-indexed position in DAG)
                   - upstream_avg_tokens: float | None (override)
            verbose: print routing decisions as they're made

        Returns:
            dict mapping node_id -> RoutingDecision
        """
        routing: dict[str, RoutingDecision] = {}
        context_estimates: dict[str, float] = {}  # node_id -> expected output tokens

        for node in nodes:
            node_id    = node["id"]
            node_class = node["class"]
            depth      = node["depth"]
            upstream_override = node.get("upstream_avg_tokens")

            # Use upstream context estimates if available
            upstream_avg = upstream_override
            if upstream_avg is None and depth > 1:
                upstream_vals = list(context_estimates.values())
                if upstream_vals:
                    upstream_avg = sum(upstream_vals) / len(upstream_vals)

            decision = self._assign_node(
                node_id=node_id,
                node_class=node_class,
                depth=depth,
                upstream_avg_tokens=upstream_avg,
            )
            routing[node_id] = decision

            # Propagate this node's expected output to downstream estimates
            profile = self.profiler.get(node_class, decision.tier)
            context_estimates[node_id] = profile.avg_output

            if verbose:
                print(
                    f"  [{node_id}] {node_class} → "
                    f"tier={decision.tier}, strategy={decision.strategy}, "
                    f"est_tca=${decision.estimated_tca:.6f}, "
                    f"mem%={decision.mem_pct_estimated:.1f}%"
                )

        return routing

    def _assign_node(
        self,
        node_id: str,
        node_class: str,
        depth: int,
        upstream_avg_tokens: float | None,
    ) -> RoutingDecision:
        """Find the minimum-TCA (strategy, tier) for one node."""
        candidates = []

        for tier in TIER_ORDER:
            profile = self.profiler.get(node_class, tier)

            # Skip if accuracy SLO not met
            if profile.accuracy < self.accuracy_slo:
                continue

            for strategy in STRATEGIES:
                est = self.estimator.estimate(
                    node_class=node_class,
                    tier=tier,
                    strategy=strategy,
                    depth=depth,
                    upstream_avg_tokens=upstream_avg_tokens,
                )

                # Estimate inference cost
                p_in  = TIER_PRICING[tier]["input"]
                p_out = TIER_PRICING[tier]["output"]
                inf_cost = (
                    profile.avg_input_base * p_in
                    + profile.avg_output * p_out
                )

                total_tca = inf_cost + est.estimated_total_memory_cost_usd

                candidates.append({
                    "tier":             tier,
                    "strategy":         strategy,
                    "estimated_tca":    total_tca,
                    "estimated_inject": est.estimated_total_memory_cost_usd,
                    "estimated_inf":    inf_cost,
                    "profile_is_cold":  profile.is_cold(),
                })

        if not candidates:
            # No candidate meets accuracy SLO — use frontier as safe fallback
            if self.frontier_fallback:
                tier = "opus"
                strategy = "warm-isolated"
                profile = self.profiler.get(node_class, tier)
                p_in  = TIER_PRICING[tier]["input"]
                p_out = TIER_PRICING[tier]["output"]
                inf_cost = profile.avg_input_base * p_in + profile.avg_output * p_out
                est = self.estimator.estimate(node_class, tier, strategy, depth)
                return RoutingDecision(
                    node_id=node_id,
                    node_class=node_class,
                    tier=tier,
                    strategy=strategy,
                    estimated_tca=inf_cost + est.estimated_total_memory_cost_usd,
                    estimated_inject=est.estimated_total_memory_cost_usd,
                    estimated_inf=inf_cost,
                    accuracy_slo_met=False,
                    profile_is_cold=profile.is_cold(),
                    model_id=TIER_MODELS[tier],
                )
            raise ValueError(
                f"No routing candidate meets accuracy SLO {self.accuracy_slo} "
                f"for node '{node_id}' (class: {node_class})"
            )

        # Select minimum TCA candidate
        best = min(candidates, key=lambda c: c["estimated_tca"])

        return RoutingDecision(
            node_id=node_id,
            node_class=node_class,
            tier=best["tier"],
            strategy=best["strategy"],
            estimated_tca=best["estimated_tca"],
            estimated_inject=best["estimated_inject"],
            estimated_inf=best["estimated_inf"],
            accuracy_slo_met=True,
            profile_is_cold=best["profile_is_cold"],
            model_id=TIER_MODELS[best["tier"]],
        )

    def routing_summary(
        self,
        routing: dict[str, RoutingDecision],
    ) -> dict:
        """Aggregate routing decisions for logging and paper tables."""
        decisions = list(routing.values())
        total_est_tca     = sum(d.estimated_tca for d in decisions)
        total_est_inject  = sum(d.estimated_inject for d in decisions)
        total_est_inf     = sum(d.estimated_inf for d in decisions)
        tier_counts       = {t: 0 for t in TIER_ORDER}
        strategy_counts   = {s: 0 for s in STRATEGIES}

        for d in decisions:
            tier_counts[d.tier] += 1
            strategy_counts[d.strategy] += 1

        return {
            "n_nodes":           len(decisions),
            "total_est_tca":     round(total_est_tca, 6),
            "total_est_inject":  round(total_est_inject, 6),
            "total_est_inf":     round(total_est_inf, 6),
            "mem_pct_estimated": round(
                total_est_inject / total_est_tca * 100
                if total_est_tca > 0 else 0, 1
            ),
            "tier_distribution": tier_counts,
            "strategy_distribution": strategy_counts,
            "cold_profiles":     sum(1 for d in decisions if d.profile_is_cold),
        }
