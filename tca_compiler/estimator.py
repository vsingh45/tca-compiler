"""
MemoryInjectionEstimator: predicts C_inject per node before execution.

Uses Theorem 1's closed-form bounds to estimate memory injection cost
at compile time — before any specialist runs. This is what makes
TCA-Compiler a compile-time system rather than a runtime one.

Theorem 1 (from paper §3.4):
  Full-history:  C_inject(d) = τ × p_in × d(d-1)/2  → O(d²)
  WarmMemory:    C_inject(d) = min(d, K) × τ × p_in  → O(min(d,K))

Where:
  d = workflow depth (position of node in DAG)
  τ = average tokens per upstream output
  K = warm buffer capacity
  p_in = input price per token for assigned tier
"""
from __future__ import annotations

from dataclasses import dataclass

from .pricing import Tier, TIER_PRICING
from .cost_profiler import CostProfiler, NodeProfile

# Warm buffer capacities per strategy (from paper §4 config)
WARM_CAPACITY: dict[str, int] = {
    "warm-isolated": 16,
    "warm-shared":   32,
    "vector-only":   5,    # top-k retrieved
    "full-history":  9999, # unbounded
}


@dataclass
class InjectionEstimate:
    """Injection cost estimate for one (node, strategy, tier) combination."""
    node_class: str
    tier: Tier
    strategy: str
    depth: int
    estimated_inject_tokens: int
    estimated_inject_cost_usd: float
    estimated_miss_penalty_usd: float
    estimated_total_memory_cost_usd: float
    theorem1_formula: str   # human-readable formula used


class MemoryInjectionEstimator:
    """
    Predicts memory injection cost per node using Theorem 1 bounds.

    Used by TierAssigner at compile time to select the cheapest
    (strategy, tier) pair without executing the node.
    """

    def __init__(self, profiler: CostProfiler) -> None:
        self.profiler = profiler

    def estimate(
        self,
        node_class: str,
        tier: Tier,
        strategy: str,
        depth: int,
        upstream_avg_tokens: float | None = None,
    ) -> InjectionEstimate:
        """
        Estimate injection cost for one (node, strategy, tier) at a given depth.

        Args:
            node_class: e.g. "sql-gen", "billing-recon"
            tier: model tier for this node
            strategy: memory strategy
            depth: 1-indexed position of node in workflow (1 = first node)
            upstream_avg_tokens: override for average upstream output tokens.
                If None, uses profile's avg_inject as proxy.

        Returns:
            InjectionEstimate with cost breakdown.
        """
        profile = self.profiler.get(node_class, tier)
        p_in = TIER_PRICING[tier]["input"]

        # Use profile's avg_inject as proxy for upstream token size
        # (what upstream nodes typically produce that this node consumes)
        tau = upstream_avg_tokens or profile.avg_inject

        # ── Theorem 1: injection tokens by strategy ───────────────────────
        if strategy == "full-history":
            # O(d²): node at depth d injects outputs of ALL prior turns
            # Each turn adds tau tokens — total grows as triangular number
            # At depth d: inject = tau * (1 + 2 + ... + (d-1)) = tau * d*(d-1)/2
            n_prior = max(0, depth - 1)
            inject_tokens = int(tau * n_prior * (n_prior + 1) / 2)
            formula = f"τ × d(d-1)/2 = {tau:.0f} × {n_prior}×{n_prior+1}/2"

        elif strategy in ("warm-isolated", "warm-shared"):
            K = WARM_CAPACITY[strategy]
            # O(min(d,K)): warm buffer caps at K entries — linear not quadratic
            inject_tokens = int(min(max(0, depth - 1), K) * tau)
            formula = f"min(depth-1, K={K}) × τ = {min(depth-1, K)} × {tau:.0f}"

        elif strategy == "vector-only":
            # Fixed top-k retrieval regardless of depth
            k = WARM_CAPACITY["vector-only"]
            inject_tokens = int(k * tau)
            formula = f"top_k={k} × τ = {k} × {tau:.0f}"

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # ── Miss penalty estimate ─────────────────────────────────────────
        # Durable retrieval returns ~2x more tokens than warm hit
        # Miss rate estimated from profile accuracy as a proxy
        miss_rate = 0.0
        if strategy in ("warm-isolated", "warm-shared"):
            # Rough proxy: lower accuracy nodes have higher miss rates
            miss_rate = max(0.0, 1.0 - profile.accuracy) * 0.3
        elif strategy == "vector-only":
            miss_rate = 0.15  # vector retrieval has ~15% empty-result rate

        miss_penalty_tokens = int(miss_rate * inject_tokens * 0.5)

        # ── Cost computation ──────────────────────────────────────────────
        inject_cost = inject_tokens * p_in
        miss_cost   = miss_penalty_tokens * p_in
        total_mem   = inject_cost + miss_cost

        return InjectionEstimate(
            node_class=node_class,
            tier=tier,
            strategy=strategy,
            depth=depth,
            estimated_inject_tokens=inject_tokens,
            estimated_inject_cost_usd=inject_cost,
            estimated_miss_penalty_usd=miss_cost,
            estimated_total_memory_cost_usd=total_mem,
            theorem1_formula=formula,
        )

    def estimate_all(
        self,
        node_class: str,
        depth: int,
        strategies: list[str] | None = None,
        tiers: list[Tier] | None = None,
    ) -> list[InjectionEstimate]:
        """
        Estimate injection cost for all (strategy, tier) combinations.
        Used by TierAssigner to enumerate routing candidates.
        """
        from .pricing import TIER_ORDER
        _strategies = strategies or list(WARM_CAPACITY.keys())
        _tiers = tiers or TIER_ORDER

        return [
            self.estimate(node_class, tier, strategy, depth)
            for strategy in _strategies
            for tier in _tiers
        ]

    def depth_curve(
        self,
        node_class: str,
        tier: Tier,
        strategy: str,
        max_depth: int = 6,
    ) -> list[tuple[int, float]]:
        """
        Return (depth, inject_cost_usd) pairs for depths 1..max_depth.
        Used to generate Figure 2 (depth amplification curves) in the paper.
        """
        return [
            (d, self.estimate(node_class, tier, strategy, d).estimated_inject_cost_usd)
            for d in range(1, max_depth + 1)
        ]
