"""
Budget guard with hard kill-switch.

Tracks cumulative API spend across all experiments and raises
BudgetExceededError if the ceiling is breached. Designed to fail
loud, not silent — a runaway experiment should crash the harness,
not quietly drain your account.

Usage:
    guard = BudgetGuard(ceiling_usd=80.00, per_tier_limits={
        "haiku": 30.00, "sonnet": 50.00, "opus": 0.00
    })
    guard.check_and_record(tier="haiku", cost_usd=0.0012)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
from typing import Optional

from .pricing import Tier


class BudgetExceededError(RuntimeError):
    """Raised when cumulative spend exceeds the configured ceiling."""


@dataclass
class BudgetGuard:
    """
    Tracks cumulative API spend with a hard ceiling.

    The guard persists state to disk so it survives across runs.
    Restarting an experiment does NOT reset the budget — by design.
    """
    ceiling_usd: float
    per_tier_limits: dict[Tier, float] = field(default_factory=dict)
    state_path: Path = Path(".budget_state.json")
    _spent_total: float = field(default=0.0, init=False)
    _spent_by_tier: dict[Tier, float] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        # Load prior state if it exists
        if self.state_path.exists():
            data = json.loads(self.state_path.read_text())
            self._spent_total = data.get("spent_total", 0.0)
            self._spent_by_tier = data.get("spent_by_tier", {})

    def check_and_record(self, tier: Tier, cost_usd: float) -> None:
        """
        Raise BudgetExceededError if recording this cost would breach
        either the per-tier limit or the global ceiling.
        """
        if cost_usd < 0:
            raise ValueError(f"Cost cannot be negative: {cost_usd}")

        projected_total = self._spent_total + cost_usd
        projected_tier = self._spent_by_tier.get(tier, 0.0) + cost_usd

        # Per-tier check (if configured)
        if tier in self.per_tier_limits:
            tier_limit = self.per_tier_limits[tier]
            if projected_tier > tier_limit:
                raise BudgetExceededError(
                    f"[KILL SWITCH] {tier} budget exceeded: "
                    f"projected ${projected_tier:.4f} > limit ${tier_limit:.2f}. "
                    f"Already spent on {tier}: ${self._spent_by_tier.get(tier, 0.0):.4f}"
                )

        # Global ceiling check
        if projected_total > self.ceiling_usd:
            raise BudgetExceededError(
                f"[KILL SWITCH] Global budget exceeded: "
                f"projected ${projected_total:.4f} > ceiling ${self.ceiling_usd:.2f}. "
                f"Spent so far: ${self._spent_total:.4f}"
            )

        # Record
        self._spent_total = projected_total
        self._spent_by_tier[tier] = projected_tier
        self._persist()

    def _persist(self) -> None:
        self.state_path.write_text(json.dumps({
            "spent_total": self._spent_total,
            "spent_by_tier": self._spent_by_tier,
            "ceiling_usd": self.ceiling_usd,
            "last_updated": datetime.utcnow().isoformat(),
        }, indent=2))

    def remaining(self) -> float:
        return max(0.0, self.ceiling_usd - self._spent_total)

    def remaining_for_tier(self, tier: Tier) -> Optional[float]:
        if tier not in self.per_tier_limits:
            return None
        return max(0.0, self.per_tier_limits[tier] - self._spent_by_tier.get(tier, 0.0))

    def summary(self) -> dict:
        return {
            "ceiling_usd": self.ceiling_usd,
            "spent_total": round(self._spent_total, 4),
            "remaining": round(self.remaining(), 4),
            "spent_by_tier": {k: round(v, 4) for k, v in self._spent_by_tier.items()},
            "per_tier_limits": self.per_tier_limits,
        }

    def reset(self) -> None:
        """
        Delete persisted state. ONLY use this when starting a brand new
        experiment series — never to bypass a kill switch.
        """
        self._spent_total = 0.0
        self._spent_by_tier = {}
        if self.state_path.exists():
            self.state_path.unlink()
