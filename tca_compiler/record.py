"""
TCARecord: immutable measurement record from a single node execution.

Records capture all metrics from a TCA node's execution: costs, correctness,
latency, memory state. Used for post-hoc analysis and TCA fairness auditing.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Optional
from .pricing import Tier


@dataclass(frozen=True)
class TCARecord:
    """Immutable execution record for a single TCA node."""

    # Node identity
    task_id: str
    workflow_id: str
    node_id: str
    node_class: str

    # Experimental conditions
    seed: int
    tier: Tier
    memory_strategy: str
    condition: str
    workflow_depth: int
    workflow_category: str

    # Token accounting
    base_input_tokens: int
    memory_tokens_injected: int
    output_tokens: int

    # Cost components (all in USD)
    cost_inference: float
    cost_injection: float
    cost_miss_penalty: float
    cost_accum: float
    cost_total: float

    # Memory and fallback state
    warm_hit: bool
    fallback_used: bool
    retrieval_latency_ms: float

    # Correctness metrics
    answer_correct: bool
    required_topics_recall: float

    # Optional fields with defaults
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    end_to_end_ms: float = 0.0
    model_id: str = ""

    def __post_init__(self) -> None:
        """Validate that cost totals are consistent."""
        expected_total = (
            self.cost_inference
            + self.cost_injection
            + self.cost_miss_penalty
            + self.cost_accum
        )
        # Allow small floating-point error (1e-10)
        if abs(self.cost_total - expected_total) > 1e-10:
            raise ValueError(
                f"cost_total ({self.cost_total:.10f}) does not match sum of "
                f"components ({expected_total:.10f}). "
                f"Difference: {abs(self.cost_total - expected_total):.10e}"
            )

    def to_dict(self) -> dict:
        """Serialize record to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> TCARecord:
        """Deserialize record from dictionary."""
        return cls(**data)

    def __eq__(self, other: object) -> bool:
        """Compare two records for equality."""
        if not isinstance(other, TCARecord):
            return NotImplemented
        return asdict(self) == asdict(other)
