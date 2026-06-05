"""
GraphRewriter: compile-time DAG transformation to minimize TCA.

Three transforms applied in sequence before TierAssigner runs:

T1 — Node Fusion:
    Merges adjacent nodes (v_i → v_j where v_j has only one dependency)
    to reduce workflow depth. Fewer nodes = less context accumulation.
    Conservative safety guards prevent incorrect fusion.

T2 — Injection-Aware Reordering:
    For parallel nodes (no dependency between them), places higher-output
    nodes LAST so fewer downstream nodes inherit their token footprint.

T3 — Shared Namespace Promotion:
    Nodes sharing ≥60% of required_topics get promoted to a shared
    warm buffer namespace, increasing warm-hit rate across both nodes.

All transforms are semantics-preserving by construction (via safety guards).
The rewrite_log records every transformation for ablation analysis.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .cost_profiler import CostProfiler
from .pricing import TIER_MODELS


# Shared namespace promotion threshold (T3)
SHARED_NAMESPACE_THRESHOLD = 0.60  # 60% topic overlap required

# Context window safety margin for T1 fusion (80% of limit)
CONTEXT_WINDOW_SAFETY = 0.80

# Context window per tier (tokens)
CONTEXT_WINDOW: dict[str, int] = {
    "haiku":  200_000,
    "sonnet": 200_000,
    "opus":   200_000,
}


@dataclass
class WorkflowNode:
    """
    Node in the workflow DAG.
    This is TCA-Compiler's internal representation — not a LangGraph node.
    """
    id:               str
    node_class:       str
    depth:            int
    depends_on:       list[str] = field(default_factory=list)
    required_topics:  list[str] = field(default_factory=list)
    namespace:        str = "isolated"   # "isolated" | "shared:{group_id}"
    fused_from:       list[str] = field(default_factory=list)  # T1 provenance
    avg_output_tokens: float = 200.0     # estimated from profiler


@dataclass
class RewriteLog:
    """Records every transformation for ablation analysis."""
    t1_fusions:      list[dict] = field(default_factory=list)
    t2_reorders:     list[dict] = field(default_factory=list)
    t3_promotions:   list[dict] = field(default_factory=list)

    @property
    def t1_fired(self) -> bool:
        return len(self.t1_fusions) > 0

    @property
    def t2_fired(self) -> bool:
        return len(self.t2_reorders) > 0

    @property
    def t3_fired(self) -> bool:
        return len(self.t3_promotions) > 0

    def summary(self) -> dict:
        return {
            "t1_fusions":    len(self.t1_fusions),
            "t2_reorders":   len(self.t2_reorders),
            "t3_promotions": len(self.t3_promotions),
            "t1_fired":      self.t1_fired,
            "t2_fired":      self.t2_fired,
            "t3_fired":      self.t3_fired,
        }


class GraphRewriter:
    """
    Applies T1, T2, T3 transforms to a workflow DAG.
    Returns the rewritten DAG and a log of transformations made.
    """

    def __init__(self, profiler: CostProfiler) -> None:
        self.profiler = profiler

    def _enrich_node(self, node: WorkflowNode) -> WorkflowNode:
        """Enrich node with profile data before transforms run."""
        profile = self.profiler.get(node.node_class, "sonnet")
        node.avg_output_tokens = profile.avg_output
        return node

    def transform(
        self,
        nodes: list[WorkflowNode],
        default_tier: str = "sonnet",
        apply_t1: bool = True,
        apply_t2: bool = True,
        apply_t3: bool = True,
    ) -> tuple[list[WorkflowNode], RewriteLog]:
        """
        Apply T1 → T2 → T3 in sequence.

        Args:
            nodes: workflow nodes in topological order
            default_tier: tier used for context window guard in T1
            apply_t1/t2/t3: toggle individual transforms (for ablation)

        Returns:
            (rewritten_nodes, rewrite_log)
        """
        log = RewriteLog()
        result = [self._enrich_node(n) for n in nodes]

        if apply_t1:
            result = self._apply_t1(result, default_tier, log)
        if apply_t2:
            result = self._apply_t2(result, log)
        if apply_t3:
            result = self._apply_t3(result, log)

        # Recompute depths after transforms
        result = self._recompute_depths(result)

        return result, log

    # ── T1: Node Fusion ───────────────────────────────────────────────────────

    def _apply_t1(
        self,
        nodes: list[WorkflowNode],
        default_tier: str,
        log: RewriteLog,
    ) -> list[WorkflowNode]:
        """
        Merge adjacent nodes to reduce workflow depth.
        Only fuses when safety guards are satisfied.
        """
        changed = True
        while changed:
            changed = False
            for i, node in enumerate(nodes):
                if len(node.depends_on) != 1:
                    continue  # only fuse nodes with exactly one dependency

                parent_id = node.depends_on[0]
                parent = next((n for n in nodes if n.id == parent_id), None)
                if parent is None:
                    continue

                # Check nothing else depends on parent
                other_dependents = [
                    n for n in nodes
                    if parent_id in n.depends_on and n.id != node.id
                ]
                if other_dependents:
                    continue  # parent has multiple children — cannot fuse

                if self._can_fuse(parent, node, default_tier):
                    fused = self._fuse_nodes(parent, node)
                    log.t1_fusions.append({
                        "fused": f"{parent.id} + {node.id}",
                        "result": fused.id,
                        "depth_saved": 1,
                    })
                    # Replace parent and node with fused node
                    nodes = [
                        fused if n.id == parent.id else n
                        for n in nodes
                        if n.id != node.id
                    ]
                    # Update any nodes that depended on the old node
                    for n in nodes:
                        n.depends_on = [
                            fused.id if d == node.id else d
                            for d in n.depends_on
                        ]
                    changed = True
                    break  # restart scan after any fusion

        return nodes

    def _can_fuse(
        self,
        parent: WorkflowNode,
        child: WorkflowNode,
        tier: str,
    ) -> bool:
        """
        Safety guards for T1 fusion.
        Returns True only when fusion is genuinely beneficial and safe.
        """
        child_profile = self.profiler.get(child.node_class, "sonnet")

        # Guard 1: combined prompt must stay under practical fusion limit
        # We use 2,000 tokens as a practical upper bound — beyond this,
        # the fused node becomes too complex for reliable execution.
        combined_tokens = (
            parent.avg_output_tokens
            + child_profile.avg_input_base
        )
        if combined_tokens > 2_000:
            return False

        # Guard 2: don't fuse nodes of the same class
        if parent.node_class == child.node_class:
            return False

        # Guard 3: don't fuse if either node is a reconciliation/reasoning node
        # These nodes need their full context and shouldn't be merged
        complex_classes = {"billing-recon", "cross-recon", "policy-check"}
        if parent.node_class in complex_classes or child.node_class in complex_classes:
            return False

        # Guard 4: don't fuse if parent has no output tokens estimated
        if parent.avg_output_tokens <= 0:
            return False

        # Guard 5: only fuse if depth saving is meaningful
        # (parent must be at depth >= 2 to make fusion worthwhile)
        if parent.depth < 2:
            return False

        return True

    def _fuse_nodes(
        self,
        parent: WorkflowNode,
        child: WorkflowNode,
    ) -> WorkflowNode:
        """Create a fused node from parent + child."""
        return WorkflowNode(
            id=f"{parent.id}_{child.id}",
            node_class=f"{parent.node_class}+{child.node_class}",
            depth=parent.depth,
            depends_on=parent.depends_on,
            required_topics=list(set(parent.required_topics + child.required_topics)),
            namespace=parent.namespace,
            fused_from=[parent.id, child.id],
            avg_output_tokens=child.avg_output_tokens,
        )

    # ── T2: Injection-Aware Reordering ────────────────────────────────────────

    def _apply_t2(
        self,
        nodes: list[WorkflowNode],
        log: RewriteLog,
    ) -> list[WorkflowNode]:
        """
        For parallel nodes (no dependency between them), place
        higher-output nodes LAST to minimize downstream accumulation.
        """
        # Find parallel pairs at the same depth
        depth_groups: dict[int, list[WorkflowNode]] = {}
        for node in nodes:
            depth_groups.setdefault(node.depth, []).append(node)

        reordered = list(nodes)

        for depth, group in depth_groups.items():
            if len(group) < 2:
                continue

            # Check all pairs in this depth group for independence
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    a, b = group[i], group[j]

                    # Independent if neither depends on the other
                    if a.id in b.depends_on or b.id in a.depends_on:
                        continue

                    # T2: place higher-output node last
                    if a.avg_output_tokens > b.avg_output_tokens:
                        # a should come after b — swap their positions
                        idx_a = next(
                            k for k, n in enumerate(reordered) if n.id == a.id
                        )
                        idx_b = next(
                            k for k, n in enumerate(reordered) if n.id == b.id
                        )
                        if idx_a < idx_b:
                            reordered[idx_a], reordered[idx_b] = (
                                reordered[idx_b], reordered[idx_a]
                            )
                            log.t2_reorders.append({
                                "moved_last": a.id,
                                "moved_first": b.id,
                                "depth": depth,
                                "reason": (
                                    f"{a.id} avg_output={a.avg_output_tokens:.0f} > "
                                    f"{b.id} avg_output={b.avg_output_tokens:.0f}"
                                ),
                            })

        return reordered

    # ── T3: Shared Namespace Promotion ───────────────────────────────────────

    def _apply_t3(
        self,
        nodes: list[WorkflowNode],
        log: RewriteLog,
    ) -> list[WorkflowNode]:
        """
        Promote node pairs with ≥60% topic overlap to shared warm namespace.
        Increases warm-hit rate across both nodes without extra cost.
        """
        group_counter = 0

        for i, a in enumerate(nodes):
            for j, b in enumerate(nodes):
                if i >= j:
                    continue
                if not a.required_topics or not b.required_topics:
                    continue

                overlap = self._topic_overlap(
                    a.required_topics, b.required_topics
                )
                if overlap >= SHARED_NAMESPACE_THRESHOLD:
                    group_id = f"shared_group_{group_counter}"
                    group_counter += 1

                    # Promote both to shared namespace
                    nodes[i].namespace = f"shared:{group_id}"
                    nodes[j].namespace = f"shared:{group_id}"

                    log.t3_promotions.append({
                        "nodes": [a.id, b.id],
                        "overlap": round(overlap, 3),
                        "group_id": group_id,
                    })

        return nodes

    @staticmethod
    def _topic_overlap(topics_a: list[str], topics_b: list[str]) -> float:
        """Jaccard-style overlap: |A ∩ B| / |A ∪ B|"""
        set_a, set_b = set(topics_a), set(topics_b)
        if not set_a and not set_b:
            return 0.0
        return len(set_a & set_b) / len(set_a | set_b)

    @staticmethod
    def _recompute_depths(nodes: list[WorkflowNode]) -> list[WorkflowNode]:
        """Recompute depths after fusion may have changed topology."""
        id_to_node = {n.id: n for n in nodes}
        computed: dict[str, int] = {}

        def depth_of(node_id: str) -> int:
            if node_id in computed:
                return computed[node_id]
            node = id_to_node[node_id]
            if not node.depends_on:
                computed[node_id] = 1
            else:
                computed[node_id] = (
                    max(depth_of(d) for d in node.depends_on) + 1
                )
            return computed[node_id]

        for node in nodes:
            node.depth = depth_of(node.id)

        return nodes
