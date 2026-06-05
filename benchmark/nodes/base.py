"""
BaseNode: shared execution logic for all benchmark nodes.
"""
from __future__ import annotations

import sqlite3
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import anthropic

from tca_compiler.pricing import Tier, TIER_MODELS, cost_inference, cost_memory_injection
from tca_compiler.record import TCARecord
from tca_compiler.token_counter import AnthropicTokenCounter
from tca_compiler.cost_profiler import CostProfiler
from tca_memory.store import SharedNamespaceStore


DB_DIR = Path(__file__).parent.parent / "databases"


@dataclass
class NodeResult:
    node_id:    str
    node_class: str
    success:    bool
    answer:     str
    structured: dict
    record:     TCARecord
    error:      Optional[str] = None


class BaseNode(ABC):
    node_class: str = "base"

    def __init__(
        self,
        client: anthropic.Anthropic,
        profiler: CostProfiler,
        workflow_id: str,
        node_id: str,
        tier: Tier,
        memory_strategy: str,
        condition: str,
        max_tokens: int = 300,
        shared_capacity: int = 32,
    ) -> None:
        self.client          = client
        self.profiler        = profiler
        self.workflow_id     = workflow_id
        self.node_id         = node_id
        self.tier            = tier
        self.memory_strategy = memory_strategy
        self.condition       = condition
        self.max_tokens      = max_tokens
        self.counter         = AnthropicTokenCounter(client=client)
        self.memory          = SharedNamespaceStore(
            workflow_id=workflow_id,
            agent_id=node_id,
            shared_capacity=shared_capacity,
        )

    def get_db(self, db_name: str) -> sqlite3.Connection:
        db_path = DB_DIR / f"{db_name}.db"
        if not db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def retrieve_memory(self, query: str, limit: int = 5) -> list[str]:
        if self.memory_strategy == "full-history":
            rows = self.memory.shared.iter_rows()
            return [r["content"] for r in rows[-limit:]]
        elif self.memory_strategy in ("warm-isolated", "warm-shared", "vector-only"):
            results = self.memory.retrieve(query=query, limit=limit)
            if results.empty:
                return []
            return results["content"].tolist()
        return []

    def write_memory(self, content: str, token_count: int = 0, metadata: dict | None = None) -> None:
        share = self.memory_strategy == "warm-shared"
        self.memory.add(
            role="assistant",
            content=content,
            metadata={"node_id": self.node_id, **(metadata or {})},
            token_count=token_count,
            tier=self.tier,
            share=share,
        )

    def execute(self, task: dict, workflow_depth: int, seed: int = 42) -> NodeResult:
        query   = task["query"]
        db_name = task.get("db", "billing")

        # Step 1: retrieve memory
        t0 = time.perf_counter()
        memory_context = self.retrieve_memory(query)
        retrieval_ms = (time.perf_counter() - t0) * 1000
        warm_hit     = len(memory_context) > 0 and self.memory_strategy != "full-history"
        fallback_used = not warm_hit and self.memory_strategy != "full-history"

        # Step 2: build prompt
        try:
            db_conn     = self.get_db(db_name)
            base_prompt = self.build_prompt(query, memory_context, db_conn)
        except Exception as e:
            return self._error_result(task, workflow_depth, seed, str(e))

        # Step 3: exact token attribution (free)
        base_tokens, injection_tokens = self.counter.count_injection(
            base_prompt, memory_context, tier=self.tier
        )

        # Step 4: real LLM call
        try:
            full_prompt = base_prompt
            if memory_context:
                full_prompt += "\n\nContext from prior agents:\n" + "\n".join(
                    f"[{i+1}] {c}" for i, c in enumerate(memory_context)
                )
            t1 = time.perf_counter()
            response = self.client.messages.create(
                model=TIER_MODELS[self.tier],
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": full_prompt}],
            )
            end_to_end_ms = (time.perf_counter() - t1) * 1000 + retrieval_ms
        except Exception as e:
            return self._error_result(task, workflow_depth, seed, str(e))

        usage         = response.usage
        actual_input  = usage.input_tokens
        actual_output = usage.output_tokens
        cache_read    = getattr(usage, "cache_read_input_tokens", 0) or 0

        # Step 5: TCA attribution
        inf_cost  = cost_inference(
            input_tokens=max(0, actual_input - injection_tokens),
            output_tokens=actual_output,
            tier=self.tier,
            cache_read_tokens=cache_read,
        )
        inj_cost  = cost_memory_injection(injection_tokens, tier=self.tier)
        total_tca = inf_cost + inj_cost

        # Step 6: parse + evaluate
        answer_text = response.content[0].text
        try:
            structured = self.parse_result(answer_text, query, db_conn)
        except Exception:
            structured = {"raw": answer_text}

        answer_correct, recall = self.evaluate(structured, answer_text, task)

        # Step 7: TCARecord
        record = TCARecord(
            task_id=task["task_id"],
            workflow_id=self.workflow_id,
            node_id=self.node_id,
            node_class=self.node_class,
            seed=seed,
            tier=self.tier,
            memory_strategy=self.memory_strategy,
            condition=self.condition,
            workflow_depth=workflow_depth,
            workflow_category=task["category"],
            base_input_tokens=max(0, actual_input - injection_tokens),
            memory_tokens_injected=injection_tokens,
            output_tokens=actual_output,
            cache_read_tokens=cache_read,
            cache_creation_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
            cost_inference=inf_cost,
            cost_injection=inj_cost,
            cost_miss_penalty=0.0,
            cost_accum=0.0,
            cost_total=total_tca,
            warm_hit=warm_hit,
            fallback_used=fallback_used,
            retrieval_latency_ms=retrieval_ms,
            answer_correct=answer_correct,
            required_topics_recall=recall,
            end_to_end_ms=end_to_end_ms,
            model_id=TIER_MODELS[self.tier],
        )

        # Step 8: write to memory
        self.write_memory(
            content=f"{self.node_class} result: {answer_text[:500]}",
            token_count=actual_output,
        )

        # Step 9: update profiler
        self.profiler.update(
            node_class=self.node_class,
            tier=self.tier,
            observed_input_base=max(0, actual_input - injection_tokens),
            observed_output=actual_output,
            observed_inject=injection_tokens,
            observed_accuracy=recall,
        )

        return NodeResult(
            node_id=self.node_id,
            node_class=self.node_class,
            success=answer_correct,
            answer=answer_text,
            structured=structured,
            record=record,
        )

    def evaluate(self, structured: dict, answer_text: str, task: dict) -> tuple[bool, float]:
        required    = task.get("required_topics", [])
        if not required:
            return True, 1.0
        answer_lower = answer_text.lower()
        hits  = sum(
            1 for t in required
            if t.lower().replace("-", " ") in answer_lower or t.lower() in answer_lower
        )
        recall  = hits / len(required)
        correct = recall >= 0.6
        return correct, recall

    def _error_result(self, task: dict, depth: int, seed: int, error_msg: str) -> NodeResult:
        zero = TCARecord(
            task_id=task["task_id"], workflow_id=self.workflow_id,
            node_id=self.node_id, node_class=self.node_class, seed=seed,
            tier=self.tier, memory_strategy=self.memory_strategy,
            condition=self.condition, workflow_depth=depth,
            workflow_category=task.get("category", "unknown"),
            base_input_tokens=0, memory_tokens_injected=0, output_tokens=0,
            cache_read_tokens=0, cache_creation_tokens=0,
            cost_inference=0.0, cost_injection=0.0,
            cost_miss_penalty=0.0, cost_accum=0.0, cost_total=0.0,
            warm_hit=False, fallback_used=False, retrieval_latency_ms=0.0,
            answer_correct=False, required_topics_recall=0.0,
            end_to_end_ms=0.0, model_id=TIER_MODELS[self.tier],
            error=error_msg,
        )
        return NodeResult(
            node_id=self.node_id, node_class=self.node_class,
            success=False, answer="", structured={},
            record=zero, error=error_msg,
        )

    @abstractmethod
    def build_prompt(self, query: str, context: list[str], db_conn: sqlite3.Connection) -> str: ...

    @abstractmethod
    def parse_result(self, response_text: str, query: str, db_conn: sqlite3.Connection) -> dict: ...
