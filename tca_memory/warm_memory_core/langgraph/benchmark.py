"""
LangGraph-store benchmark.

Compares three memory strategies for a multi-turn agent, all driven through the
LangGraph BaseStore API so the comparison is apples-to-apples:

- `full-history`  : pass every prior exchange every turn (no retrieval)
- `vector-only`   : LangGraph InMemoryStore with embedding index for retrieval
- `warm-fallback` : WarmStore for hot lookups, falls back to the vector store on misses

Fully synthetic by default (DeterministicFakeEmbedding, no API key).
Pass your own LangChain Embeddings to compare against real semantic search.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import pandas as pd
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings
from langgraph.store.memory import InMemoryStore

from ..workload import ScenarioTurn, default_workload
from .embeddings import EmbeddingsImportanceScorer
from .store import WarmStore


@dataclass(slots=True)
class LangGraphBenchmarkConfig:
    warm_capacity: int = 8
    top_k: int = 5
    embedding_dims: int = 64
    warm_hit_threshold: float = 0.34
    long_term_latency_ms: float = 8.0
    llm_base_latency_ms: float = 35.0
    llm_latency_per_token_ms: float = 0.12
    prompt_build_per_item_ms: float = 0.08


@dataclass(slots=True)
class LangGraphBenchmarkResult:
    name: str
    turn_log: pd.DataFrame
    summary: dict[str, float]


def _estimate_tokens(items: Iterable[str]) -> int:
    return sum(max(1, len(item.split())) for item in items)


def _build_prompt_metrics(
    config: LangGraphBenchmarkConfig, retrieved_contents: list[str]
) -> tuple[int, float]:
    tokens = _estimate_tokens(retrieved_contents)
    prompt_build_ms = len(retrieved_contents) * config.prompt_build_per_item_ms
    llm_ms = config.llm_base_latency_ms + (tokens * config.llm_latency_per_token_ms)
    return tokens, prompt_build_ms + llm_ms


def _summarize(name: str, turn_log: pd.DataFrame) -> LangGraphBenchmarkResult:
    summary = {
        "turns": float(len(turn_log)),
        "warm_hit_rate": float(turn_log["warm_hit"].mean()),
        "fallback_rate": float(turn_log["fallback_used"].mean()),
        "avg_prompt_tokens": float(turn_log["prompt_tokens"].mean()),
        "avg_end_to_end_ms": float(turn_log["end_to_end_ms"].mean()),
        "p95_end_to_end_ms": float(turn_log["end_to_end_ms"].quantile(0.95)),
        "answer_accuracy": float(turn_log["answer_correct"].mean()),
        "memory_precision_at_k": float(turn_log["memory_precision_at_k"].mean()),
    }
    return LangGraphBenchmarkResult(name=name, turn_log=turn_log, summary=summary)


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(row[column]) for column in columns) + " |"
        for _, row in frame.iterrows()
    ]
    return "\n".join([header, divider, *rows])


def _evaluate(
    retrieved_topics: set[str], turn: ScenarioTurn
) -> tuple[float, bool]:
    expected = set(turn.required_topics)
    if not retrieved_topics:
        return 0.0, False
    precision = len(expected & retrieved_topics) / max(len(retrieved_topics), 1)
    answer_correct = expected.issubset(retrieved_topics)
    return precision, answer_correct


def _topics_from_items(items: Iterable[Any]) -> set[str]:
    out: set[str] = set()
    for item in items:
        value = getattr(item, "value", None) or {}
        topic = value.get("topic")
        if topic:
            out.add(str(topic))
    return out


def _contents_from_items(items: Iterable[Any]) -> list[str]:
    out: list[str] = []
    for item in items:
        value = getattr(item, "value", None) or {}
        text = value.get("user", "") or value.get("text", "")
        if text:
            out.append(str(text))
    return out


def _write_full_history_log(
    config: LangGraphBenchmarkConfig, turns: list[ScenarioTurn]
) -> LangGraphBenchmarkResult:
    history: list[str] = []
    history_topics: list[str] = []
    rows: list[dict[str, Any]] = []

    for turn in turns:
        retrieved_contents = list(history)
        retrieved_topics = set(history_topics)
        prompt_tokens, generation_ms = _build_prompt_metrics(config, retrieved_contents)
        precision, answer_correct = _evaluate(retrieved_topics, turn)
        rows.append(
            {
                "turn_id": turn.turn_id,
                "topic": turn.topic,
                "warm_hit": False,
                "fallback_used": False,
                "prompt_tokens": prompt_tokens,
                "end_to_end_ms": generation_ms,
                "memory_precision_at_k": precision,
                "answer_correct": answer_correct,
            }
        )
        history.append(turn.query)
        history_topics.append(turn.topic)

    return _summarize("full-history", pd.DataFrame(rows))


def _run_vector_only(
    config: LangGraphBenchmarkConfig,
    turns: list[ScenarioTurn],
    embeddings: Embeddings,
    namespace: tuple[str, ...],
) -> LangGraphBenchmarkResult:
    vector = InMemoryStore(
        index={"embed": embeddings, "dims": config.embedding_dims, "fields": ["text"]}
    )
    rows: list[dict[str, Any]] = []

    for turn in turns:
        start = perf_counter()
        hits = vector.search(namespace, query=turn.query, limit=config.top_k)
        retrieval_ms = (perf_counter() - start) * 1000 + config.long_term_latency_ms

        retrieved_topics = _topics_from_items(hits)
        retrieved_contents = _contents_from_items(hits)
        prompt_tokens, generation_ms = _build_prompt_metrics(config, retrieved_contents)
        precision, answer_correct = _evaluate(retrieved_topics, turn)

        rows.append(
            {
                "turn_id": turn.turn_id,
                "topic": turn.topic,
                "warm_hit": False,
                "fallback_used": True,
                "prompt_tokens": prompt_tokens,
                "end_to_end_ms": retrieval_ms + generation_ms,
                "memory_precision_at_k": precision,
                "answer_correct": answer_correct,
            }
        )
        vector.put(
            namespace,
            f"turn-{turn.turn_id}",
            {"text": turn.query, "topic": turn.topic},
        )

    return _summarize("vector-only", pd.DataFrame(rows))


def _run_warm_fallback(
    config: LangGraphBenchmarkConfig,
    turns: list[ScenarioTurn],
    embeddings: Embeddings,
    namespace: tuple[str, ...],
) -> LangGraphBenchmarkResult:
    warm = WarmStore(capacity=config.warm_capacity)
    vector = InMemoryStore(
        index={"embed": embeddings, "dims": config.embedding_dims, "fields": ["text"]}
    )
    rows: list[dict[str, Any]] = []

    for turn in turns:
        start = perf_counter()
        warm_hits = warm.search(namespace, query=turn.query, limit=config.top_k)
        warm_lookup_ms = (perf_counter() - start) * 1000

        best_score = max((h.score or 0.0 for h in warm_hits), default=0.0)
        warm_hit = bool(warm_hits and best_score >= config.warm_hit_threshold)

        fallback_used = False
        retrieval_ms = warm_lookup_ms
        chosen = warm_hits
        if not warm_hit:
            fallback_used = True
            retrieval_ms += config.long_term_latency_ms
            chosen = vector.search(namespace, query=turn.query, limit=config.top_k)

        retrieved_topics = _topics_from_items(chosen)
        retrieved_contents = _contents_from_items(chosen)
        prompt_tokens, generation_ms = _build_prompt_metrics(config, retrieved_contents)
        precision, answer_correct = _evaluate(retrieved_topics, turn)

        rows.append(
            {
                "turn_id": turn.turn_id,
                "topic": turn.topic,
                "warm_hit": warm_hit,
                "fallback_used": fallback_used,
                "prompt_tokens": prompt_tokens,
                "end_to_end_ms": retrieval_ms + generation_ms,
                "memory_precision_at_k": precision,
                "answer_correct": answer_correct,
            }
        )
        warm.put(
            namespace,
            f"turn-{turn.turn_id}",
            {"text": turn.query, "topic": turn.topic, "user": turn.query},
        )
        vector.put(
            namespace,
            f"turn-{turn.turn_id}",
            {"text": turn.query, "topic": turn.topic, "user": turn.query},
        )

    return _summarize("warm-fallback", pd.DataFrame(rows))


def _render_report(
    config: LangGraphBenchmarkConfig, results: list[LangGraphBenchmarkResult]
) -> str:
    summary_frame = pd.DataFrame(
        [{"strategy": r.name, **r.summary} for r in results]
    )
    best_latency = summary_frame.sort_values("avg_end_to_end_ms").iloc[0]["strategy"]
    best_accuracy = summary_frame.sort_values("answer_accuracy", ascending=False).iloc[0]["strategy"]
    best_tokens = summary_frame.sort_values("avg_prompt_tokens").iloc[0]["strategy"]

    lines = [
        "# WarmMemory + LangGraph Benchmark",
        "",
        "Compares three retrieval strategies driven through the LangGraph BaseStore API.",
        "",
        "## Configuration",
        "",
        f"- warm_capacity: {config.warm_capacity}",
        f"- top_k: {config.top_k}",
        f"- embedding_dims: {config.embedding_dims}",
        f"- warm_hit_threshold: {config.warm_hit_threshold}",
        "",
        "## Summary",
        "",
        _markdown_table(summary_frame),
        "",
        "## Readout",
        "",
        f"- Lowest average latency: `{best_latency}`",
        f"- Highest answer accuracy: `{best_accuracy}`",
        f"- Smallest prompt footprint: `{best_tokens}`",
        "",
        "## Strategies",
        "",
        "- `full-history` is the naive baseline: every prior turn is in the prompt.",
        "- `vector-only` uses LangGraph's InMemoryStore with an embedding index.",
        "- `warm-fallback` puts WarmStore in front of the vector store; the vector store",
        "  is only consulted when warm relevance falls below `warm_hit_threshold`.",
    ]
    return "\n".join(lines) + "\n"


def run_langgraph_benchmark(
    *,
    config: LangGraphBenchmarkConfig | None = None,
    embeddings: Embeddings | None = None,
    namespace: tuple[str, ...] = ("bench", "user"),
    report_path: str | Path | None = None,
) -> dict[str, LangGraphBenchmarkResult]:
    active_config = config or LangGraphBenchmarkConfig()
    active_embeddings = embeddings or DeterministicFakeEmbedding(
        size=active_config.embedding_dims
    )
    turns = default_workload()

    results = {
        "full-history": _write_full_history_log(active_config, turns),
        "vector-only": _run_vector_only(active_config, turns, active_embeddings, namespace),
        "warm-fallback": _run_warm_fallback(active_config, turns, active_embeddings, namespace),
    }

    if report_path is not None:
        report_text = _render_report(active_config, list(results.values()))
        target = Path(report_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(report_text, encoding="utf-8")

    return results


__all__ = [
    "LangGraphBenchmarkConfig",
    "LangGraphBenchmarkResult",
    "run_langgraph_benchmark",
    "EmbeddingsImportanceScorer",
]
