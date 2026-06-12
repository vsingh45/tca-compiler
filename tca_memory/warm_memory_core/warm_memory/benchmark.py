from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd

from .buffer import WarmMemoryBuffer
from .workload import ScenarioTurn, default_workload


@dataclass(slots=True)
class BenchmarkConfig:
    capacity: int = 8
    top_k: int = 5
    long_term_limit: int = 8
    warm_hit_threshold: float = 0.34
    long_term_latency_ms: float = 8.0
    llm_base_latency_ms: float = 35.0
    llm_latency_per_token_ms: float = 0.12
    prompt_build_per_item_ms: float = 0.08


@dataclass(slots=True)
class BenchmarkResult:
    name: str
    turn_log: pd.DataFrame
    summary: dict[str, float]


def _estimate_tokens(items: list[str]) -> int:
    return sum(max(1, len(item.split())) for item in items)


def _summarize_turn_log(name: str, turn_log: pd.DataFrame) -> BenchmarkResult:
    summary = {
        "turns": float(len(turn_log)),
        "warm_hit_rate": float(turn_log["warm_hit"].mean()),
        "fallback_rate": float(turn_log["fallback_used"].mean()),
        "avg_prompt_tokens": float(turn_log["prompt_tokens"].mean()),
        "avg_end_to_end_ms": float(turn_log["end_to_end_ms"].mean()),
        "p95_end_to_end_ms": float(turn_log["end_to_end_ms"].quantile(0.95)),
        "answer_accuracy": float(turn_log["answer_correct"].mean()),
        "memory_precision_at_k": float(turn_log["memory_precision_at_k"].mean()),
        "repeated_tool_calls": float(turn_log["repeated_tool_call"].sum()),
    }
    return BenchmarkResult(name=name, turn_log=turn_log, summary=summary)


def _build_prompt_metrics(config: BenchmarkConfig, retrieved_contents: list[str]) -> tuple[int, float]:
    tokens = _estimate_tokens(retrieved_contents)
    prompt_build_ms = len(retrieved_contents) * config.prompt_build_per_item_ms
    llm_ms = config.llm_base_latency_ms + (tokens * config.llm_latency_per_token_ms)
    return tokens, prompt_build_ms + llm_ms


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(row[column]) for column in columns) + " |"
        for _, row in frame.iterrows()
    ]
    return "\n".join([header, divider, *rows])


def _write_memory(memory: WarmMemoryBuffer, turn: ScenarioTurn, response: str) -> None:
    memory.add("user", turn.query, metadata={"scenario_id": turn.turn_id, "topic": turn.topic})
    memory.add("assistant", response, metadata={"scenario_id": turn.turn_id, "topic": turn.topic})


def _retrieve_recent(memory: WarmMemoryBuffer, limit: int) -> pd.DataFrame:
    recent = memory.recent(limit=limit).copy(deep=True)
    if not recent.empty:
        recent["score"] = 1.0
    return recent


def _retrieve_relevant(memory: WarmMemoryBuffer, query: str, limit: int) -> pd.DataFrame:
    return memory.relevant(query=query, limit=limit)


def _run_strategy(
    name: str,
    config: BenchmarkConfig,
    turns: list[ScenarioTurn],
) -> BenchmarkResult:
    warm = WarmMemoryBuffer(capacity=config.capacity)
    long_term = WarmMemoryBuffer(capacity=max(len(turns) * 2, config.capacity * 4))
    rows: list[dict[str, Any]] = []
    resolved_topics: set[str] = set()

    for turn in turns:
        lookup_start = perf_counter()
        if name == "recency":
            warm_candidates = _retrieve_recent(warm, config.top_k)
        else:
            warm_candidates = _retrieve_relevant(warm, turn.query, config.top_k)
        warm_lookup_ms = (perf_counter() - lookup_start) * 1000

        best_score = float(warm_candidates["score"].max()) if "score" in warm_candidates.columns and not warm_candidates.empty else 0.0
        warm_hit = bool(not warm_candidates.empty and (name == "recency" or best_score >= config.warm_hit_threshold))

        fallback_used = False
        retrieval_ms = warm_lookup_ms
        retrieved = warm_candidates

        if name == "fallback" and not warm_hit:
            fallback_used = True
            retrieval_ms += config.long_term_latency_ms
            retrieved = long_term.relevant(turn.query, limit=config.long_term_limit)

        retrieved_contents = retrieved["content"].tolist() if not retrieved.empty else []
        prompt_tokens, generation_ms = _build_prompt_metrics(config, retrieved_contents)
        end_to_end_ms = retrieval_ms + generation_ms

        relevant_retrieved = {
            str(meta.get("topic"))
            for meta in retrieved["metadata"].tolist()
            if isinstance(meta, dict) and meta.get("topic")
        }
        expected = set(turn.required_topics)
        memory_precision = len(expected & relevant_retrieved) / max(len(relevant_retrieved), 1) if relevant_retrieved else 0.0
        answer_correct = expected.issubset(relevant_retrieved)

        repeated_tool_call = int(turn.topic in resolved_topics and not answer_correct)
        if answer_correct:
            resolved_topics.add(turn.topic)

        response = f"Response for {turn.topic}: {'correct' if answer_correct else 'incomplete'}"
        _write_memory(warm, turn, response)
        _write_memory(long_term, turn, response)

        if name == "relevance":
            warm.retain_relevant(turn.query, limit=config.capacity)

        rows.append(
            {
                "turn_id": turn.turn_id,
                "topic": turn.topic,
                "query": turn.query,
                "warm_lookup_ms": warm_lookup_ms,
                "warm_hit": warm_hit,
                "fallback_used": fallback_used,
                "retrieval_ms": retrieval_ms,
                "prompt_tokens": prompt_tokens,
                "end_to_end_ms": end_to_end_ms,
                "answer_correct": answer_correct,
                "memory_precision_at_k": memory_precision,
                "repeated_tool_call": repeated_tool_call,
            }
        )

    turn_log = pd.DataFrame(rows)
    return _summarize_turn_log(name, turn_log)


def _render_report(config: BenchmarkConfig, results: list[BenchmarkResult]) -> str:
    summary_frame = pd.DataFrame(
        [
            {"strategy": result.name, **result.summary}
            for result in results
        ]
    )
    best_latency = summary_frame.sort_values("avg_end_to_end_ms").iloc[0]["strategy"]
    best_accuracy = summary_frame.sort_values("answer_accuracy", ascending=False).iloc[0]["strategy"]
    best_tokens = summary_frame.sort_values("avg_prompt_tokens").iloc[0]["strategy"]

    lines = [
        "# WarmMemory Benchmark Report",
        "",
        "## Configuration",
        "",
        f"- capacity: {config.capacity}",
        f"- top_k: {config.top_k}",
        f"- long_term_limit: {config.long_term_limit}",
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
        "## Interpretation",
        "",
        "- `recency` shows the baseline cost of always trusting the latest interactions.",
        "- `relevance` shows the effect of ranking and retaining the hottest working set.",
        "- `fallback` shows a two-tier memory architecture where long-term retrieval is only used on warm misses.",
    ]
    return "\n".join(lines) + "\n"


def run_benchmark(
    *,
    config: BenchmarkConfig | None = None,
    report_path: str | Path | None = None,
) -> dict[str, BenchmarkResult]:
    active_config = config or BenchmarkConfig()
    turns = default_workload()
    results = {
        name: _run_strategy(name, active_config, turns)
        for name in ("recency", "relevance", "fallback")
    }

    if report_path is not None:
        report_text = _render_report(active_config, list(results.values()))
        target = Path(report_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(report_text, encoding="utf-8")

    return results
