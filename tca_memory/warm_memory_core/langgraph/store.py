from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

from langgraph.store.base import (
    BaseStore,
    GetOp,
    Item,
    ListNamespacesOp,
    MatchCondition,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
)

from ..buffer import WarmMemoryBuffer
from ..scoring import ImportanceScorer, KeywordImportanceScorer


_STORE_KEY = "__store_key__"
_STORE_VALUE = "__store_value__"
_STORE_CREATED_AT = "__store_created_at__"
_STORE_UPDATED_AT = "__store_updated_at__"


def _apply_operator(value: Any, operator: str, op_value: Any) -> bool:
    """
    Apply a single MongoDB-style filter operator.

    Matches LangGraph's `InMemoryStore` semantics for `$eq` / `$ne` (direct
    equality, any type) and `$gt` / `$gte` / `$lt` / `$lte` (float coercion).
    Diverges in one place: when either side cannot be coerced to float (e.g.,
    a missing field on a row), comparison operators return `False` rather
    than raising `TypeError`. This is the more permissive behavior — rows
    that don't carry the field simply do not match the comparison, which
    matches what most users expect from a filter.
    """
    if operator == "$eq":
        return value == op_value
    if operator == "$ne":
        return value != op_value
    if operator in ("$gt", "$gte", "$lt", "$lte"):
        try:
            a = float(value)
            b = float(op_value)
        except (TypeError, ValueError):
            return False
        if operator == "$gt":
            return a > b
        if operator == "$gte":
            return a >= b
        if operator == "$lt":
            return a < b
        return a <= b
    raise ValueError(f"Unsupported filter operator: {operator}")


def _matches_filter(value: Mapping[str, Any], filter_spec: Mapping[str, Any] | None) -> bool:
    if not filter_spec:
        return True
    for field_name, condition in filter_spec.items():
        actual = value.get(field_name)
        if isinstance(condition, Mapping):
            for op_name, expected in condition.items():
                if not _apply_operator(actual, op_name, expected):
                    return False
        else:
            if actual != condition:
                return False
    return True


def _namespace_matches(
    namespace: tuple[str, ...],
    condition: MatchCondition,
) -> bool:
    path = condition.path
    if condition.match_type == "prefix":
        if len(path) > len(namespace):
            return False
        for actual, expected in zip(namespace[: len(path)], path):
            if expected != "*" and actual != expected:
                return False
        return True
    if condition.match_type == "suffix":
        if len(path) > len(namespace):
            return False
        for actual, expected in zip(namespace[-len(path):], path):
            if expected != "*" and actual != expected:
                return False
        return True
    raise ValueError(f"Unknown match_type: {condition.match_type}")


def _starts_with_prefix(namespace: tuple[str, ...], prefix: tuple[str, ...]) -> bool:
    if len(prefix) > len(namespace):
        return False
    return namespace[: len(prefix)] == prefix


def _extract_search_text(value: Mapping[str, Any]) -> str:
    """Flatten a value dict into a single string for keyword/embedding scoring."""
    parts: list[str] = []
    for field, sub_value in value.items():
        if isinstance(sub_value, str):
            parts.append(sub_value)
        elif isinstance(sub_value, (int, float, bool)):
            parts.append(str(sub_value))
        else:
            try:
                parts.append(json.dumps(sub_value, default=str))
            except (TypeError, ValueError):
                parts.append(str(sub_value))
    return "\n".join(parts)


class WarmStore(BaseStore):
    """
    LangGraph BaseStore backed by per-namespace WarmMemoryBuffers.

    Each top-level namespace gets its own bounded warm buffer. Writes that exceed
    `capacity` evict the oldest entries within that namespace, so multi-user agents
    naturally get per-user warm memory without cross-tenant eviction.

    Search supports:
    - exact and operator-based filters ($eq, $ne, $gt, $gte, $lt, $lte) over `value`
    - natural-language `query` ranked by a pluggable ImportanceScorer
      (defaults to KeywordImportanceScorer; pass EmbeddingsImportanceScorer for
      semantic search)

    WarmStore is not thread-safe. The default ImportanceScorer (`KeywordImportanceScorer`)
    has no shared state; `EmbeddingsImportanceScorer` has a bounded LRU cache. Wrap
    calls in a lock if you share a single store across threads.
    """

    __slots__ = ("capacity", "scorer", "_buffers", "_key_counters")

    def __init__(
        self,
        *,
        capacity: int = 32,
        scorer: ImportanceScorer | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.scorer = scorer or KeywordImportanceScorer()
        self._buffers: dict[tuple[str, ...], WarmMemoryBuffer] = {}
        self._key_counters: dict[tuple[str, ...], int] = {}

    def next_key(self, namespace: tuple[str, ...], *, prefix: str = "exchange-") -> str:
        """Return a fresh, monotonically-increasing key within a namespace.

        Unlike `len(buffer) + 1`, this counter is *never* reset by eviction —
        so keys remain unique across the entire lifetime of the namespace
        even after capacity-bounded eviction removes older entries.

        Useful when writing a stream of items (e.g., conversation turns) into
        a namespace where you don't have a natural domain key.
        """
        self._key_counters[namespace] = self._key_counters.get(namespace, 0) + 1
        return f"{prefix}{self._key_counters[namespace]}"

    def _buffer_for(self, namespace: tuple[str, ...]) -> WarmMemoryBuffer:
        existing = self._buffers.get(namespace)
        if existing is not None:
            return existing
        buffer = WarmMemoryBuffer(capacity=self.capacity, scorer=self.scorer)
        self._buffers[namespace] = buffer
        return buffer

    def _row_to_item(self, namespace: tuple[str, ...], row: Mapping[str, Any]) -> Item:
        metadata = row["metadata"] if isinstance(row.get("metadata"), dict) else {}
        return Item(
            value=dict(metadata.get(_STORE_VALUE) or {}),
            key=str(metadata.get(_STORE_KEY, "")),
            namespace=namespace,
            created_at=metadata.get(_STORE_CREATED_AT) or datetime.now(timezone.utc),
            updated_at=metadata.get(_STORE_UPDATED_AT) or datetime.now(timezone.utc),
        )

    def _row_to_search_item(
        self,
        namespace: tuple[str, ...],
        row: Mapping[str, Any],
        score: float | None,
    ) -> SearchItem:
        metadata = row["metadata"] if isinstance(row.get("metadata"), dict) else {}
        return SearchItem(
            namespace=namespace,
            key=str(metadata.get(_STORE_KEY, "")),
            value=dict(metadata.get(_STORE_VALUE) or {}),
            created_at=metadata.get(_STORE_CREATED_AT) or datetime.now(timezone.utc),
            updated_at=metadata.get(_STORE_UPDATED_AT) or datetime.now(timezone.utc),
            score=score,
        )

    def _do_get(self, op: GetOp) -> Item | None:
        buffer = self._buffers.get(op.namespace)
        if buffer is None:
            return None
        idx = buffer.find_index_by_metadata(_STORE_KEY, op.key)
        if idx is None:
            return None
        rows = buffer.iter_rows()
        return self._row_to_item(op.namespace, rows[idx])

    def _do_put(self, op: PutOp) -> None:
        if op.value is None:
            buffer = self._buffers.get(op.namespace)
            if buffer is None:
                return
            idx = buffer.find_index_by_metadata(_STORE_KEY, op.key)
            if idx is None:
                return
            buffer.drop_at(idx)
            return

        buffer = self._buffer_for(op.namespace)
        now = datetime.now(timezone.utc)
        existing_idx = buffer.find_index_by_metadata(_STORE_KEY, op.key)
        created_at = now
        if existing_idx is not None:
            old_row = buffer.iter_rows()[existing_idx]
            old_metadata = old_row.get("metadata") if isinstance(old_row, dict) else None
            if isinstance(old_metadata, dict):
                created_at = old_metadata.get(_STORE_CREATED_AT) or now
            buffer.drop_at(existing_idx)

        content = _extract_search_text(op.value)
        buffer.add(
            role="item",
            content=content,
            metadata={
                _STORE_KEY: op.key,
                _STORE_VALUE: dict(op.value),
                _STORE_CREATED_AT: created_at,
                _STORE_UPDATED_AT: now,
            },
        )

    def _do_search(self, op: SearchOp) -> list[SearchItem]:
        limit = max(0, op.limit)
        offset = max(0, op.offset)
        if limit == 0:
            return []

        matched_namespaces = [
            ns for ns in self._buffers.keys() if _starts_with_prefix(ns, op.namespace_prefix)
        ]
        candidates: list[tuple[tuple[str, ...], dict[str, Any], float | None]] = []
        has_query = bool(op.query and op.query.strip())

        for namespace in matched_namespaces:
            buffer = self._buffers[namespace]
            if has_query:
                ranked = buffer.relevant(op.query, limit=len(buffer))
                for _, row in ranked.iterrows():
                    metadata = row["metadata"] if isinstance(row.get("metadata"), dict) else {}
                    value = dict(metadata.get(_STORE_VALUE) or {})
                    if not _matches_filter(value, op.filter):
                        continue
                    candidates.append((namespace, row.to_dict(), float(row.get("score") or 0.0)))
            else:
                for row in buffer.iter_rows():
                    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
                    value = dict(metadata.get(_STORE_VALUE) or {})
                    if not _matches_filter(value, op.filter):
                        continue
                    candidates.append((namespace, row, None))

        if has_query:
            candidates.sort(key=lambda triple: (triple[2] or 0.0), reverse=True)
        else:
            def _updated_at(triple: tuple[tuple[str, ...], dict[str, Any], float | None]) -> datetime:
                metadata = triple[1].get("metadata") if isinstance(triple[1].get("metadata"), dict) else {}
                return metadata.get(_STORE_UPDATED_AT) or datetime.min.replace(tzinfo=timezone.utc)
            candidates.sort(key=_updated_at, reverse=True)

        sliced = candidates[offset : offset + limit]
        return [self._row_to_search_item(ns, row, score) for ns, row, score in sliced]

    def _do_list_namespaces(self, op: ListNamespacesOp) -> list[tuple[str, ...]]:
        limit = max(0, op.limit)
        offset = max(0, op.offset)
        if limit == 0:
            return []

        namespaces = list(self._buffers.keys())
        if op.match_conditions:
            namespaces = [
                ns for ns in namespaces if all(_namespace_matches(ns, cond) for cond in op.match_conditions)
            ]

        if op.max_depth is not None:
            truncated = {ns[: op.max_depth] for ns in namespaces}
            namespaces = sorted(truncated)
        else:
            namespaces = sorted(set(namespaces))

        return namespaces[offset : offset + limit]

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        ops_list: Sequence[Op] = list(ops)
        results: list[Result] = []
        for op in ops_list:
            if isinstance(op, GetOp):
                results.append(self._do_get(op))
            elif isinstance(op, PutOp):
                self._do_put(op)
                results.append(None)
            elif isinstance(op, SearchOp):
                results.append(self._do_search(op))
            elif isinstance(op, ListNamespacesOp):
                results.append(self._do_list_namespaces(op))
            else:
                raise TypeError(f"Unsupported op type: {type(op).__name__}")
        return results

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        # Materialize once here; self.batch will not re-copy a list.
        ops_list = list(ops)
        return await asyncio.to_thread(self.batch, ops_list)

    def namespaces(self) -> list[tuple[str, ...]]:
        return sorted(self._buffers.keys())

    def size(self, namespace: tuple[str, ...]) -> int:
        buffer = self._buffers.get(namespace)
        return 0 if buffer is None else len(buffer)
