from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from itertools import count
from typing import Any

import pandas as pd

from .scoring import ImportanceScorer, KeywordImportanceScorer


@dataclass(slots=True)
class InteractionRecord:
    interaction_id: int
    timestamp: datetime
    role: str
    content: str
    summary: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)


class WarmMemoryBuffer:
    """
    Pandas-backed in-memory interaction buffer for agent experiments.

    Two usage modes are supported:
    - `recent(limit)`: classic sliding-window retrieval.
    - `relevant(query, limit)`: query-aware retrieval using a pluggable scorer.
    """

    COLUMNS = ["interaction_id", "timestamp", "role", "content", "summary", "tags", "metadata"]

    def __init__(
        self,
        capacity: int = 32,
        scorer: ImportanceScorer | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")

        self.capacity = capacity
        self.scorer = scorer or KeywordImportanceScorer()
        self._id_source = count(1)
        self._frame = pd.DataFrame(columns=self.COLUMNS)

    @property
    def frame(self) -> pd.DataFrame:
        """Return a copy so callers cannot mutate the live buffer implicitly."""
        return self._frame.copy(deep=True)

    def __len__(self) -> int:
        return len(self._frame.index)

    def add(
        self,
        role: str,
        content: str,
        *,
        summary: str = "",
        tags: list[str] | tuple[str, ...] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        record = InteractionRecord(
            interaction_id=next(self._id_source),
            timestamp=datetime.now(timezone.utc),
            role=role,
            content=content,
            summary=summary,
            tags=tuple(tags or ()),
            metadata=dict(metadata or {}),
        )

        row = pd.DataFrame([asdict(record)], columns=self.COLUMNS)
        self._frame = pd.concat([self._frame, row], ignore_index=True)
        self._evict_over_capacity()
        return record.interaction_id

    def recent(self, limit: int = 5) -> pd.DataFrame:
        if limit <= 0:
            return self._frame.head(0).copy(deep=True)

        ordered = self._frame.sort_values("interaction_id", ascending=False).head(limit)
        return ordered.sort_values("interaction_id", ascending=True).reset_index(drop=True)

    def relevant(self, query: str, limit: int = 5) -> pd.DataFrame:
        if limit <= 0 or self._frame.empty:
            return self._frame.head(0).copy(deep=True)

        scored = self._frame.copy(deep=True)
        scored["score"] = scored.apply(lambda row: self.scorer.score(query, row), axis=1)
        scored = scored.sort_values(["score", "interaction_id"], ascending=[False, False]).head(limit)
        return scored.reset_index(drop=True)

    def context_window(self, query: str | None = None, limit: int = 5) -> pd.DataFrame:
        """
        Return either a recent window or a query-aware relevant window.

        This gives callers a single method for switching between fixed-window and
        importance-aware retention policies.
        """

        if query is None or not query.strip():
            return self.recent(limit=limit)
        return self.relevant(query=query, limit=limit)

    def retain_relevant(self, query: str, limit: int | None = None) -> pd.DataFrame:
        """
        Compact the live buffer down to the top relevant rows for a query.

        This is useful when you want a strict "working set" rather than keeping the
        most recent interactions by default.
        """

        target_size = self.capacity if limit is None else limit
        if target_size <= 0:
            self.clear()
            return self._frame.copy(deep=True)

        retained = self.relevant(query=query, limit=target_size).sort_values("interaction_id")
        self._frame = retained[self.COLUMNS].reset_index(drop=True)
        return self.frame

    def clear(self) -> None:
        self._frame = pd.DataFrame(columns=self.COLUMNS)

    def find_index_by_metadata(self, field: str, value: Any) -> int | None:
        """
        Return the integer index of the first row whose `metadata[field]` equals `value`.

        Used by external mutation paths (e.g., key-based stores layered on top of
        the buffer) that need to update or delete a specific logical entry without
        scanning the dataframe themselves.
        """
        if self._frame.empty:
            return None
        for idx, metadata in enumerate(self._frame["metadata"].tolist()):
            if isinstance(metadata, dict) and metadata.get(field) == value:
                return idx
        return None

    def drop_at(self, index: int) -> None:
        """
        Remove the row at the given positional index.

        Pairs with `find_index_by_metadata` to give external code a clean mutation
        path without reaching into private dataframe internals.
        """
        if index < 0 or index >= len(self._frame.index):
            return
        self._frame = self._frame.drop(self._frame.index[index]).reset_index(drop=True)

    def iter_rows(self) -> list[dict[str, Any]]:
        """
        Return a list of row dicts in insertion order. Use this for read-only
        scans (e.g., search) instead of poking at the live dataframe.
        """
        return self._frame.to_dict("records")

    def _evict_over_capacity(self) -> None:
        overflow = len(self._frame.index) - self.capacity
        if overflow <= 0:
            return

        self._frame = (
            self._frame.sort_values("interaction_id", ascending=False)
            .head(self.capacity)
            .sort_values("interaction_id", ascending=True)
            .reset_index(drop=True)
        )
