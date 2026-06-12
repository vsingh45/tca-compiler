from __future__ import annotations

import math
from collections import OrderedDict
from typing import Any, Mapping

from langchain_core.embeddings import Embeddings

from ..scoring import ImportanceScorer


_DEFAULT_CACHE_SIZE = 4096


class EmbeddingsImportanceScorer(ImportanceScorer):
    """
    Importance scorer that ranks rows by cosine similarity in a
    LangChain `Embeddings` space.

    Bring your own embedding model: any object implementing
    `embed_query` / `embed_documents` works (OpenAI, HuggingFace, Anthropic,
    Voyage, or `DeterministicFakeEmbedding` for tests).

    The scorer caches embeddings by content string in a bounded LRU cache so
    repeated reads of the same row never re-embed. Tune `cache_size` based on
    how much unique content you expect to flow through your warm store; set
    `cache_size=0` to disable caching entirely.

    Example:
        from langchain_openai import OpenAIEmbeddings
        scorer = EmbeddingsImportanceScorer(OpenAIEmbeddings(), cache_size=10_000)
        store = WarmStore(scorer=scorer)
    """

    __slots__ = ("embeddings", "role_weights", "_cache_size", "_cache")

    def __init__(
        self,
        embeddings: Embeddings,
        *,
        role_weights: Mapping[str, float] | None = None,
        cache_size: int = _DEFAULT_CACHE_SIZE,
    ) -> None:
        if cache_size < 0:
            raise ValueError("cache_size must be >= 0")
        self.embeddings = embeddings
        self.role_weights = dict(role_weights or {})
        self._cache_size = cache_size
        self._cache: OrderedDict[str, list[float]] = OrderedDict()

    def _embed(self, text: str) -> list[float]:
        if self._cache_size and text in self._cache:
            # mark as most-recently-used
            self._cache.move_to_end(text)
            return self._cache[text]

        vector = self.embeddings.embed_query(text)

        if self._cache_size:
            self._cache[text] = vector
            self._cache.move_to_end(text)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)

        return vector

    def cache_clear(self) -> None:
        """Drop all cached embeddings. Useful between long-running phases."""
        self._cache.clear()

    def cache_size(self) -> int:
        """Number of entries currently cached."""
        return len(self._cache)

    def score(self, query: str, row: Mapping[str, Any]) -> float:
        content = str(row.get("content", ""))
        if not content.strip() or not query.strip():
            return 0.0

        query_vec = self._embed(query)
        doc_vec = self._embed(content)
        if not query_vec or not doc_vec or len(query_vec) != len(doc_vec):
            return 0.0

        dot = sum(a * b for a, b in zip(query_vec, doc_vec))
        query_norm = math.sqrt(sum(a * a for a in query_vec))
        doc_norm = math.sqrt(sum(b * b for b in doc_vec))
        if query_norm == 0 or doc_norm == 0:
            return 0.0

        similarity = dot / (query_norm * doc_norm)
        role = str(row.get("role", "user"))
        weight = self.role_weights.get(role, 1.0)
        return similarity * weight
