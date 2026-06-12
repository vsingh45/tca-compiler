from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
import math
import re
from typing import Any, Mapping


TOKEN_RE = re.compile(r"\b[a-zA-Z0-9_]+\b")


def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class ImportanceScorer(ABC):
    """Strategy interface for ranking stored memory rows against a query."""

    @abstractmethod
    def score(self, query: str, row: Mapping[str, Any]) -> float:
        raise NotImplementedError


class KeywordImportanceScorer(ImportanceScorer):
    """
    Lightweight scorer for relevance experiments.

    The default signal blends:
    - lexical overlap between query and content
    - document term frequency weighting
    - a small role prior so assistant/tool output can be ranked differently later
    """

    def __init__(self, role_weights: Mapping[str, float] | None = None) -> None:
        self.role_weights = dict(role_weights or {"user": 1.0, "assistant": 1.05, "tool": 0.95})

    def score(self, query: str, row: Mapping[str, Any]) -> float:
        content = str(row.get("content", ""))
        if not content.strip():
            return 0.0

        query_terms = _tokenize(query)
        doc_terms = _tokenize(content)
        if not query_terms or not doc_terms:
            return 0.0

        query_counts = Counter(query_terms)
        doc_counts = Counter(doc_terms)

        dot_product = sum(query_counts[token] * doc_counts[token] for token in query_counts)
        if dot_product == 0:
            return 0.0

        query_norm = math.sqrt(sum(count * count for count in query_counts.values()))
        doc_norm = math.sqrt(sum(count * count for count in doc_counts.values()))
        cosine_like = dot_product / (query_norm * doc_norm)

        role = str(row.get("role", "user"))
        role_weight = self.role_weights.get(role, 1.0)
        return cosine_like * role_weight
