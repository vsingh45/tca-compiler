"""
Exact token attribution via Anthropic count_tokens API.

The paper's central claim requires exact measurement of memory injection
tokens — not estimates. This module implements two-pass token counting:

Pass 1: count tokens WITHOUT memory context → base_tokens
Pass 2: count tokens WITH memory context → full_tokens
injection_tokens = full_tokens - base_tokens (exact, not estimated)

count_tokens is a non-billable API call — it costs $0.
Two passes add ~10-20ms overhead per node, acceptable given
LLM inference takes 80-3000ms per node.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import anthropic

from .pricing import Tier, TIER_MODELS


@runtime_checkable
class TokenCounter(Protocol):
    """
    Provider-agnostic token counting interface.
    Implementations: AnthropicTokenCounter, EstimateTokenCounter.
    """
    def count(self, text: str) -> int:
        """Count tokens in a text string."""
        ...

    def count_injection(
        self,
        base_prompt: str,
        memory_context: list[str],
        tier: Tier,
    ) -> tuple[int, int]:
        """
        Return (base_tokens, injection_tokens) for a prompt + memory pair.
        injection_tokens = count(base + memory) - count(base)
        """
        ...


@dataclass
class AnthropicTokenCounter:
    """
    Exact token counting using Anthropic count_tokens API.
    Non-billable. Adds ~10-20ms per node per two-pass measurement.
    """
    client: anthropic.Anthropic

    def count(self, text: str) -> int:
        """Count tokens in a text string using Anthropic tokenizer."""
        response = self.client.messages.count_tokens(
            model=TIER_MODELS["haiku"],  # tokenizer is same across tiers
            messages=[{"role": "user", "content": text}],
        )
        return response.input_tokens

    def count_injection(
        self,
        base_prompt: str,
        memory_context: list[str],
        tier: Tier,
    ) -> tuple[int, int]:
        """
        Two-pass exact injection token measurement.

        Returns:
            (base_tokens, injection_tokens) where:
            - base_tokens = tokens in system + query without memory
            - injection_tokens = additional tokens from memory context
        """
        model = TIER_MODELS[tier]

        # Pass 1: base prompt only (no memory)
        base_response = self.client.messages.count_tokens(
            model=model,
            messages=[{"role": "user", "content": base_prompt}],
        )
        base_tokens = base_response.input_tokens

        if not memory_context:
            return base_tokens, 0

        # Pass 2: base prompt + memory injected
        memory_block = "\n\nRelevant context:\n" + "\n".join(
            f"[{i+1}] {item}" for i, item in enumerate(memory_context)
        )
        full_prompt = base_prompt + memory_block

        full_response = self.client.messages.count_tokens(
            model=model,
            messages=[{"role": "user", "content": full_prompt}],
        )
        full_tokens = full_response.input_tokens
        injection_tokens = full_tokens - base_tokens

        return base_tokens, injection_tokens


@dataclass
class EstimateTokenCounter:
    """
    Fallback token counter using character-based estimation.

    Used in dry-run mode (no API key required).
    Accuracy: ±15% vs real tokenizer. NOT suitable for paper results.
    Suitable for: development, dry runs, unit tests.
    """
    chars_per_token: float = 4.0  # rough average for English text

    def count(self, text: str) -> int:
        return max(1, int(len(text) / self.chars_per_token))

    def count_injection(
        self,
        base_prompt: str,
        memory_context: list[str],
        tier: Tier,
    ) -> tuple[int, int]:
        base_tokens = self.count(base_prompt)
        if not memory_context:
            return base_tokens, 0
        memory_text = "\n".join(memory_context)
        injection_tokens = self.count(memory_text)
        return base_tokens, injection_tokens


def make_token_counter(
    client: anthropic.Anthropic | None,
    mode: str = "real",
) -> AnthropicTokenCounter | EstimateTokenCounter:
    """
    Factory: return the right counter based on run mode.

    Args:
        client: Anthropic client (required for real mode)
        mode: "real" uses count_tokens API; "dry" uses estimation
    """
    if mode == "dry":
        return EstimateTokenCounter()
    if client is None:
        raise ValueError("Anthropic client required for real mode token counting")
    return AnthropicTokenCounter(client=client)
