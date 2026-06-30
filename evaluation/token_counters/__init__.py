"""Token counting across providers for the DEFLATE-D evaluation.

Public API::

    from evaluation.token_counters import build_counters, count_all

    counters, errors = build_counters(["openai", "claude", "gemini"])
    for tc in count_all(open("func.c").read(), counters):
        print(tc.provider, tc.model, tc.tokens)
"""

from __future__ import annotations

from .base import TokenCount, TokenCounter, TokenCounterError
from .claude_counter import ClaudeTokenCounter
from .gemini_counter import GeminiTokenCounter
from .openai_counter import OpenAITokenCounter
from .registry import (
    ALIASES,
    COUNTERS,
    DEFAULT_PROVIDERS,
    build_counter,
    build_counters,
    count_all,
    resolve_provider,
)

__all__ = [
    "TokenCount",
    "TokenCounter",
    "TokenCounterError",
    "OpenAITokenCounter",
    "ClaudeTokenCounter",
    "GeminiTokenCounter",
    "COUNTERS",
    "ALIASES",
    "DEFAULT_PROVIDERS",
    "build_counter",
    "build_counters",
    "count_all",
    "resolve_provider",
]
