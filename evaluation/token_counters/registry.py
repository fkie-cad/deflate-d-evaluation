"""Build token counters by provider name and run them over text."""

from __future__ import annotations

from .base import TokenCount, TokenCounter, TokenCounterError
from .claude_counter import ClaudeTokenCounter
from .gemini_counter import GeminiTokenCounter
from .openai_counter import OpenAITokenCounter

#: Provider key -> counter class.
COUNTERS: dict[str, type[TokenCounter]] = {
    "openai": OpenAITokenCounter,
    "anthropic": ClaudeTokenCounter,
    "google": GeminiTokenCounter,
}

#: Convenience aliases accepted on the CLI / API.
ALIASES: dict[str, str] = {
    "gpt": "openai",
    "claude": "anthropic",
    "gemini": "google",
}

DEFAULT_PROVIDERS = ("openai", "anthropic", "google")


def resolve_provider(name: str) -> str:
    """Map an alias or provider key to its canonical provider key."""
    key = name.strip().lower()
    key = ALIASES.get(key, key)
    if key not in COUNTERS:
        valid = ", ".join(sorted(set(COUNTERS) | set(ALIASES)))
        raise TokenCounterError(f"unknown provider {name!r}; valid: {valid}")
    return key


def build_counter(provider: str, model: str | None = None) -> TokenCounter:
    """Construct a single counter, raising :class:`TokenCounterError` on failure."""
    key = resolve_provider(provider)
    cls = COUNTERS[key]
    return cls(model) if model else cls()


def build_counters(
    providers: list[str] | tuple[str, ...] | None = None,
    *,
    models: dict[str, str] | None = None,
    skip_unavailable: bool = True,
) -> tuple[list[TokenCounter], dict[str, str]]:
    """Construct counters for ``providers``.

    Returns ``(counters, errors)`` where ``errors`` maps each provider that
    could not be constructed to the reason. With ``skip_unavailable=True``
    (the default) unavailable providers are reported in ``errors`` rather than
    raising, so a partial run can still proceed (e.g. local OpenAI only).
    """
    providers = providers or list(DEFAULT_PROVIDERS)
    models = models or {}
    counters: list[TokenCounter] = []
    errors: dict[str, str] = {}
    for name in providers:
        try:
            key = resolve_provider(name)
            counters.append(build_counter(key, models.get(key)))
        except TokenCounterError as exc:
            if not skip_unavailable:
                raise
            errors[name] = str(exc)
    return counters, errors


def count_all(text: str, counters: list[TokenCounter]) -> list[TokenCount]:
    """Count ``text`` with every counter in order."""
    return [c.measure(text) for c in counters]
