"""Claude token counting via the free ``count_tokens`` endpoint."""

from __future__ import annotations

from .base import TokenCounter, TokenCounterError, retry_request


class ClaudeTokenCounter(TokenCounter):
    """Counts Claude tokens via ``messages.count_tokens``.

    The endpoint is free (not billed as tokens) but is a network call and needs
    ``ANTHROPIC_API_KEY`` in the environment. Claude has no public local
    tokenizer for current models. The endpoint counts a full message (a few
    tokens of role/delimiter overhead); this constant cancels out of a
    raw-vs-compressed delta.

    The tokenizer is model-dependent: Opus 4.7 / 4.8 / Fable 5 share one
    tokenizer, distinct from older models. Pass ``model`` accordingly.
    """

    provider = "anthropic"
    is_local = False
    requires_api_key = True

    DEFAULT_MODEL = "claude-opus-4-8"

    def __init__(self, model: str = DEFAULT_MODEL, client: object | None = None) -> None:
        super().__init__(model)
        if client is not None:
            self._client = client
            return
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise TokenCounterError(
                "anthropic: SDK not installed. Run `pip install anthropic`."
            ) from exc
        try:
            self._client = anthropic.Anthropic()
        except Exception as exc:  # missing key, etc.
            raise TokenCounterError(
                "anthropic: could not initialize client "
                "(set ANTHROPIC_API_KEY). " + str(exc)
            ) from exc

    def count(self, text: str) -> int:
        resp = retry_request(lambda: self._client.messages.count_tokens(
            model=self.model,
            messages=[{"role": "user", "content": text}],
        ))
        return resp.input_tokens
