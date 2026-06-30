"""OpenAI token counting via ``tiktoken`` (local, offline, free)."""

from __future__ import annotations

from .base import TokenCounter, TokenCounterError


class OpenAITokenCounter(TokenCounter):
    """Counts GPT tokens locally with ``tiktoken``.

    No API key and no network call. ``tiktoken`` resolves the encoding from the
    model name; unknown model names fall back to ``o200k_base``. The GPT-5 family
    uses ``o200k_base``, so newer 5.x point releases tokenize identically even
    when their exact id is not yet in tiktoken's table.
    """

    provider = "openai"
    is_local = True
    requires_api_key = False

    # Latest top-tier model; encoding is o200k_base either way.
    DEFAULT_MODEL = "gpt-5.1"
    FALLBACK_ENCODING = "o200k_base"

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        super().__init__(model)
        try:
            import tiktoken
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise TokenCounterError(
                "openai: `tiktoken` is not installed. Run `pip install tiktoken`."
            ) from exc

        try:
            self._encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Unknown/newer model name: fall back to the current base encoding.
            self._encoding = tiktoken.get_encoding(self.FALLBACK_ENCODING)

    def count(self, text: str) -> int:
        return len(self._encoding.encode(text))
