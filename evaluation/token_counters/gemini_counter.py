"""Gemini token counting via the free ``count_tokens`` endpoint."""

from __future__ import annotations

from .base import TokenCounter, TokenCounterError, retry_request


class GeminiTokenCounter(TokenCounter):
    """Counts Gemini tokens via ``client.models.count_tokens``.

    Free, but a network call needing ``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``)
    in the environment. Uses the unified ``google-genai`` SDK
    (``pip install google-genai``); the older ``google-generativeai`` package
    has a different API.

    The default is the current top-tier text model with ``countTokens`` support.
    """

    provider = "google"
    is_local = False
    requires_api_key = True

    DEFAULT_MODEL = "gemini-3.1-pro-preview"

    def __init__(self, model: str = DEFAULT_MODEL, client: object | None = None) -> None:
        super().__init__(model)
        if client is not None:
            self._client = client
            return
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise TokenCounterError(
                "google: `google-genai` not installed. Run `pip install google-genai`."
            ) from exc
        try:
            self._client = genai.Client()
        except Exception as exc:  # missing key, etc.
            raise TokenCounterError(
                "google: could not initialize client "
                "(set GEMINI_API_KEY or GOOGLE_API_KEY). " + str(exc)
            ) from exc

    def count(self, text: str) -> int:
        resp = retry_request(
            lambda: self._client.models.count_tokens(
                model=self.model, contents=text
            )
        )
        return resp.total_tokens
