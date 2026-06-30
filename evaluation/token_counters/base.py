"""Provider-agnostic token-counting interface.

Each backend implements :class:`TokenCounter.count`, which maps a raw string to
an integer token count for one provider/model. Counting is decoupled from
generation: OpenAI and open-weight tokenizers run locally and free; Claude and
Gemini expose free token-counting endpoints (not billed as tokens).

For the RQ1 token-savings study we tokenize *raw decompiler-output strings*, so
any per-message/role overhead an API endpoint adds is a constant that cancels
out of a raw-vs-compressed delta.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass


class TokenCounterError(RuntimeError):
    """Raised when a counter cannot be constructed or used.

    Typically a missing SDK dependency or a missing API key. The message is
    intended to be surfaced to the user verbatim.
    """


# HTTP statuses worth retrying: rate limit (429), request timeouts, and the
# transient 5xx family. Anything else (auth, bad request) is a hard failure.
_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


def _status_of(exc: Exception):
    """Best-effort HTTP status extraction across SDK exception shapes."""
    for attr in ("status_code", "code", "http_status"):
        val = getattr(exc, attr, None)
        if isinstance(val, int):
            return val
    resp = getattr(exc, "response", None)
    if resp is not None:
        val = getattr(resp, "status_code", None)
        if isinstance(val, int):
            return val
    return None


def _is_retryable(exc: Exception) -> bool:
    if _status_of(exc) in _RETRYABLE_STATUS:
        return True
    name = type(exc).__name__.lower()
    return any(k in name for k in (
        "ratelimit", "timeout", "connection", "servererror",
        "overloaded", "unavailable", "apiconnection",
    ))


def _retry_after_seconds(exc: Exception):
    """Honour a ``Retry-After`` header when the SDK surfaces one."""
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None) if resp is not None else None
    if not headers:
        return None
    try:
        raw = headers.get("retry-after") or headers.get("Retry-After")
    except AttributeError:
        return None
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def retry_request(fn, *, max_attempts: int = 6, base_delay: float = 1.0,
                  max_delay: float = 30.0, sleep=time.sleep):
    """Call ``fn`` with exponential backoff on transient/rate-limit errors.

    Retries 429 and transient 5xx/connection failures (honouring a
    ``Retry-After`` header when present), capping each wait at ``max_delay``.
    Non-retryable errors and :class:`TokenCounterError` propagate immediately.
    """
    attempt = 0
    while True:
        try:
            return fn()
        except TokenCounterError:
            raise
        except Exception as exc:  # noqa: BLE001 - classified by _is_retryable
            attempt += 1
            if attempt >= max_attempts or not _is_retryable(exc):
                raise
            delay = _retry_after_seconds(exc)
            if delay is None:
                delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            sleep(delay)


@dataclass(frozen=True)
class TokenCount:
    """One token-count measurement."""

    provider: str
    model: str
    tokens: int


class TokenCounter(ABC):
    """Counts tokens for a single provider/model.

    Subclasses set the class attributes below and implement :meth:`count`.
    Construction may raise :class:`TokenCounterError` (missing SDK or API key).
    """

    #: Short provider key, e.g. ``"openai"``.
    provider: str = "unknown"
    #: ``True`` if counting runs locally with no network call.
    is_local: bool = False
    #: ``True`` if construction/use requires an API key.
    requires_api_key: bool = False

    def __init__(self, model: str) -> None:
        self.model = model

    @abstractmethod
    def count(self, text: str) -> int:
        """Return the number of tokens in ``text`` for this provider/model."""

    def measure(self, text: str) -> TokenCount:
        """Return a :class:`TokenCount` wrapping :meth:`count`."""
        return TokenCount(self.provider, self.model, self.count(text))

    @property
    def label(self) -> str:
        """Human-readable ``provider:model`` label."""
        return f"{self.provider}:{self.model}"

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} {self.label} local={self.is_local}>"
