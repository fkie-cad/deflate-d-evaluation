"""LLM clients for RQ2: Claude Opus 4.8 for generation, Gemini for embeddings.

Both keys are read from the repo's key files via :mod:`evaluation.keys` (which
maps ``CLAUDE_API_KEY`` -> ``ANTHROPIC_API_KEY`` and ``GEMINI_API_KEY`` ->
``GEMINI_API_KEY``); key values are never printed.

Opus 4.8's extended thinking is opt-in: it occurs only when a ``thinking``
request parameter is sent, and none is sent here. The client issues a bare
``messages.create`` (model + max_tokens + messages, with no thinking or
sampling config), keeping the prompt fixed so the only varying input is the
decompiler rendering.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..keys import load_keys

# Load keys once on import so every client in this package sees them.
load_keys()

OPUS_MODEL = "claude-opus-4-8"
GEMINI_EMBED_MODEL = "gemini-embedding-001"

# Content-addressed request/response cache. Every generation call is keyed by
# the exact request (model + max_tokens + messages + any thinking config) plus a
# repeat index, and the full raw response is stored on disk. A re-run with the
# same prompts therefore replays the saved responses for free -- so when you
# change the *scoring* code (parsing, metrics) you can delete the per-task
# JSONL and re-run to recompute scores against the cached generations without
# paying the API again. The cache lives outside any single out-dir so it is
# shared across runs. Disable with RQ2_NO_CACHE=1; relocate with RQ2_CACHE_DIR.
_DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[1] / "results" / "api_cache"


def _response_to_dict(resp) -> dict:
    """Best-effort JSON-safe dump of an Anthropic Message (the full response)."""
    dump = getattr(resp, "model_dump", None)
    if callable(dump):
        try:
            return dump(mode="json")
        except TypeError:
            return dump()
    to_dict = getattr(resp, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    return {"repr": repr(resp)}


@dataclass
class Usage:
    """Billed token usage for a single ``messages.create`` call.

    ``output_tokens`` would include extended-thinking tokens if thinking were
    enabled; thinking is never enabled here, so ``output_tokens`` is just the
    answer text. ``total_tokens`` is input+output, i.e. the quantity the RQ2
    input/output-cost trade-off is reported on.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @classmethod
    def from_response(cls, resp) -> "Usage":
        u = getattr(resp, "usage", None)
        if u is None:
            return cls()
        return cls(
            input_tokens=int(getattr(u, "input_tokens", 0) or 0),
            output_tokens=int(getattr(u, "output_tokens", 0) or 0),
            cache_creation_input_tokens=int(getattr(u, "cache_creation_input_tokens", 0) or 0),
            cache_read_input_tokens=int(getattr(u, "cache_read_input_tokens", 0) or 0),
        )

    def to_dict(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "total_tokens": self.total_tokens,
        }


class OpusClient:
    """Thin wrapper over ``anthropic.Anthropic`` with retry/backoff.

    The key comes from ``ANTHROPIC_API_KEY``, loaded from
    ``evaluation/CLAUDE_API_KEY`` by :mod:`evaluation.keys`.
    """

    def __init__(
        self,
        model: str = OPUS_MODEL,
        max_retries: int = 8,
        cache_dir: str | Path | None = None,
    ) -> None:
        import anthropic

        self._client = anthropic.Anthropic()
        self.model = model
        self.max_retries = max_retries

        # Resolve the request/response cache directory (see module docstring).
        if cache_dir is None and os.environ.get("RQ2_NO_CACHE") != "1":
            cache_dir = os.environ.get("RQ2_CACHE_DIR") or _DEFAULT_CACHE_DIR
        self.cache_dir: Path | None = Path(cache_dir) if cache_dir else None
        self.cache_hits = 0
        self.cache_misses = 0

    def complete(
        self, prompt: str, *, max_tokens: int = 4096, repeat: int = 0, meta: dict | None = None
    ) -> str:
        """Return the model's text answer for ``prompt``, retrying transient errors."""
        return self.complete_with_usage(prompt, max_tokens=max_tokens, repeat=repeat, meta=meta)[0]

    def complete_with_usage(
        self, prompt: str, *, max_tokens: int = 4096, repeat: int = 0, meta: dict | None = None
    ) -> tuple[str, Usage]:
        """Like :meth:`complete` but also return the billed :class:`Usage`.

        Every call is served from (and written to) the on-disk request/response
        cache when one is configured. The cache key is the exact request plus the
        ``repeat`` index, so identical prompts re-run for free while the repeats
        of a single (function, tier) stay distinct. ``meta`` (e.g. task/id/tier)
        is stored alongside the raw response for human inspection but does not
        affect the key. Usage reflects only the final (successful) request;
        retried attempts that errored before returning a response are not billed.
        """
        req = self._request_params(prompt, max_tokens)

        cached = self._cache_load(req, repeat)
        if cached is not None:
            self.cache_hits += 1
            return cached

        self.cache_misses += 1
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.messages.create(**req)
                # Concatenate text blocks; ignore thinking blocks.
                text = "".join(
                    block.text for block in resp.content if getattr(block, "type", "") == "text"
                )
                usage = Usage.from_response(resp)
                self._cache_store(req, repeat, resp, text, usage, meta)
                return text, usage
            except Exception as exc:  # noqa: BLE001 - network/rate-limit/overloaded
                last_exc = exc
                wait = _backoff_seconds(exc, attempt)
                if wait is None:  # non-retryable
                    raise
                time.sleep(wait)
        assert last_exc is not None
        raise last_exc

    # ----------------------------------------------------------------- cache --
    def _request_params(self, prompt: str, max_tokens: int) -> dict:
        """The exact kwargs passed to ``messages.create`` -- also the cache key.

        Anything that changes the response (model, max_tokens, the prompt, and
        any future thinking/sampling config added here) must live in this dict so
        the cache key tracks it and a changed request misses correctly.
        """
        return {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }

    def _cache_key(self, req: dict, repeat: int) -> str:
        blob = json.dumps(req, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return f"{hashlib.sha256(blob).hexdigest()}-r{repeat}"

    def _cache_path(self, key: str) -> Path:
        assert self.cache_dir is not None
        return self.cache_dir / key[:2] / f"{key}.json"  # shard to bound dir size

    def _cache_load(self, req: dict, repeat: int) -> tuple[str, Usage] | None:
        if self.cache_dir is None:
            return None
        path = self._cache_path(self._cache_key(req, repeat))
        if not path.exists():
            return None
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None  # treat a corrupt entry as a miss
        u = rec.get("usage", {}) or {}
        usage = Usage(
            input_tokens=int(u.get("input_tokens", 0) or 0),
            output_tokens=int(u.get("output_tokens", 0) or 0),
            cache_creation_input_tokens=int(u.get("cache_creation_input_tokens", 0) or 0),
            cache_read_input_tokens=int(u.get("cache_read_input_tokens", 0) or 0),
        )
        return rec.get("text", ""), usage

    def _cache_store(
        self, req: dict, repeat: int, resp, text: str, usage: Usage, meta: dict | None
    ) -> None:
        if self.cache_dir is None:
            return
        path = self._cache_path(self._cache_key(req, repeat))
        path.parent.mkdir(parents=True, exist_ok=True)
        rec = {
            "ts": time.time(),
            "model": self.model,
            "repeat": repeat,
            "meta": meta or {},
            "request": req,
            "response": _response_to_dict(resp),  # full raw response
            "text": text,  # the extracted answer
            "usage": usage.to_dict(),
        }
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)  # atomic: never leave a half-written cache entry

    def count_tokens(self, prompt: str) -> int:
        """Free input-token count for ``prompt`` (no generation, not billed)."""
        resp = self._client.messages.count_tokens(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.input_tokens


def _backoff_seconds(exc: Exception, attempt: int) -> float | None:
    """Return a sleep duration for retryable errors, or ``None`` to give up."""
    import anthropic

    retryable = (
        getattr(exc, "status_code", None) in (429, 500, 502, 503, 529),
        isinstance(exc, anthropic.RateLimitError),
        isinstance(exc, anthropic.APIStatusError) and exc.status_code >= 500,
    )
    if not any(retryable):
        return None
    # Respect Retry-After when present, else exponential with jitter-free floor.
    retry_after = getattr(exc, "response", None)
    headers = getattr(retry_after, "headers", {}) or {}
    if "retry-after" in headers:
        try:
            return float(headers["retry-after"])
        except (TypeError, ValueError):
            pass
    return min(60.0, 2.0 ** attempt)


class GeminiEmbedder:
    """Semantic-similarity embeddings via Google's free text-embedding endpoint."""

    def __init__(self, model: str = GEMINI_EMBED_MODEL) -> None:
        from google import genai

        self._client = genai.Client()  # reads GEMINI_API_KEY
        self.model = model

    def embed(self, texts: Iterable[str], max_retries: int = 12) -> list[list[float]]:
        """Embed ``texts``, retrying through the free tier's per-minute quota.

        The Gemini free embedding tier caps requests per minute (each item in a
        batch counts), so a large scoring pass hits ``429 RESOURCE_EXHAUSTED``.
        We honor the server's ``retryDelay`` (falling back to ~30s, just over the
        1-minute window) and retry rather than failing the whole run -- the call
        is free, so the only cost of a 429 is waiting.
        """
        texts = list(texts)
        for attempt in range(max_retries):
            try:
                resp = self._client.models.embed_content(model=self.model, contents=texts)
                # Each item in resp.embeddings is an Embedding object whose vector
                # is in ``.values``; ``list(obj)`` would iterate fields, not it.
                return [list(emb.values) for emb in resp.embeddings]
            except Exception as exc:  # noqa: BLE001 - genai ClientError on 429
                msg = str(exc)
                is_rate = getattr(exc, "code", None) == 429 or "RESOURCE_EXHAUSTED" in msg or "429" in msg
                if not is_rate or attempt == max_retries - 1:
                    raise
                time.sleep(_gemini_retry_seconds(msg))
        raise RuntimeError("unreachable")


def _gemini_retry_seconds(msg: str, default: float = 35.0) -> float:
    """Pull the server-suggested retry delay out of a 429 message, else default.

    Gemini 429s carry either ``Please retry in 23.4s`` or ``retryDelay: '23s'``;
    we add a small cushion and cap it so a malformed value can't stall the run.
    """
    import re

    m = re.search(r"retry in ([\d.]+)\s*s", msg) or re.search(
        r"retryDelay['\"]?\s*[:=]\s*['\"]?([\d.]+)\s*s", msg
    )
    if m:
        try:
            return min(120.0, float(m.group(1)) + 2.0)
        except ValueError:
            pass
    return default


def cosine(a: list[float], b: list[float]) -> float:
    import numpy as np

    va, vb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(va @ vb / (na * nb))