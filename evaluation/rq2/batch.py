"""Message Batches API prefill for RQ2 (50% cheaper; offline, non-latency-sensitive).

RQ2 is an offline study, so the Batch API is the right vehicle: half price, and
turnaround is typically well under the 24h ceiling. Rather than rewrite
the task runners around the asynchronous submit/poll/retrieve shape, we treat the
batch as a **cache prefill**:

    enumerate every request the runners would make
      -> filter out anything already in the request/response cache
      -> submit the misses as one (or a few) Message Batch(es)
      -> poll until ended
      -> write each response into the SAME content-addressed cache (client._cache_store)

The normal runners are then executed unchanged. Because every generation call
routes through ``OpusClient.complete_with_usage`` (cache-first), they all hit the
prefilled cache, pay nothing, and only do local parsing/scoring. Any request that
errored in the batch is simply absent from the cache, so the runner falls back to
a live call for that one.

The enumerator reproduces the runners' prompts exactly (same ``PROMPTS``,
``tiered_versions`` and ``render``), so the cache keys it computes are identical
to the ones the runners will look up. ``tests`` assert this no-drift property.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

from .client import OpusClient, Usage
from .tasks import gen_prompt, ref_prompt
from .tiers import TIERS, tiered_versions

# Batch limits: <=100k requests / <=256MB each. We chunk well under both so a
# WHOLE-dataset varname run (>100k requests) is split into sequential batches.
_CHUNK = 40_000


@dataclass
class ReqSpec:
    """One generation the runners will make: prompt + repeat index + provenance."""

    prompt: str
    repeat: int
    meta: dict
    max_tokens: int = 4096


def enumerate_specs(task: str, records: list[dict], repeats: int) -> list[ReqSpec]:
    """Every request ``run_<task>`` issues, in the same construction as the runner.

    Must stay byte-identical to the runner's prompt building or the prefilled
    cache keys won't match what the runner looks up (the runner would then re-call
    the API). The summarize task additionally issues one source-derived reference
    generation per function (repeat 0), exactly as ``_summarize_references``.
    """
    r = max(1, repeats)
    specs: list[ReqSpec] = []
    # Summarize references are generated from source only when the dataset ships
    # none. CAPYBARA records carry a human ``ref_summary``, so no reference-
    # generation requests are enumerated (and ``scode`` is absent). Mirrors the
    # same guard in ``tasks.run_summarize``.
    if task == "summarize" and not all(rec.get("ref_summary") for rec in records):
        for rec in records:
            specs.append(ReqSpec(ref_prompt(rec), 0, {"task": "summarize_ref", "id": rec["id"]}))
    for rec in records:
        versions = tiered_versions(rec["raw_code"])
        for tier in TIERS:
            prompt = gen_prompt(task, rec, tier, versions)
            for i in range(r):
                specs.append(ReqSpec(prompt, i, {"task": task, "id": rec["id"], "tier": tier}))
    return specs


def _batches_dir(client: OpusClient):
    assert client.cache_dir is not None
    return client.cache_dir / "_batches"


def _save_pending(client: OpusClient, batch_id: str, task: str, entries: list[dict]) -> None:
    """Persist a submitted batch BEFORE polling, so a crash mid-poll is recoverable.

    ``entries`` is ``[{custom_id, req, repeat, meta}, ...]`` -- everything needed to
    map each result back to its cache key without the in-memory state of the run
    that submitted it.
    """
    d = _batches_dir(client)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{batch_id}.json"
    rec = {"batch_id": batch_id, "task": task, "ts": time.time(),
           "status": "submitted", "requests": entries}
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _update_status(client: OpusClient, batch_id: str, status: str) -> None:
    path = _batches_dir(client) / f"{batch_id}.json"
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    rec["status"] = status
    rec[f"{status}_ts"] = time.time()
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _poll_to_end(client: OpusClient, batch_id: str, task: str,
                 poll_interval: float, max_wait: float) -> bool:
    """Poll a batch until ``ended``. Returns False if it outlived ``max_wait``."""
    waited = 0.0
    while True:
        b = client._client.messages.batches.retrieve(batch_id)
        if b.processing_status == "ended":
            return True
        rc = getattr(b, "request_counts", None)
        if rc is not None:
            print(f"[batch:{task}] {batch_id} {b.processing_status}: "
                  f"processing={rc.processing} succeeded={rc.succeeded} errored={rc.errored}")
        if waited >= max_wait:
            print(f"[batch:{task}] {batch_id} still running after max_wait={max_wait}s. "
                  "The batch keeps processing server-side -- re-run to resume and collect it.")
            return False
        time.sleep(poll_interval)
        waited += poll_interval


def _collect(client: OpusClient, batch_id: str, by_cid: dict[str, dict], report: dict) -> None:
    """Stream an ended batch's results into the cache. Idempotent (safe to re-collect)."""
    for result in client._client.messages.batches.results(batch_id):
        entry = by_cid.get(result.custom_id)
        if entry is None:
            continue
        rtype = result.result.type
        if rtype == "succeeded":
            message = result.result.message
            text = "".join(b.text for b in message.content if getattr(b, "type", "") == "text")
            usage = Usage.from_response(message)
            client._cache_store(entry["req"], entry["repeat"], message, text, usage, entry["meta"])
            report["succeeded"] += 1
        else:
            report["errored"] += 1
            detail = getattr(getattr(result.result, "error", None), "type", rtype)
            report["errors"].append(f"{result.custom_id} ({entry['meta']}): {rtype}/{detail}")


def _resume_pending(client: OpusClient, poll_interval: float, max_wait: float, report: dict) -> None:
    """Reattach to any batch a prior (possibly killed) run submitted but never
    collected, and store its already-paid-for results. This is what makes a dead
    terminal safe: those requests are never re-submitted (= never paid for twice).
    """
    d = _batches_dir(client)
    if not d.exists():
        return
    for path in sorted(d.glob("*.json")):
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if rec.get("status") != "submitted":
            continue
        bid, task = rec["batch_id"], rec.get("task", "?")
        by_cid = {e["custom_id"]: e for e in rec.get("requests", [])}
        print(f"[batch:{task}] resuming in-flight batch {bid} ({len(by_cid)} requests) "
              "from a previous run")
        try:
            ended = _poll_to_end(client, bid, task, poll_interval, max_wait)
        except Exception as exc:  # noqa: BLE001 - batch expired / not found
            print(f"[batch:{task}] cannot reattach to {bid}: {exc!r}; its requests "
                  "will be re-submitted if still uncached.")
            _update_status(client, bid, "lost")
            continue
        if not ended:
            continue  # still running; leave 'submitted' for a later run
        _collect(client, bid, by_cid, report)
        _update_status(client, bid, "collected")
        report["resumed"] += 1


def prefill(
    task: str,
    records: list[dict],
    repeats: int,
    *,
    model: str = "claude-opus-4-8",
    client: OpusClient | None = None,
    poll_interval: float = 30.0,
    max_wait: float = 24 * 3600,
    chunk: int = _CHUNK,
) -> dict:
    """Prefill the request cache for ``task`` via the Batch API. Returns a report.

    Resumable: each submitted batch is recorded to ``<cache>/_batches/`` *before*
    polling, so if the process dies mid-poll a later run reattaches and collects
    the (already-billed) results instead of re-submitting them. Only cache-misses
    are ever submitted, so re-running is always safe.
    """
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    client = client or OpusClient(model=model)
    if client.cache_dir is None:
        raise RuntimeError(
            "request cache is disabled (RQ2_NO_CACHE=1); batch prefill writes into "
            "the cache, so it cannot run with caching off."
        )

    report = {"task": task, "total": 0, "cached": 0, "submitted": 0,
              "succeeded": 0, "errored": 0, "resumed": 0, "errors": []}

    # Phase 0: drain batches a previous run left in flight.
    _resume_pending(client, poll_interval, max_wait, report)

    # Phase 1: submit only what is still not cached (after the drain).
    specs = enumerate_specs(task, records, repeats)
    misses: list[tuple[ReqSpec, dict]] = []
    for s in specs:
        req = client._request_params(s.prompt, s.max_tokens)
        if client._cache_load(req, s.repeat) is None:
            misses.append((s, req))
    report["total"] = len(specs)
    report["cached"] = len(specs) - len(misses)
    report["submitted"] = len(misses)
    print(f"[batch:{task}] {len(specs)} requests total, {report['cached']} already "
          f"cached, {len(misses)} to submit")
    if not misses:
        return report

    # Phase 2: submit in chunks; persist each batch BEFORE polling, then collect.
    n_chunks = (len(misses) + chunk - 1) // chunk
    for ci, start in enumerate(range(0, len(misses), chunk), 1):
        part = misses[start : start + chunk]
        entries: list[dict] = []
        by_cid: dict[str, dict] = {}
        batch_requests = []
        for idx, (spec, req) in enumerate(part):
            cid = f"r{idx}"  # unique within this batch
            entry = {"custom_id": cid, "req": req, "repeat": spec.repeat, "meta": spec.meta}
            entries.append(entry)
            by_cid[cid] = entry
            batch_requests.append(
                Request(
                    custom_id=cid,
                    params=MessageCreateParamsNonStreaming(
                        model=req["model"],
                        max_tokens=req["max_tokens"],
                        messages=req["messages"],
                    ),
                )
            )

        batch = client._client.messages.batches.create(requests=batch_requests)
        _save_pending(client, batch.id, task, entries)  # persisted before any poll
        print(f"[batch:{task}] chunk {ci}/{n_chunks}: submitted {len(part)} requests as "
              f"{batch.id} (saved for resume); polling every {poll_interval:.0f}s")
        if not _poll_to_end(client, batch.id, task, poll_interval, max_wait):
            continue  # record stays 'submitted'; a later run resumes it
        _collect(client, batch.id, by_cid, report)
        _update_status(client, batch.id, "collected")

    print(f"[batch:{task}] done: {report['succeeded']} cached from batch "
          f"({report['resumed']} resumed from earlier runs), {report['errored']} errored "
          "(uncached ones fall back to a live call in the runner)")
    return report
