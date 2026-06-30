"""Concurrent rebuild of the per-function RQ1 study across all three tokenizers.

Identical output schema to :mod:`evaluation.run_function_study`, but built to
make the Claude/Gemini ``count_tokens`` network calls tractable for the whole
coreutils corpus: it first materializes every per-function tier string, dedups
them on the exact text (coreutils statically links gnulib, so the same function
recurs across binaries), counts the unique set once -- tiktoken locally,
Claude/Gemini over a thread pool -- and only then reassembles the per-function
records by table lookup. Sequential counting of ~100k unique strings over two
network endpoints would take many hours; the pool brings it down to minutes.

    python -m evaluation.run_function_study_concurrent <decompiled-dir> <out.json>
        [--providers openai,anthropic,google] [--workers 16] [--manifest M ...]
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from deflated.transforms import build_pipeline

from .functions import split_functions, split_functions_lines
from .keys import load_keys
from .run_study import discover, load_provenance
from .token_counters import build_counters

TIERS = ("T0", "T1", "T2", "T3", "T4")
DEFAULT_MIN_RAW = 32  # kept for meta parity with run_function_study


def _split_raw(raw: str) -> tuple[list[str], bool]:
    funcs, balanced = split_functions(raw)
    if balanced:
        return funcs, True
    return split_functions_lines(raw), False


def run(decompiled_dir, out_json, *, providers, workers, manifests):
    decompiled_dir = Path(decompiled_dir)
    load_keys()
    counters, errors = build_counters(providers)
    if not counters:
        raise SystemExit(f"no token counters available: {errors}")
    for provider, reason in errors.items():
        print(f"[func-c] provider skipped: {provider}: {reason}", file=sys.stderr)
    cmap = {c.provider: c for c in counters}
    local = [p for p in cmap if cmap[p].is_local]
    network = [p for p in cmap if not cmap[p].is_local]
    print(f"[func-c] local={local} network={network} workers={workers}",
          file=sys.stderr)

    prov, corpora = load_provenance(manifests)
    pipelines = {t: build_pipeline(t) for t in TIERS if t != "T0"}

    # Pass 1: materialize every per-function tier string, keeping the record
    # skeleton so we can fill token counts by lookup afterwards. ``strings`` maps
    # each tier rendering to a slot; ``unique`` is the dedup set we actually count.
    files = discover(decompiled_dir)
    records = []
    unique: set[str] = set()
    total_functions = 0
    fallback_used: list[str] = []
    for name, by_backend in files.items():
        record: dict = {"binary": name}
        if name in prov:
            record["provenance"] = prov[name]
        for backend, path in sorted(by_backend.items()):
            raw = path.read_text(encoding="utf-8", errors="replace")
            funcs, balanced = _split_raw(raw)
            fn_strings: list[dict[str, str]] = []
            for f in funcs:
                rec = {t: (f if t == "T0" else pipelines[t].apply(f)) for t in TIERS}
                for s in rec.values():
                    unique.add(s)
                fn_strings.append(rec)
            record[backend] = {
                "n_functions": len(funcs),
                "split_balanced": balanced,
                "_strings": fn_strings,  # stripped before write
            }
            total_functions += len(funcs)
            if not balanced:
                fallback_used.append(f"{name}/{backend}")
        records.append(record)
    print(f"[func-c] {total_functions} functions, {len(unique)} unique tier "
          f"strings to count", file=sys.stderr, flush=True)

    counts: dict[str, dict[str, int]] = {s: {} for s in unique}

    # Empty / whitespace-only renderings (a tier that reduces a function to
    # nothing) tokenize to 0 for every provider, and the Claude/Gemini endpoints
    # reject empty content -- so resolve them here and keep them out of the
    # counted set.
    empties = [s for s in unique if not s.strip()]
    for s in empties:
        counts[s] = {p: 0 for p in cmap}
    uniq = [s for s in unique if s.strip()]
    if empties:
        print(f"[func-c] {len(empties)} empty tier string(s) -> 0 tokens",
              file=sys.stderr)

    # Local tokenizers: cheap, run inline.
    for p in local:
        c = cmap[p]
        for s in uniq:
            counts[s][p] = c.count(s)
    print(f"[func-c] local counts done ({len(local)} provider(s))", file=sys.stderr,
          flush=True)

    # Network tokenizers: thread pool. The SDK clients are httpx-backed and
    # thread-safe for independent requests; each counter already retries 429/5xx
    # with backoff. Progress is logged every few thousand strings.
    if network:
        done = 0
        lock = threading.Lock()
        t0 = time.time()

        def count_one(s: str):
            nonlocal done
            d = {p: cmap[p].count(s) for p in network}
            with lock:
                done += 1
                if done % 2000 == 0 or done == len(uniq):
                    rate = done / max(1e-9, time.time() - t0)
                    print(f"[func-c] network {done}/{len(uniq)} "
                          f"({rate:.1f}/s)", file=sys.stderr, flush=True)
            return s, d

        with ThreadPoolExecutor(max_workers=workers) as ex:
            for s, d in ex.map(count_one, uniq):
                counts[s].update(d)

    # Pass 2: reassemble per-function records by lookup, drop the scratch strings.
    for record in records:
        for backend in list(record):
            be = record[backend]
            if not isinstance(be, dict) or "_strings" not in be:
                continue
            be["functions"] = [
                {t: counts[rec[t]] for t in TIERS} for rec in be["_strings"]
            ]
            del be["_strings"]

    result = {
        "meta": {
            "providers": {c.provider: c.model for c in counters},
            "provider_errors": errors,
            "tiers": list(TIERS),
            "decompilers": sorted(
                {b for r in records for b in r
                 if b not in ("binary", "provenance")}),
            "corpora": corpora,
            "unit": "function",
            "transform_regime": "per-function (matches RQ2 tiered_versions)",
            "per_function_min_raw": DEFAULT_MIN_RAW,
        },
        "files": records,
    }
    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[func-c] wrote {out_path} ({len(records)} files, "
          f"{total_functions} functions)", file=sys.stderr)
    if fallback_used:
        print(f"[func-c] line-splitter fallback for {len(fallback_used)} "
              f"decompilation(s)", file=sys.stderr)
    return result


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("decompiled_dir")
    ap.add_argument("out_json")
    ap.add_argument("--providers", default="openai,anthropic,google")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--manifest", action="append", default=[])
    args = ap.parse_args(argv[1:])
    providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    run(args.decompiled_dir, args.out_json, providers=providers,
        workers=args.workers, manifests=list(args.manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
