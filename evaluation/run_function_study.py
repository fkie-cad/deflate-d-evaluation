"""RQ1 per-function token-savings study.

Companion to :mod:`evaluation.run_study`, which measures savings per *binary*
(whole-file transform). This driver measures savings per *function* under the
*per-function* transform regime --- exactly what RQ2 does: RQ2 grades quality on
one function at a time and :func:`evaluation.rq2.tiers.tiered_versions` renders
each function by transforming it on its own. So the per-function distribution
here and the per-function quality distribution in RQ2 sit on the same unit and
the same transform regime, and a reader can map "this function saves X%" onto
"this function is graded with no quality loss" directly.

Method (per-function transform):

  1. Split the raw decompiled file into top-level functions with the
     layout-independent splitter (:func:`evaluation.functions.split_functions`),
     falling back to :func:`evaluation.functions.split_functions_lines` for
     Binary Ninja files whose raw brace depth does not balance (a Binja
     truncated-string artifact; see ``functions.py``).
  2. For each function, transform it independently at each tier
     (``T0`` = the raw function, ``T1``..``T4`` = ``transform(func, tier)``) and
     count tokens with every available counter. The GPT tokenizer is local and
     free; the Claude/Gemini ``count_tokens`` endpoints are also free and used
     when their keys are present (for cross-tokenizer comparison).

Because each function is transformed on its own --- never the whole file --- the
splitter never has to operate on DEFLATE-D-transformed text, so the T1 line-joining
that defeats layout-based splitting (and the Binja truncated-string interaction
with it) never arises.

A per-function raw-token floor (default 32) drops degenerate tiny functions
whose percentage savings is noise.

    python -m evaluation.run_function_study <decompiled-dir> <out.json>
    python -m evaluation.run_function_study <decompiled-dir> <out.json> --providers openai
    python -m evaluation.run_function_study <decompiled-dir> <out.json> --min-raw 32
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from deflated.transforms import build_pipeline

from .functions import split_functions, split_functions_lines
from .keys import load_keys
from .run_study import discover, load_provenance
from .token_counters import build_counters, count_all

TIERS = ("T0", "T1", "T2", "T3", "T4")
# A function below this many raw tokens is dropped from the distribution: a
# 3-token function's percentage savings is noise. Logged, not silently dropped.
DEFAULT_MIN_RAW = 32


def _counts(text: str, counters, cache: dict) -> dict[str, int]:
    """Per-provider token count for ``text``, memoized on the exact string."""
    if text in cache:
        return cache[text]
    result = {tc.provider: tc.tokens for tc in count_all(text, counters)}
    cache[text] = result
    return result


def _split_raw(raw: str) -> tuple[list[str], bool]:
    """Top-level functions in ``raw``; fall back to the line splitter if unbalanced."""
    funcs, balanced = split_functions(raw)
    if balanced:
        return funcs, True
    return split_functions_lines(raw), False


def _study_backend(raw: str, counters, cache: dict, pipelines: dict) -> dict:
    """Per-function token counts across tiers for one decompilation.

    Each function is transformed independently (the RQ2 regime). ``pipelines``
    maps each non-T0 tier to a pipeline built once (``transform`` rebuilds it on
    every call, which is wasteful across tens of thousands of functions).
    Returns ``{n_functions, split_balanced, functions: [ {tier: {provider: int}} ... ]}``,
    where each function record mirrors the per-binary unit shape (``T0`` is the
    function's raw-token count) so the plot code can reuse the per-binary
    ``collect`` logic over functions.
    """
    funcs, balanced = _split_raw(raw)
    functions: list[dict[str, dict[str, int]]] = []
    for f in funcs:
        rec: dict[str, dict[str, int]] = {}
        for tier in TIERS:
            text = f if tier == "T0" else pipelines[tier].apply(f)
            rec[tier] = _counts(text, counters, cache)
        functions.append(rec)
    return {"n_functions": len(functions), "split_balanced": balanced,
            "functions": functions}


def run_function_study(decompiled_dir, out_json, *, providers=None,
                       manifests=None, min_raw=DEFAULT_MIN_RAW) -> dict:
    decompiled_dir = Path(decompiled_dir)
    load_keys()
    counters, errors = build_counters(providers)
    if not counters:
        raise SystemExit(f"no token counters available: {errors}")
    for provider, reason in errors.items():
        print(f"[func-study] provider skipped: {provider}: {reason}", file=sys.stderr)

    prov, corpora = load_provenance(manifests)
    files = discover(decompiled_dir)
    backends_seen: set[str] = set()
    records = []
    cache: dict = {}
    total_functions = 0
    fallback_used: list[str] = []
    # Build each tier's pipeline once and reuse it across every function
    # (transform() rebuilds per call; tens of thousands of functions make that
    # the dominant cost).
    pipelines = {t: build_pipeline(t) for t in TIERS if t != "T0"}
    for name, by_backend in files.items():
        record = {"binary": name}
        if name in prov:
            record["provenance"] = prov[name]
        for backend, path in sorted(by_backend.items()):
            backends_seen.add(backend)
            raw = path.read_text(encoding="utf-8", errors="replace")
            rec = _study_backend(raw, counters, cache, pipelines)
            record[backend] = rec
            total_functions += rec["n_functions"]
            if not rec["split_balanced"]:
                fallback_used.append(f"{name}/{backend}")
        records.append(record)
        n_funcs = sum(record[b]["n_functions"] for b in by_backend)
        print(f"[func-study] {name}: {n_funcs} functions across "
              f"{', '.join(sorted(by_backend))}", file=sys.stderr, flush=True)

    result = {
        "meta": {
            "providers": {c.provider: c.model for c in counters},
            "provider_errors": errors,
            "tiers": list(TIERS),
            "decompilers": sorted(backends_seen),
            "corpora": corpora,
            "unit": "function",
            "transform_regime": "per-function (matches RQ2 tiered_versions)",
            "per_function_min_raw": min_raw,
        },
        "files": records,
    }
    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[func-study] wrote {out_path} ({len(records)} files, "
          f"{total_functions} functions)", file=sys.stderr)
    if fallback_used:
        print(f"[func-study] line-splitter fallback used for {len(fallback_used)} "
              f"decompilation(s) with unbalanced raw braces:", file=sys.stderr)
        for fb in fallback_used:
            print(f"  - {fb}", file=sys.stderr)
    return result


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("decompiled_dir", help="dir of <backend>/<name>.c outputs")
    ap.add_argument("out_json", help="output JSON path")
    ap.add_argument("--manifest", action="append", default=[],
                    help="corpus_manifest.json to tag records with provenance "
                         "(repeatable; one per corpus)")
    ap.add_argument("--providers",
                    help="comma-separated subset of openai/anthropic/google "
                         "(default: all available)")
    ap.add_argument("--min-raw", type=int, default=DEFAULT_MIN_RAW,
                    help=f"per-function raw-token floor (default {DEFAULT_MIN_RAW})")
    args = ap.parse_args(argv[1:])

    providers = (
        [p.strip() for p in args.providers.split(",") if p.strip()]
        if args.providers else None
    )
    run_function_study(args.decompiled_dir, args.out_json,
                       providers=providers, manifests=list(args.manifest),
                       min_raw=args.min_raw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))