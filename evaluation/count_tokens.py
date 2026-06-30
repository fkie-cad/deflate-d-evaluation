#!/usr/bin/env python3
"""Count tokens in files (or stdin) across OpenAI, Claude, and Gemini.

Examples::

    # Count one file across all available providers
    python -m evaluation.count_tokens func.c

    # Several files, OpenAI only (fully local, no key needed)
    python -m evaluation.count_tokens --providers openai a.c b.c

    # Read from stdin, JSON output
    cat func.c | python -m evaluation.count_tokens --json -

Providers whose SDK or API key is missing are skipped with a warning unless
``--strict`` is passed. OpenAI is local (``tiktoken``); Claude and Gemini use
their free counting endpoints and need ANTHROPIC_API_KEY / GEMINI_API_KEY.
"""

from __future__ import annotations

import argparse
import json
import sys

from .token_counters import build_counters, resolve_provider
from .token_counters.registry import DEFAULT_PROVIDERS


def _read_source(path: str) -> str:
    if path == "-":
        return sys.stdin.read()
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def _parse_model_overrides(pairs: list[str]) -> dict[str, str]:
    """Parse ``provider=model`` overrides into a {provider: model} dict."""
    models: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--model expects provider=model, got {pair!r}")
        provider, model = pair.split("=", 1)
        models[resolve_provider(provider)] = model
    return models


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="count_tokens",
        description="Count tokens across OpenAI, Claude, and Gemini.",
    )
    parser.add_argument("files", nargs="+", help="file paths, or '-' for stdin")
    parser.add_argument(
        "--providers",
        default=",".join(DEFAULT_PROVIDERS),
        metavar="P[,P...]",
        help=f"comma-separated providers/aliases (default: {','.join(DEFAULT_PROVIDERS)})",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="PROVIDER=MODEL",
        help="override the model for a provider (repeatable)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="fail if any requested provider is unavailable",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    provider_list = [p for p in (p.strip() for p in args.providers.split(",")) if p]
    counters, errors = build_counters(
        provider_list,
        models=_parse_model_overrides(args.model),
        skip_unavailable=not args.strict,
    )
    for provider, reason in errors.items():
        print(f"warning: skipping {provider}: {reason}", file=sys.stderr)
    if not counters:
        print("error: no usable token counters", file=sys.stderr)
        return 1

    results: dict[str, dict[str, int]] = {}
    for path in args.files:
        text = _read_source(path)
        per_file: dict[str, int] = {}
        for counter in counters:
            try:
                per_file[counter.label] = counter.count(text)
            except Exception as exc:  # network/API failure at count time
                print(f"warning: {counter.label} failed on {path}: {exc}", file=sys.stderr)
        results[path] = per_file

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        _print_table(results)
    return 0


def _print_table(results: dict[str, dict[str, int]]) -> None:
    labels: list[str] = []
    for per_file in results.values():
        for label in per_file:
            if label not in labels:
                labels.append(label)
    name_w = max([len("file")] + [len(p) for p in results])
    col_w = {lbl: max(len(lbl), 8) for lbl in labels}

    header = "file".ljust(name_w) + "  " + "  ".join(lbl.rjust(col_w[lbl]) for lbl in labels)
    print(header)
    print("-" * len(header))
    for path, per_file in results.items():
        row = path.ljust(name_w) + "  " + "  ".join(
            str(per_file.get(lbl, "-")).rjust(col_w[lbl]) for lbl in labels
        )
        print(row)


if __name__ == "__main__":
    raise SystemExit(main())
