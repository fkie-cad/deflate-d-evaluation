"""Baseline token-reduction comparison for RQ1 (decision aid).

Computes per-binary token reduction vs raw decompiler output for a set of
*off-the-shelf* baselines, alongside Deflate-D's own tiers, on the SAME corpus
and the SAME tokenizer the headline table uses, so the numbers are directly
comparable.

Deterministic baselines (their reduction is "earned", not a dialed knob):
  B1  ws-strip   naive whitespace strip: strip indentation + trailing ws +
                 blank lines, collapse runs of spaces; KEEP comments and line
                 structure. The sed-level minify a practitioner reaches for.
  B2  minify     aggressive *lossless* minify: B1 + strip comments + join
                 lines + tighten operator spacing. The ceiling of pure
                 formatting removal (a real code minifier).

Model-based baseline (rate-controlled -- reduction is a target you set, so it
belongs in an iso-reduction quality comparison, reported here for context):
  B3  llmlingua2 microsoft/llmlingua-2 task-agnostic prompt compressor.

All deterministic transforms reuse the Deflate-D lexer so string/char literals
are never corrupted.

    python -m evaluation.baselines                 # B1/B2 + Deflate-D tiers
    python -m evaluation.baselines --llmlingua RATE # also B3 at keep-RATE
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

from deflated import transform
from deflated.transforms.lexer import (
    SegmentType,
    map_code,
    scan,
    strip_comments,
)

from .token_counters import build_counter

CORPUS = Path(__file__).parent / "decompiled"
BACKENDS = ("ghidra", "binja", "hexrays")
MIN_RAW_TOKENS = 64  # drop degenerate near-empty decompilations

_SPACES = re.compile(r"[ \t]+")


# --- B1: naive whitespace strip ------------------------------------------------
def minify_ws(raw: str) -> str:
    """Strip indentation + trailing ws, drop blank lines, collapse space runs.

    Keeps comments and the line structure; only touches CODE-segment whitespace
    (string/char literals pass through verbatim via map_code). This is the
    low-effort minify (no line joining, no operator tightening)."""
    def squeeze(code: str) -> str:
        return _SPACES.sub(" ", code)

    collapsed = map_code(raw, squeeze)
    out_lines = []
    for line in collapsed.splitlines():
        s = line.strip()
        if s:
            out_lines.append(s)
    return "\n".join(out_lines) + "\n"


# --- B2: aggressive lossless minify --------------------------------------------
_TIGHTEN = re.compile(r"\s*([{}()\[\];,])\s*")


def minify_full(raw: str) -> str:
    """B1 + strip comments + join all lines + tighten punctuation spacing.

    Lossless: reuses the lexer so literals/comments are handled correctly, and
    only collapses whitespace that is not token-separating. A single space is
    kept between bare tokens so identifiers/keywords never merge; spacing around
    structural punctuation ({}()[];,) is removed."""
    no_comments = strip_comments(raw)

    def flatten(code: str) -> str:
        # newlines/tabs/space-runs -> single space (token separator preserved)
        return _SPACES.sub(" ", code.replace("\n", " ").replace("\t", " "))

    flat = map_code(no_comments, flatten)

    # Tighten spacing around structural punctuation, but only in CODE segments
    # so a "; " or " ," inside a string literal is untouched.
    def tighten(code: str) -> str:
        return _TIGHTEN.sub(r"\1", code)

    return map_code(flat, tighten).strip() + "\n"


# --- B3: LLMLingua-2 (model-based) ---------------------------------------------
_LLMLINGUA = None


def _get_llmlingua():
    global _LLMLINGUA
    if _LLMLINGUA is None:
        from llmlingua import PromptCompressor

        _LLMLINGUA = PromptCompressor(
            model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
            use_llmlingua2=True,
            device_map="cpu",
        )
    return _LLMLINGUA


def llmlingua2(raw: str, rate: float) -> str:
    """Compress with LLMLingua-2 to keep ~``rate`` of the tokens."""
    comp = _get_llmlingua()
    res = comp.compress_prompt(raw, rate=rate, force_tokens=["\n"])
    return res["compressed_prompt"]


# --- driver --------------------------------------------------------------------
def _pct(raw_tok: int, cur_tok: int) -> float:
    return 100.0 * (1.0 - cur_tok / raw_tok)


def _summ(vals):
    if not vals:
        return None
    a = np.array(vals)
    return (
        float(np.percentile(a, 5)),
        float(np.median(a)),
        float(np.percentile(a, 95)),
        float(a.mean()),
        len(a),
    )


def run(methods, llmlingua_rate=None):
    counter = build_counter("openai")  # tiktoken, local, identical to the table
    print(f"# tokenizer: openai / {counter.model}", file=sys.stderr)

    # method name -> backend -> list of per-binary reduction %
    data: dict[str, dict[str, list]] = {m: {b: [] for b in BACKENDS} for m in methods}

    for b in BACKENDS:
        files = sorted((CORPUS / b).glob("*.c"))
        for i, fp in enumerate(files):
            raw = fp.read_text(encoding="utf-8", errors="replace")
            raw_tok = counter.count(raw)
            if raw_tok < MIN_RAW_TOKENS:
                continue
            for m in methods:
                if m in ("T1", "T2", "T3", "T4"):
                    txt = transform(raw, m)
                elif m == "ws-strip":
                    txt = minify_ws(raw)
                elif m == "minify":
                    txt = minify_full(raw)
                elif m == "llmlingua2":
                    txt = llmlingua2(raw, llmlingua_rate)
                else:
                    continue
                data[m][b].append(_pct(raw_tok, counter.count(txt)))
            print(f"  [{b}] {i+1}/{len(files)} {fp.stem}", file=sys.stderr)

    # report
    label = {
        "T1": "Deflate-D T1 (cosmetic, lossless)",
        "T2": "Deflate-D T2 (structural, lossless)",
        "T3": "Deflate-D T3 (contextual, LOSSY)",
        "T4": "Deflate-D T4 (reductive, LOSSY)",
        "ws-strip": "B1 ws-strip (naive minify, lossless)",
        "minify": "B2 minify (aggressive minify, lossless)",
        "llmlingua2": f"B3 LLMLingua-2 (keep={llmlingua_rate}, LOSSY)",
    }
    print("\n== Per-binary token reduction vs raw (%, P5 / median / P95 / mean) ==")
    hdr = f"{'method':40s} " + "  ".join(f"{b:>22s}" for b in BACKENDS)
    print(hdr)
    print("-" * len(hdr))
    for m in methods:
        cells = []
        for b in BACKENDS:
            s = _summ(data[m][b])
            cells.append(
                f"{s[0]:5.1f}/{s[1]:5.1f}/{s[2]:5.1f}" if s else "        --        "
            )
        print(f"{label.get(m, m):40s} " + "  ".join(f"{c:>22s}" for c in cells))
    # medians-only compact view
    print("\n== Median reduction only ==")
    print(f"{'method':40s} " + "  ".join(f"{b:>10s}" for b in BACKENDS))
    for m in methods:
        cells = []
        for b in BACKENDS:
            s = _summ(data[m][b])
            cells.append(f"{s[1]:6.1f}%" if s else "   --  ")
        print(f"{label.get(m, m):40s} " + "  ".join(f"{c:>10s}" for c in cells))
    return data


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--llmlingua", type=float, metavar="RATE",
                    help="also run LLMLingua-2 keeping ~RATE of tokens (e.g. 0.5)")
    ap.add_argument("--only-baselines", action="store_true",
                    help="skip Deflate-D tiers, baselines only")
    args = ap.parse_args(argv[1:])

    methods = [] if args.only_baselines else ["T1", "T2", "T3", "T4"]
    methods += ["ws-strip", "minify"]
    if args.llmlingua is not None:
        methods.append("llmlingua2")
    run(methods, llmlingua_rate=args.llmlingua)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
