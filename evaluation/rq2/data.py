"""Load each published dataset into a uniform per-function record.

A record carries the *raw* decompiler output (the T0 input), the task-specific
ground truth, and the provenance needed to cite the source. Tiers are applied
later by :mod:`evaluation.rq2.tiers`.

  funcname  : {id, decompiler, raw_code, ref_name}            <- SymGen (Ghidra)
  varname   : {id, decompiler, raw_code, gt_vars}             <- ReSym  (IDA)
  summarize : {id, decompiler, raw_code, ref_summary, func_name}  <- CAPYBARA (Ghidra)

``gt_vars`` is ``{placeholder: (dwarf_name, dwarf_type)}``.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

DATASETS = Path(__file__).resolve().parents[1] / "datasets"

# Default seed for the sample permutation. A run picks `limit` functions as a
# prefix of this fixed permutation, so the selection is (a) unbiased w.r.t. the
# dataset's file order (which is grouped by binary for ReSym), (b) reproducible
# from the seed alone, and (c) *nested*: the first 400 are a subset of the first
# 800, so a larger later run reuses the smaller run's cache instead of redrawing.
DEFAULT_SAMPLE_SEED = 0


def _sample(records: list[dict], limit: int | None, seed: int) -> list[dict]:
    """Return `limit` records as a prefix of a seeded permutation of `records`.

    The whole population is shuffled with `seed` first, *then* sliced, so the
    same seed yields nested prefixes across limits (400 ⊂ 800) and the cache a
    smaller run filled is reused verbatim by a larger one. IDs are assigned by
    the loader before this call, so shuffling never changes which id maps to
    which function -- only which functions a given `limit` selects.
    """
    order = list(range(len(records)))
    random.Random(seed).shuffle(order)
    shuffled = [records[i] for i in order]
    return shuffled[:limit] if limit else shuffled


# --------------------------------------------------------------------------- #
# T-a: function naming --- SymGen (Ghidra)
# --------------------------------------------------------------------------- #


def load_funcname(limit: int | None = None, seed: int = DEFAULT_SAMPLE_SEED) -> list[dict]:
    """SymGen test set: alpaca records with ``input``=Ghidra body, ``output``=name.

    SymGen releases exactly 400 functions, so ``limit=400`` selects the whole
    split regardless of ``seed`` (the permutation only reorders it).
    """
    path = DATASETS / "symgen" / "test_set.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    out: list[dict] = []
    for i, item in enumerate(data):
        ref = _extract_symgen_name(item.get("output", ""))
        if not ref:
            continue
        code = (item.get("input") or "").strip()
        if not code:
            continue
        out.append(
            {
                "id": f"symgen-{i}",
                "decompiler": "ghidra",
                "raw_code": code,
                "ref_name": ref,
            }
        )
    return _sample(out, limit, seed)


def _extract_symgen_name(output: str) -> str:
    """'The predicted function name is send_hello_verify' -> 'send_hello_verify'."""
    marker = "is "
    if marker in output:
        return output.split(marker, 1)[1].strip().split()[0]
    return output.strip().split()[0] if output.strip() else ""


# --------------------------------------------------------------------------- #
# T-b: variable naming --- ReSym (IDA)
# --------------------------------------------------------------------------- #


def load_varname(limit: int | None = None, seed: int = DEFAULT_SAMPLE_SEED) -> list[dict]:
    """ReSym VarDecoder test JSONL: ``input`` is a full ReSym prompt (prose + the
    decompiled code in a fenced block) and ``output`` is the DWARF labels.

    We extract only the code from the fenced block (so our tiers and our own
    prompt apply, not ReSym's variable-list prompt) and pair it with the DWARF
    ground truth. The JSONL lives in ``datasets/resym/ReSym_data/``; if absent a
    clear error tells the user what to download.
    """
    data_dir = DATASETS / "resym" / "ReSym_data"
    jsonl = _find_resym_jsonl(data_dir)
    if jsonl is None:
        raise FileNotFoundError(
            "ReSym VarDecoder test data not found under "
            f"{data_dir}. Download ReSym_data.zip from "
            "https://zenodo.org/records/13923982/files/ReSym_data.zip and extract "
            "so that the VarDecoder test .jsonl is under "
            "evaluation/datasets/resym/ReSym_data/. (Zenodo may block automated "
            "downloads; fetch it manually.)"
        )
    out: list[dict] = []
    with jsonl.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            code = _extract_resym_code(obj.get("input", ""))
            gt = _parse_resym_gt(obj.get("output", ""))
            if not code or not gt:
                continue
            out.append(
                {
                    "id": f"resym-{obj.get('bin','?')}-{obj.get('fun_id', i)}",
                    "decompiler": "ida",
                    "raw_code": code,
                    "gt_vars": gt,  # {placeholder: (name, type)}
                }
            )
    return _sample(out, limit, seed)


def _find_resym_jsonl(data_dir: Path) -> Path | None:
    if not data_dir.exists():
        return None
    # Prefer an explicit test split if present, else any VarDecoder jsonl.
    candidates = sorted(data_dir.rglob("*.jsonl"))
    testish = [p for p in candidates if "test" in p.name.lower()]
    return (testish[0] if testish else (candidates[0] if candidates else None))


def _extract_resym_code(text: str) -> str:
    """Pull the decompiled code out of ReSym's prompt.

    ReSym wraps the code in a triple-backtick fence after a one-line question
    ("What are the original name and data type of variables ...?"). Older entries
    may instead start with a ``#include ".../defs.hh"`` line; we strip that too.
    Returns the code between the first and last fence, or the whole text with a
    leading ``#include`` dropped if no fence is present.
    """
    if "```" in text:
        first = text.index("```")
        rest = text[first + 3 :]
        # skip an optional language tag on the opening fence line
        if rest[:1] in ("\n",):
            rest = rest[1:]
        elif rest and not rest.startswith("\n"):
            nl = rest.find("\n")
            rest = rest[nl + 1 :] if nl != -1 else rest
        last = rest.rfind("```")
        code = rest if last == -1 else rest[:last]
        return _strip_resym_header(code).strip()
    return _strip_resym_header(text).strip()


def _strip_resym_header(code: str) -> str:
    """Drop the ``#include ".../defs.hh"`` line ReSym prepends for its clang parser."""
    lines = code.splitlines()
    while lines and lines[0].lstrip().startswith("#include"):
        lines = lines[1:]
    return "\n".join(lines).strip()


def _parse_resym_gt(output: str) -> dict[str, tuple[str, str]]:
    """'v4: buffer, char *\\na1: opts, int' -> {'v4': ('buffer','char *'), ...}."""
    gt: dict[str, tuple[str, str]] = {}
    for line in output.strip().splitlines():
        if ": " not in line:
            continue
        varname, labels = line.split(": ", 1)
        parts = [p.strip() for p in labels.split(",")]
        if len(parts) != 2:
            continue
        gt[varname.strip()] = (parts[0], parts[1])
    return gt


# --------------------------------------------------------------------------- #
# T-c: summarization --- CAPYBARA (Ghidra)
# --------------------------------------------------------------------------- #


def load_summarize(limit: int | None = None, seed: int = DEFAULT_SAMPLE_SEED) -> list[dict]:
    """CAPYBARA ``dedup_stripped``: Ghidra-decompiled stripped C (``raw_code``)
    paired with the original source-comment summary (``ref_summary``, human-written,
    not model-generated). Built once from the HuggingFace parquet by
    ``datasets/capybara/build_capybara_jsonl.py``; we read the vendored JSONL.
    """
    path = DATASETS / "capybara" / "dedup_stripped.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"CAPYBARA data not found at {path}. Build it with "
            "`python -m evaluation.datasets.capybara.build_capybara_jsonl` "
            "(needs huggingface_hub + pyarrow; one-time)."
        )
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if not obj.get("raw_code") or not obj.get("ref_summary"):
            continue
        out.append(
            {
                "id": obj["id"],
                "decompiler": "ghidra",
                "raw_code": obj["raw_code"],
                "ref_summary": obj["ref_summary"],
                "func_name": obj.get("func_name", ""),
            }
        )
    return _sample(out, limit, seed)


LOADERS = {
    "funcname": load_funcname,
    "varname": load_varname,
    "summarize": load_summarize,
}