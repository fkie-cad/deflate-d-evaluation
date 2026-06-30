"""Build the vendored CAPYBARA stripped-summarization JSONL for RQ2 task T-c.

CAPYBARA (Al-Kaswan et al., BinT5, SANER 2023) ships its data on HuggingFace as
parquet with the code and the reference summary stored as *space-separated token
lists* (`code_tokens`, `docstring_tokens`). We use the ``dedup_stripped`` variant
(deduplicated, fully stripped Ghidra decompilation; FUN_*/param_* placeholders),
whose ``docstring_tokens`` are the *original source-code comments* the authors
extracted -- i.e. human-written references, not model-generated. This is the
whole reason we prefer it over BinaryLLMs-Eval's ChatGPT references.

Because the code is a flat token stream with no layout, we *detokenize* it back
to conventional C (punctuation adjacency + statement/brace line breaks +
indentation) so that T0 (the raw baseline, used verbatim by the tiers) looks
like real decompiler output rather than a one-line token soup. This is a
one-time, lossless-of-content preprocessing step; the docstring is just
space-joined (it is prose).

Run once to (re)generate the vendored JSONL::

    python -m evaluation.datasets.capybara.build_capybara_jsonl

Requires ``huggingface_hub`` + ``pyarrow`` (build-time only; the eval loader
reads the JSONL with the standard library alone).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ID = "AISE-TUDelft/Capybara"
PARQUET = "data/dedup_stripped-00000-of-00001.parquet"
OUT = Path(__file__).resolve().parent / "dedup_stripped.jsonl"

# C keywords that idiomatically keep a space before "(" (so "if (" is not
# collapsed to "if(" when we tighten function-call spacing).
_KW_PAREN = ("if", "while", "for", "switch", "return", "sizeof", "do", "else")


def detokenize_c(tokens: list[str]) -> str:
    """Reconstruct conventional C source from a space-separated token stream."""
    s = " ".join(tokens)
    # Punctuation adjacency.
    s = re.sub(r"\s+([,;])", r"\1", s)            # no space before , ;
    s = re.sub(r"\s+([)\]])", r"\1", s)           # no space before ) ]
    s = re.sub(r"([(\[])\s+", r"\1", s)           # no space after ( [
    s = re.sub(r"\s*->\s*", "->", s)              # pointer member
    s = re.sub(r"\s*\.\s*", ".", s)               # struct member
    s = re.sub(r"\s*::\s*", "::", s)
    s = re.sub(r"(\w)\s+\(", r"\1(", s)           # call/decl: ident ( -> ident(
    s = re.sub(rf"\b({'|'.join(_KW_PAREN)})\(", r"\1 (", s)  # restore keyword space

    # Re-introduce line breaks + indentation around ; { }.
    out: list[str] = []
    indent = 0
    buf = ""

    def flush():
        nonlocal buf
        if buf.strip():
            out.append("    " * indent + buf.strip())
        buf = ""

    for ch in s:
        if ch == "{":
            buf += " {"
            flush()
            indent += 1
        elif ch == "}":
            flush()
            indent = max(0, indent - 1)
            out.append("    " * indent + "}")
        elif ch == ";":
            buf += ";"
            flush()
        else:
            buf += ch
    flush()
    return "\n".join(out)


def main() -> int:
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    path = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=PARQUET)
    table = pq.read_table(path)
    cols = {c: table.column(c).to_pylist() for c in table.column_names}
    n = table.num_rows

    seen: set[str] = set()
    written = 0
    with OUT.open("w", encoding="utf-8") as fh:
        for i in range(n):
            code = detokenize_c(cols["code_tokens"][i])
            summary = " ".join(cols["docstring_tokens"][i]).strip()
            if not code.strip() or not summary:
                continue
            # Stable, unique id from the dataset's own id (fall back to row index).
            raw_id = cols["id"][i]
            rid = f"capybara-{raw_id}"
            if rid in seen:
                rid = f"capybara-row{i}"
            seen.add(rid)
            rec = {
                "id": rid,
                "decompiler": "ghidra",
                "raw_code": code,
                "ref_summary": summary,
                "func_name": cols["fun_name"][i],
                "repo": cols["repo"][i],
                "opt_level": cols["opt_level"][i],
                "partition": cols["partition"][i],
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
    print(f"wrote {written}/{n} records -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
