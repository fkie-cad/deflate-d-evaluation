"""Split decompiler output into top-level functions for the per-function RQ1 study.

RQ2 grades quality per function and, in :func:`evaluation.rq2.tiers.tiered_versions`,
transforms each function *independently* at every tier. The per-function RQ1
distribution is meant to sit on the same unit and the same transform regime, so
:func:`run_function_study` splits a decompiled file's RAW text into functions and
transforms each function on its own --- it never has to split transformed text.

Why a dedicated splitter: a naive line heuristic (signature on its own line,
next non-blank line exactly ``{``) works on simple layouts but misses a
nontrivial fraction of functions in general and, more importantly, it cannot
recover the function set from DEFLATE-D-transformed
text (the T1 cosmetic tier joins each function onto one line). That last point is
moot for the per-function study (only raw is split), but the missed-function
rate is not, so :func:`split_functions` here is brace-depth based and uses the
project's own lexer (:func:`deflated.transforms.lexer.scan`) to segment code
from string/char/comment regions before counting braces. Reusing that lexer ---
the same one the transforms run through --- guarantees the splitter's notion of
"inside a string" matches the tool's, so a ``}`` in a string literal or comment
is never counted as a function boundary.

Caveat (Binary Ninja): Binja emits long string literals truncated with an
ellipsis (``"https://…``) and sometimes drops the closing quote on that line.
On multi-line raw text the lexer's string scan terminates at the newline, so the
truncation is harmless and brace counting is correct for the large majority of
files. In some Binja files the truncation lands such that a brace on the
truncated line is absorbed into the string and the raw brace depth does not
balance; :func:`split_functions` detects this (the depth walk ends off zero) and
the driver falls back to :func:`split_functions_lines`, a layout-based splitter
that does not depend on brace balance, for those files.
"""

from __future__ import annotations

from deflated.transforms.lexer import SegmentType, scan

CODE = SegmentType.CODE



def split_functions(text: str) -> tuple[list[str], bool]:
    """Split ``text`` into top-level function bodies via brace depth.

    Returns ``(functions, balanced)`` where ``balanced`` is False when the
    brace-depth walk ends off zero (the raw text has an unbalanced brace,
    typically the Binja truncated-string artifact described above); the
    caller should then fall back to :func:`split_functions_lines`.

    A function is captured from the start of the top-level statement that opens
    its signature (the statement containing the first depth-0 ``(``) through the
    closing ``}`` of its body. Prototypes (``...;``), externs, typedefs, and
    global data initializers have no following body brace and are not returned,
    so the non-function remainder (e.g. Ghidra ``/* WARNING */`` comments and
    blank lines between functions) is intentionally excluded --- it is
    inter-function decompiler bookkeeping, not the function unit a downstream
    task grades.

    Brace counting runs only over ``code`` segments from :func:`scan`, so braces
    inside string, char, and comment literals never affect depth.
    """
    funcs: list[str] = []
    depth = 0
    # Start offset of the current candidate signature's statement, or None when
    # no top-level `(` has opened a statement we are tracking.
    sig_start: int | None = None
    # Walk segments in order, tracking the running character offset so captured
    # slices index back into the original text.
    pos = 0
    for kind, seg in scan(text):
        if kind != SegmentType.CODE:
            pos += len(seg)
            continue
        base = pos
        i = 0
        n = len(seg)
        while i < n:
            c = seg[i]
            if c == "{":
                if depth == 0 and sig_start is not None:
                    depth = 1
                else:
                    depth += 1
                i += 1
                continue
            if c == "}":
                depth -= 1
                i += 1
                if depth == 0 and sig_start is not None:
                    funcs.append(text[sig_start : base + i])
                    sig_start = None
                continue
            if depth == 0 and c == "(" and sig_start is None:
                # A top-level `(` begins a function signature (a prototype or a
                # definition). Mark the statement start by backing up over the
                # current statement, stopping at the previous `}`, `;`, or
                # newline so the capture begins at the return type, not at `(`.
                # Stopping at `}` keeps the boundary clean even on T1-joined
                # single-line text (where there are no newlines between funcs).
                k = base + i
                while k > 0 and text[k - 1] not in ";\n}":
                    k -= 1
                sig_start = k
            i += 1
        pos += len(seg)
    return funcs, depth == 0


def split_functions_lines(text: str) -> list[str]:
    """Layout-based fallback: capture functions by signature line + body brace.

    Used when :func:`split_functions` reports an unbalanced brace walk (the
    Binja truncated-string artifact). It keys on the decompiler's line layout
    instead of brace balance: a function starts at a column-0 line that contains
    ``(`` and is not a prototype (not ending ``;``) or a comment, with the next
    non-blank line opening the body, and runs to the matching ``}`` by brace
    depth over code segments. Less exact than the primary splitter (it can miss
    functions whose signature is laid out differently) but robust to the
    string-truncation imbalance that defeats the primary path.
    """
    lines = text.split("\n")
    n = len(lines)
    funcs: list[str] = []
    i = 0
    while i < n:
        line = lines[i]
        if (
            line
            and line[0] not in " \t/*#}"
            and "(" in line
            and not line.rstrip().endswith(";")
        ):
            j = i + 1
            while j < n and lines[j].strip() == "":
                j += 1
            if j < n and lines[j].strip() == "{":
                body = [line]
                depth = 0
                k = i + 1
                end = None
                while k < n:
                    body.append(lines[k])
                    for kind, seg in scan(lines[k]):
                        if kind == SegmentType.CODE:
                            depth += seg.count("{") - seg.count("}")
                    if depth == 0 and lines[k].strip() == "}":
                        end = k
                        break
                    k += 1
                if end is not None:
                    funcs.append("\n".join(body).strip())
                    i = end + 1
                    continue
        i += 1
    return funcs