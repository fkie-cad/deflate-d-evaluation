"""Scoring metrics for RQ2.

Naming tasks (T-a function naming, T-b variable naming) use sub-token
precision/recall/F1 with CodeWordNet synonym matching, mirroring the source
datasets' own scoring (BinaryLLMs-Eval ``cal_funcname_metrics`` /
SymGen's CodeWordNet F1). We port the dependency-free camel/snake splitter
(``split_func_name``); the source ``my_split_func_name`` additionally applies a
sentencepiece model + suffix-merge, but the *tier-vs-tier* comparison we report
is invariant to that tokenization choice, so we drop it to avoid shipping a
sentencepiece model.

Summarization (T-c) uses embedding cosine (Gemini ``gemini-embedding-001``) as
the primary, paraphrase-robust metric over CAPYBARA's human source-comment
summaries, plus sentence BLEU-4 as a surface-level secondary. All CIs are
non-parametric bootstrap over per-function scores.
"""

from __future__ import annotations

import re
from pathlib import Path

# --------------------------------------------------------------------------- #
# Name tokenization (ported from BinaryLLMs-Eval eval/utils.py, dependency-free)
# --------------------------------------------------------------------------- #


def _split_normal(func_name: str) -> list[str]:
    func_name = "".join(c if c.isalpha() else "_" for c in func_name)
    chars = list(func_name)
    upper_idx = [i for i, c in enumerate(chars) if c.isupper()]
    for i, idx in enumerate(upper_idx):
        chars.insert(idx + i, "_")
    tokens = [t for t in "".join(chars).split("_") if t]
    return [t.lower() for t in tokens]


def _get_range(nums: list[int]) -> list[int]:
    """Collapse runs of consecutive uppercase into boundary indices."""
    left = right = 0
    result = [0]
    while right < len(nums):
        while right + 1 < len(nums) and nums[right] + 1 == nums[right + 1]:
            right += 1
        temp: list[int] = []
        if right > left:
            if nums[left] not in result:
                temp.append(nums[left])
            temp.append(nums[right] + 1)
        result.extend(temp)
        left = right + 1
        right = left
    return result


def split_func_name(func_name: str) -> list[str]:
    """Split an identifier (camelCase / snake_case / PascalCase) into sub-tokens."""
    func_name = "".join(c if c.isalpha() else "_" for c in func_name)
    chars = list(func_name)
    upper_idx = [i for i, c in enumerate(chars) if c.isupper()]
    upper_range = _get_range(upper_idx)
    if len(func_name) not in upper_range:
        upper_range.append(len(func_name))
    tmp = [
        func_name[upper_range[i] : upper_range[i + 1]]
        for i in range(len(upper_range) - 1)
    ]
    tokens: list[str] = []
    for t in tmp:
        if t.isupper():
            tokens.append(t.lower().replace("_", ""))
        else:
            tokens.extend(_split_normal(t))
    return tokens


# --------------------------------------------------------------------------- #
# CodeWordNet synonyms
# --------------------------------------------------------------------------- #


class CodeWordNet:
    """Word -> synonym-set index, loaded from the BinaryLLMs-Eval cluster file.

    Each line is a comma-separated cluster of mutually-synonymous tokens.
    """

    def __init__(self, path: str | Path) -> None:
        clusters: list[set[str]] = []
        word2cluster: dict[str, set[str]] = {}
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            cluster = {w for w in line.split(",") if w}
            if not cluster:
                continue
            clusters.append(cluster)
            for w in cluster:
                word2cluster.setdefault(w, set()).update(cluster)
        self._word2cluster = word2cluster

    def expand(self, tokens: list[str]) -> set[str]:
        out: set[str] = set()
        for t in tokens:
            out.add(t)
            out.update(self._word2cluster.get(t, ()))
        return out


# --------------------------------------------------------------------------- #
# Naming F1
# --------------------------------------------------------------------------- #


def name_prf(pred_name: str, ref_name: str, cw: CodeWordNet) -> tuple[float, float, float]:
    """Per-instance sub-token precision/recall/F1 with synonym matching."""
    p_tokens = split_func_name(pred_name)
    r_tokens = [t for t in split_func_name(ref_name) if t]
    if not r_tokens:
        return 0.0, 0.0, 0.0
    if not p_tokens:
        return 0.0, 0.0, 0.0
    ref_ext = cw.expand(r_tokens)
    tp = sum(1 for t in p_tokens if t in ref_ext)
    precision = tp / len(p_tokens)
    recall = tp / len(r_tokens)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return precision, recall, f1


def macro_f1(pairs: list[tuple[str, str]], cw: CodeWordNet) -> dict[str, float]:
    """Macro-average P/R/F1 over (pred, ref) pairs (skips empty refs)."""
    ps, rs, fs = [], [], []
    for pred, ref in pairs:
        if not split_func_name(ref):
            continue
        p, r, f1 = name_prf(pred, ref, cw)
        ps.append(p)
        rs.append(r)
        fs.append(f1)
    n = len(fs) or 1
    return {
        "precision": sum(ps) / n if ps else 0.0,
        "recall": sum(rs) / n if rs else 0.0,
        "f1": sum(fs) / n if fs else 0.0,
        "n": len(fs),
    }


# --------------------------------------------------------------------------- #
# Summarization: BLEU
# --------------------------------------------------------------------------- #


def bleu4(pred: str, ref: str) -> float:
    """Sentence-level BLEU-4 (sacrebleu). Returns 0.0 on empty prediction."""
    import sacrebleu

    pred = pred.strip()
    if not pred:
        return 0.0
    return sacrebleu.sentence_bleu(pred, [ref.strip()], lowercase=True).score / 100.0


# --------------------------------------------------------------------------- #
# Bootstrap confidence intervals
# --------------------------------------------------------------------------- #


def bootstrap_mean_ci(
    scores: list[float],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 12345,
) -> dict[str, float]:
    """Bootstrap mean and (1-alpha) CI for a list of per-instance scores."""
    import numpy as np

    arr = np.asarray([s for s in scores if s is not None], dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return {"mean": float(arr.mean()), "ci_low": lo, "ci_high": hi, "n": int(arr.size)}


def bootstrap_delta_ci(
    paired: list[tuple[float, float]],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 12345,
) -> dict[str, float]:
    """Bootstrap CI for the mean of (b - a) over paired per-instance scores."""
    import numpy as np

    diffs = np.asarray([(b - a) for a, b in paired if a is not None and b is not None], dtype=float)
    if diffs.size == 0:
        return {"mean_delta": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diffs.size, size=(n_boot, diffs.size))
    means = diffs[idx].mean(axis=1)
    return {
        "mean_delta": float(diffs.mean()),
        "ci_low": float(np.quantile(means, alpha / 2)),
        "ci_high": float(np.quantile(means, 1 - alpha / 2)),
        "n": int(diffs.size),
    }


_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def extract_first_identifier(text: str) -> str:
    """First identifier in a model's free-form answer (fallback name parse)."""
    m = _NAME_RE.search(text or "")
    return m.group(0) if m else ""