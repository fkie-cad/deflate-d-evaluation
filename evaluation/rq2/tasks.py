"""Per-task runners: apply tiers, query Opus, parse, score.

Each runner is *paired within function*: the same function is rendered at
T0--T4 with a fixed prompt, so the only varying input is the decompiler
rendering. Results are checkpointed incrementally to a JSONL file so a crash
or rate-limit never loses finished work.

Runners return a list of per-(function, tier) score dicts::

    [{"id", "tier", "decompiler", "score": float, "pred": str, "ref": str, ...}, ...]
"""

from __future__ import annotations

import json
from pathlib import Path

from .client import GeminiEmbedder, OpusClient, Usage, cosine
from .metrics import (
    CodeWordNet,
    bleu4,
    bootstrap_mean_ci,
    extract_first_identifier,
    name_prf,
)
from .prompts import PROMPTS, render
from .tiers import TIERS, tiered_versions

_CW_PATH = Path(__file__).resolve().parents[1] / "datasets" / "binllm" / "codewordnet_synonyms.txt"


def _codewordnet() -> CodeWordNet:
    return CodeWordNet(_CW_PATH)


def _load_done(path: Path) -> dict[tuple[str, str], dict]:
    """Resume support: { (id, tier): record } from an existing JSONL."""
    done: dict[tuple[str, str], dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            done[(rec["id"], rec["tier"])] = rec
    return done


def _append(path: Path, rec: dict) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# Prompt construction --- the single source of truth.
#
# Both the synchronous runners below and the batch enumerator
# (:mod:`evaluation.rq2.batch`) build prompts through these two functions, so a
# prompt can never drift between the sync and batched paths (a drift would make
# the batch prefill write cache entries the runner never looks up).
# --------------------------------------------------------------------------- #


def gen_prompt(task: str, rec: dict, tier: str, versions: dict) -> str:
    """The exact generation prompt ``run_<task>`` sends for ``(rec, tier)``."""
    return render(PROMPTS[task], code=versions[tier]["code"])


def ref_prompt(rec: dict) -> str:
    """The summarize reference prompt (from source); one per function, repeat 0."""
    return render(PROMPTS["summarize_ref"], scode=rec["scode"])


# --------------------------------------------------------------------------- #
# T-a: function naming
# --------------------------------------------------------------------------- #


def _mean(xs: list[float]) -> float:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def _usage_summary(usages: list[Usage]) -> dict:
    """Flatten per-repeat billed usage into a record fragment.

    Input tokens are constant across repeats (the prompt is fixed per
    (function, tier)), so the mean equals the per-call value; output tokens
    vary run to run as the model samples, so the mean is the headline.
    The raw per-call ``usages`` list is kept so per-request and within-function
    output variance can be analysed later.
    """
    n = len(usages) or 1
    inp = sum(u.input_tokens for u in usages)
    out = sum(u.output_tokens for u in usages)
    return {
        "input_tokens": inp / n,
        "output_tokens": out / n,
        "total_tokens": (inp + out) / n,
        "usages": [u.to_dict() for u in usages],
    }


def run_funcname(
    records: list[dict], out_path: Path, *, model: str = "claude-opus-4-8", repeats: int = 1
) -> list[dict]:
    client = OpusClient(model=model)
    cw = _codewynet_safe()
    done = _load_done(out_path)
    results: list[dict] = []
    for rec in records:
        versions = tiered_versions(rec["raw_code"])
        for tier in TIERS:
            key = (rec["id"], tier)
            if key in done:
                results.append(done[key])
                continue
            preds, ps, rs, fs, usages = [], [], [], [], []
            for i in range(max(1, repeats)):
                raw, usage = client.complete_with_usage(
                    gen_prompt("funcname", rec, tier, versions),
                    repeat=i,
                    meta={"task": "funcname", "id": rec["id"], "tier": tier},
                )
                pred = _parse_funcname_output(raw)
                p, r, f1 = name_prf(pred, rec["ref_name"], cw)
                preds.append(pred)
                ps.append(p)
                rs.append(r)
                fs.append(f1)
                usages.append(usage)
            rec_out = {
                "id": rec["id"],
                "tier": tier,
                "decompiler": rec["decompiler"],
                "ref": rec["ref_name"],
                "pred": preds[0],
                "preds": preds,
                "repeats": len(preds),
                "precision": _mean(ps),
                "recall": _mean(rs),
                "score": _mean(fs),
                **_usage_summary(usages),
            }
            _append(out_path, rec_out)
            results.append(rec_out)
    return results


def _parse_funcname_output(text: str) -> str:
    """'The predicted function name is foo_bar' -> 'foo_bar'; fallback to first id."""
    marker = "is "
    if marker in text:
        return text.split(marker, 1)[1].strip().split()[0]
    return extract_first_identifier(text)


def _codewynet_safe() -> CodeWordNet:
    try:
        return _codewordnet()
    except Exception:
        # No synonym file -> identity-only matching (no synonym expansion).
        cw = CodeWordNet.__new__(CodeWordNet)
        cw._word2cluster = {}
        return cw


# --------------------------------------------------------------------------- #
# T-b: variable naming
# --------------------------------------------------------------------------- #


def run_varname(
    records: list[dict], out_path: Path, *, model: str = "claude-opus-4-8", repeats: int = 1
) -> list[dict]:
    client = OpusClient(model=model)
    cw = _codewynet_safe()
    done = _load_done(out_path)
    results: list[dict] = []
    for rec in records:
        versions = tiered_versions(rec["raw_code"])
        gt = rec["gt_vars"]
        for tier in TIERS:
            key = (rec["id"], tier)
            if key in done:
                results.append(done[key])
                continue
            info = versions[tier]
            raws, fs, ps, rs, usages = [], [], [], [], []
            for i in range(max(1, repeats)):
                pred, usage = client.complete_with_usage(
                    gen_prompt("varname", rec, tier, versions),
                    repeat=i,
                    meta={"task": "varname", "id": rec["id"], "tier": tier},
                )
                pred_map = _parse_varname_output(pred)
                mapped = _to_original_keys(pred_map, info["placeholder_map"], gt)
                pv = _varname_per_instance(mapped, gt, cw)
                raws.append(pred)
                fs.append(pv["f1"])
                ps.append(pv["precision"])
                rs.append(pv["recall"])
                usages.append(usage)
            rec_out = {
                "id": rec["id"],
                "tier": tier,
                "decompiler": rec["decompiler"],
                "score": _mean(fs),
                "precision": _mean(ps),
                "recall": _mean(rs),
                "n_vars": len(gt),
                "pred_raw": raws[0],
                "preds_raw": raws,
                "repeats": len(raws),
                **_usage_summary(usages),
            }
            _append(out_path, rec_out)
            results.append(rec_out)
    return results


def _parse_varname_output(text: str) -> dict[str, str]:
    """Parse '<placeholder>: <name>' lines into {placeholder: name}.

    The name is reduced to its first identifier token so trailing type spellings
    or punctuation (``v4: buffer, int``) do not pollute the score.
    """
    out: dict[str, str] = {}
    for line in (text or "").splitlines():
        if ": " not in line:
            continue
        varname, rest = line.split(": ", 1)
        varname = varname.strip()
        name = extract_first_identifier(rest.strip())
        if varname and name:
            out[varname] = name
    return out


def _to_original_keys(
    pred_map: dict[str, str],
    placeholder_map: dict[str, str],
    gt: dict[str, tuple[str, str]],
) -> dict[str, str]:
    """Map predictions from compressed names back to original placeholders.

    ``placeholder_map`` is {orig: compressed}. At T0 it is empty and the code
    keeps original names, so a prediction keyed by an original name matches gt
    directly. At T3+ we invert the map so a prediction on compressed name ``a``
    resolves back to the original placeholder ``v4`` that gt is keyed by.
    """
    if not placeholder_map:
        return {k: v for k, v in pred_map.items() if k in gt}
    inverted = {comp: orig for orig, comp in placeholder_map.items()}
    out: dict[str, str] = {}
    for name_in_code, pred in pred_map.items():
        orig = inverted.get(name_in_code, name_in_code)
        if orig in gt:
            out[orig] = pred
    return out


def _varname_per_instance(mapped: dict[str, str], gt: dict, cw: CodeWordNet) -> dict:
    """Macro sub-token F1 over the function's variables (gt as the key set)."""
    ps, rs, fs = [], [], []
    for orig, (gt_name, _gt_type) in gt.items():
        pred_name = mapped.get(orig, "")
        p, r, f1 = name_prf(pred_name, gt_name, cw)
        ps.append(p)
        rs.append(r)
        fs.append(f1)
    n = len(fs) or 1
    return {
        "precision": sum(ps) / n,
        "recall": sum(rs) / n,
        "f1": sum(fs) / n,
        "n": len(fs),
    }


# --------------------------------------------------------------------------- #
# T-c: summarization
# --------------------------------------------------------------------------- #


def run_summarize(
    records: list[dict], out_path: Path, *, model: str = "claude-opus-4-8", repeats: int = 1
) -> list[dict]:
    client = OpusClient(model=model)
    done = _load_done(out_path)
    results: list[dict] = []

    # Reference summary per function. CAPYBARA ships a human, source-comment
    # docstring per function (``ref_summary``), so we use it verbatim and never
    # generate a reference. Datasets that ship none (e.g. BinaryLLMs-Eval) fall
    # back to generating one from source (``scode``), cached on disk.
    if records and all(r.get("ref_summary") for r in records):
        refs = {r["id"]: r["ref_summary"] for r in records}
    else:
        refs = _summarize_references(records, client, out_path.parent / "summarize_refs.jsonl")

    # Pass 2: per-tier summaries from the pcode rendering. Checkpoint the raw
    # (expensive) Opus output immediately; scores are added in pass 3.
    for rec in records:
        versions = tiered_versions(rec["raw_code"])
        ref = refs.get(rec["id"], "")
        for tier in TIERS:
            key = (rec["id"], tier)
            if key in done:
                results.append(done[key])
                continue
            preds, usages = [], []
            for i in range(max(1, repeats)):
                text, usage = client.complete_with_usage(
                    gen_prompt("summarize", rec, tier, versions),
                    repeat=i,
                    meta={"task": "summarize", "id": rec["id"], "tier": tier},
                )
                preds.append(text)
                usages.append(usage)
            rec_out = {
                "id": rec["id"],
                "tier": tier,
                "decompiler": rec["decompiler"],
                "ref": ref,
                "pred": preds[0],
                "preds": preds,
                "repeats": len(preds),
                **_usage_summary(usages),
            }
            _append(out_path, rec_out)
            results.append(rec_out)

    # Pass 3: batch-embed all (pred, ref) pairs, score in place, then rewrite the
    # file once with scored records so the checkpoint is the final artifact.
    _score_summaries(results)
    _dump_all(out_path, results)
    return results


def _dump_all(path: Path, results: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for rec in results:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _summarize_references(records: list[dict], client: OpusClient, ref_path: Path) -> dict[str, str]:
    done: dict[str, str] = {}
    if ref_path.exists():
        for line in ref_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                obj = json.loads(line)
                done[obj["id"]] = obj["ref"]
    missing = [r for r in records if r["id"] not in done]
    with ref_path.open("a", encoding="utf-8") as fh:
        for rec in missing:
            ref, usage = client.complete_with_usage(
                ref_prompt(rec),
                meta={"task": "summarize_ref", "id": rec["id"]},
            )
            done[rec["id"]] = ref
            fh.write(
                json.dumps(
                    {"id": rec["id"], "ref": ref, "usage": usage.to_dict()}, ensure_ascii=False
                )
                + "\n"
            )
    return done


def _score_summaries(results: list[dict]) -> None:
    """Compute embedding cosine + BLEU-4 for every record (in place).

    Each record may hold several repeat summaries in ``preds``; we score every
    repeat against the reference and average, so within-function run variance
    does not masquerade as a tier effect. Records already carrying ``cosine``
    (a prior scored run) are left untouched.
    """
    to_score = [r for r in results if "cosine" not in r]
    if not to_score:
        return
    embedder = GeminiEmbedder()
    # Build (pred, ref) pairs per (record, repeat).
    pairs: list[tuple[int, int, str, str]] = []  # (record_idx, repeat_idx, pred, ref)
    for ri, rec in enumerate(to_score):
        ref = (rec.get("ref") or "").strip() or " "
        preds = rec.get("preds") or [rec.get("pred") or ""]
        for k in range(len(preds)):
            pred = (preds[k] or "").strip() or " "
            pairs.append((ri, k, pred, ref))
    # Embed each UNIQUE text once: a function's reference repeats across its five
    # tiers, and T3/T4 predictions can be byte-identical, so deduping reduces the
    # embedding calls and keeps us well under the free per-minute quota.
    uniq_texts = sorted({t for (_ri, _k, pred, ref) in pairs for t in (pred, ref)})
    BATCH = 100
    vec_of: dict[str, list[float]] = {}
    for start in range(0, len(uniq_texts), BATCH):
        chunk = uniq_texts[start : start + BATCH]
        for t, v in zip(chunk, embedder.embed(chunk)):
            vec_of[t] = v
    # BLEU is computed locally from the strings (no embedding, no quota).
    cos: dict[tuple[int, int], float] = {}
    bleus: dict[tuple[int, int], float] = {}
    for (ri, k, pred, ref) in pairs:
        cos[(ri, k)] = cosine(vec_of[pred], vec_of[ref])
        bleus[(ri, k)] = bleu4(pred, ref)
    for ri, rec in enumerate(to_score):
        ks = [k for (rri, k, _p, _r) in pairs if rri == ri]
        rec["cosine"] = _mean([cos[(ri, k)] for k in ks])
        rec["bleu"] = _mean([bleus[(ri, k)] for k in ks])
        rec["score"] = rec["cosine"]  # primary metric
        rec["repeats"] = len(ks)