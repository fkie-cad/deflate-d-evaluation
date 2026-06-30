"""RQ2 driver: run a task across tiers and report quality with bootstrap CIs.

Examples::

    # Estimate token cost before spending (free; no generation):
    python -m evaluation.rq2.run_rq2 --task funcname --max-functions 100 --estimate-cost

    # Run all three tasks via the Batch API (default; 50% cheaper, offline):
    python -m evaluation.rq2.run_rq2 --task all --max-functions 400

    # Smoke test with live (non-batch) calls:
    python -m evaluation.rq2.run_rq2 --task funcname --max-functions 5 --sync

Every generation is cached by request content (results/api_cache/), so when you
change the scoring code you can delete results/rq2/<task>.jsonl and re-run: the
cached responses replay for free and only the scores recompute.

The summary table reports, per tier: the mean primary score with a 95%
bootstrap CI, and the mean delta vs raw (T0) with its CI. The claim we look for
is that lossless tiers (T1/T2) are within noise of T0 and that the lossy tiers
(T3/T4) are non-inferior within a pre-registered margin.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import LOADERS
from .metrics import bootstrap_delta_ci, bootstrap_mean_ci
from .tiers import TIERS
from . import tasks as T

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "rq2"

_RUNNERS = {
    "funcname": T.run_funcname,
    "varname": T.run_varname,
    "summarize": T.run_summarize,
}
# Primary metric field per task (what the headline CI is computed on).
_PRIMARY = {
    "funcname": "score",      # sub-token F1
    "varname": "score",       # macro sub-token F1
    "summarize": "score",     # embedding cosine
}
# Secondary metrics reported alongside the primary.
_SECONDARY = {
    "funcname": ["precision", "recall"],
    "varname": ["precision", "recall"],
    "summarize": ["bleu"],
}


def run_task(
    task: str,
    max_functions: int | None,
    model: str,
    out_dir: Path,
    repeats: int,
    *,
    batch: bool = True,
    poll_interval: float = 30.0,
    seed: int = 0,
) -> list[dict]:
    records = LOADERS[task](limit=max_functions, seed=seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{task}.jsonl"
    print(f"[{task}] {len(records)} functions x {repeats} repeat(s) -> {out_path}")
    # Batch (default): prefill the request cache for every generation this task
    # needs at 50% cost, then the runner replays it from cache for free. --sync
    # skips this and lets the runner make live calls (use for smoke tests).
    if batch:
        from .batch import prefill

        prefill(task, records, repeats, model=model, poll_interval=poll_interval)
    results = _RUNNERS[task](records, out_path, model=model, repeats=repeats)
    return results


_TOKEN_FIELDS = ("input_tokens", "output_tokens", "total_tokens")


def _token_delta(pairs: list[tuple[float, float]]) -> dict:
    """Paired T0->tier token change.

    ``pct_saved`` is the mean per-function percentage reduction; it is positive
    when the tier uses *fewer* tokens than T0 (a saving) and negative when it
    costs *more* (the expected sign for output tokens if compressing the input
    pushes the model to think/explain more).
    """
    if not pairs:
        return {"n": 0}
    t0 = sum(a for a, _ in pairs) / len(pairs)
    tn = sum(b for _, b in pairs) / len(pairs)
    per_fn_pct = [(a - b) / a * 100.0 for a, b in pairs if a > 0]
    return {
        "n": len(pairs),
        "t0_mean": t0,
        "tier_mean": tn,
        "abs_saved": t0 - tn,
        "pct_saved": (sum(per_fn_pct) / len(per_fn_pct)) if per_fn_pct else None,
    }


def _boilerplate_tokens(task: str, model: str) -> int | None:
    """Tokens of this task's fixed prompt with empty code.

    Billed input = this constant + the rendered code. Subtracting it isolates the
    *code-only* token reduction, which is the figure comparable to RQ1's per-code
    savings (the RQ1 corpus table counts the code rendering alone, not the prompt
    that wraps it). Returns ``None`` if the count endpoint is unreachable so the
    code-only column is simply omitted rather than failing the run.
    """
    try:
        from .client import OpusClient
        from .prompts import PROMPTS, render
        key = task if task in PROMPTS else "summarize"
        return OpusClient(model=model).count_tokens(render(PROMPTS[key], code=""))
    except Exception:  # noqa: BLE001 - network/key issues just drop the extra column
        return None


def summarize_results(results: list[dict], task: str, boilerplate: int | None = None) -> dict:
    """Per-tier means + CIs and pairwise deltas vs T0.

    ``boilerplate`` is the fixed prompt-instruction token count (empty code); when
    given, each tier's input-token saving also reports a *code-only* reduction with
    the boilerplate removed from both sides, so the table can show the saving net
    of the incompressible instruction (the RQ1-comparable number).
    """
    by_tier: dict[str, list[dict]] = {t: [] for t in TIERS}
    for rec in results:
        by_tier.setdefault(rec["tier"], []).append(rec)

    primary = _PRIMARY[task]
    per_tier: dict[str, dict] = {}
    for tier in TIERS:
        scores = [r.get(primary) for r in by_tier.get(tier, [])]
        scores = [s for s in scores if s is not None]
        entry = {"n": len(scores), "primary": primary}
        entry.update(bootstrap_mean_ci(scores))
        for sec in _SECONDARY[task]:
            sec_scores = [s for s in (r.get(sec) for r in by_tier.get(tier, [])) if s is not None]
            entry[sec] = bootstrap_mean_ci(sec_scores)
        # Mean billed token usage for this tier (over records that recorded it).
        recs = by_tier.get(tier, [])
        tok: dict[str, float | int | None] = {}
        for f in _TOKEN_FIELDS:
            vals = [r.get(f) for r in recs if r.get(f) is not None]
            tok[f] = (sum(vals) / len(vals)) if vals else None
        tok["n_usage"] = sum(1 for r in recs if r.get("input_tokens") is not None)
        entry["tokens"] = tok
        per_tier[tier] = entry

    # Pairwise deltas vs T0, paired within function.
    by_fn_tier = {(r["id"], r["tier"]): r for r in results}
    fn_ids = {r["id"] for r in results}
    deltas: dict[str, dict] = {}
    token_savings: dict[str, dict] = {}
    for tier in TIERS[1:]:
        paired = []
        tok_pairs: dict[str, list[tuple[float, float]]] = {f: [] for f in _TOKEN_FIELDS}
        for fid in fn_ids:
            a = by_fn_tier.get((fid, "T0"))
            b = by_fn_tier.get((fid, tier))
            if not (a and b):
                continue
            if a.get(primary) is not None and b.get(primary) is not None:
                paired.append((a[primary], b[primary]))
            for f in _TOKEN_FIELDS:
                if a.get(f) is not None and b.get(f) is not None:
                    tok_pairs[f].append((a[f], b[f]))
        deltas[tier] = bootstrap_delta_ci(paired)
        ts = {f: _token_delta(tok_pairs[f]) for f in _TOKEN_FIELDS}
        if boilerplate is not None:
            # Code-only saving: strip the fixed instruction from both T0 and the
            # tier so the percentage reflects the code rendering alone (RQ1-comparable).
            code_pairs = [
                (a - boilerplate, b - boilerplate)
                for a, b in tok_pairs["input_tokens"]
                if a - boilerplate > 0
            ]
            cd = _token_delta(code_pairs)
            ts["input_tokens"]["pct_saved_code"] = cd.get("pct_saved")
            ts["input_tokens"]["t0_mean_code"] = cd.get("t0_mean")
            ts["input_tokens"]["tier_mean_code"] = cd.get("tier_mean")
        token_savings[tier] = ts

    return {
        "task": task,
        "primary": primary,
        "per_tier": per_tier,
        "deltas_vs_t0": deltas,
        "token_savings_vs_t0": token_savings,
        "boilerplate_tokens": boilerplate,
    }


def print_table(summary: dict) -> None:
    task = summary["task"]
    primary = summary["primary"]
    print(f"\n=== {task}  (primary: {primary}) ===")
    print(f"{'tier':4} {'n':>4} {'mean':>8} {'95% CI':>20}")
    for tier in TIERS:
        e = summary["per_tier"][tier]
        ci = f"[{e['ci_low']:.3f}, {e['ci_high']:.3f}]"
        print(f"{tier:4} {e['n']:>4} {e['mean']:>8.3f} {ci:>20}")
    print(f"\n{'tier':4} {'Δmean vs T0':>14} {'95% CI':>22}")
    for tier in TIERS[1:]:
        d = summary["deltas_vs_t0"][tier]
        ci = f"[{d['ci_low']:.3f}, {d['ci_high']:.3f}]"
        print(f"{tier:4} {d['mean_delta']:>14.3f} {ci:>22}")

    # Billed token usage and input/output cost trade-off vs T0.
    def _tok(x: object) -> str:
        return f"{x:>9.0f}" if isinstance(x, (int, float)) else f"{'-':>9}"

    def _pct(x: object) -> str:
        return f"{x:>+9.1f}%" if isinstance(x, (int, float)) else f"{'-':>10}"

    print(f"\n{'tier':4} {'in tok':>9} {'out tok':>9} {'total':>9}")
    for tier in TIERS:
        tk = summary["per_tier"][tier].get("tokens", {})
        print(f"{tier:4} {_tok(tk.get('input_tokens'))} {_tok(tk.get('output_tokens'))} {_tok(tk.get('total_tokens'))}")
    print(f"\n{'tier':4} {'in saved':>10} {'out saved':>10} {'total saved':>10}  (+ = fewer tokens than T0)")
    for tier in TIERS[1:]:
        ts = summary["token_savings_vs_t0"][tier]
        print(f"{tier:4} {_pct(ts['input_tokens'].get('pct_saved'))} {_pct(ts['output_tokens'].get('pct_saved'))} {_pct(ts['total_tokens'].get('pct_saved'))}")


def estimate_cost(task: str, max_functions: int | None, model: str, repeats: int = 1,
                  seed: int = 0) -> None:
    """Free pre-flight: count input tokens per tier via the real Claude count endpoint."""
    from .client import OpusClient
    from .prompts import PROMPTS, render

    records = LOADERS[task](limit=max_functions, seed=seed)
    client = OpusClient(model=model)
    template = PROMPTS[task if task in PROMPTS else "summarize"]
    field = "raw_code"
    print(f"\n[{task}] {len(records)} functions; input tokens (prompt+code) per tier, model={model}")
    totals = {t: 0 for t in TIERS}
    for rec in records:
        from .tiers import tiered_versions
        vers = tiered_versions(rec[field])
        for tier in TIERS:
            totals[tier] += client.count_tokens(render(template, code=vers[tier]["code"]))
    for tier in TIERS:
        per = totals[tier] / max(1, len(records))
        print(f"  {tier}: {per:8.0f} tok/func  total {totals[tier] * max(1, repeats):>10}  (~${totals[tier] * max(1, repeats) * 5 / 1e6:.2f} input, {repeats}x repeats)")
    # Summarize needs reference-generation calls from source only when the dataset
    # ships no references (CAPYBARA ships human ``ref_summary``, so this is skipped).
    if task == "summarize" and not all(r.get("ref_summary") for r in records):
        ref_tot = sum(client.count_tokens(render(PROMPTS["summarize_ref"], scode=r["scode"])) for r in records)
        print(f"  refs (from source): {ref_tot / max(1, len(records)):.0f} tok/func  ~${ref_tot * 5 / 1e6:.2f} input")
    print("  (output tokens billed separately; extended thinking is not enabled, so thinking_tokens is 0.)")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="run_rq2", description="RQ2 downstream-quality study")
    p.add_argument("--task", choices=["funcname", "varname", "summarize", "all"], default="funcname")
    p.add_argument("--max-functions", type=int, default=None)
    p.add_argument("--model", default="claude-opus-4-8")
    p.add_argument("--out-dir", default=str(RESULTS_DIR))
    p.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="queries per (function, tier); averaged to damp run-to-run sampling variance",
    )
    p.add_argument("--estimate-cost", action="store_true", help="free pre-flight, no generation")
    p.add_argument(
        "--sync",
        action="store_true",
        help="make live (non-batch) API calls; use for smoke tests. Default is the "
        "Batch API (50%% cheaper) prefilling the request cache.",
    )
    p.add_argument(
        "--poll-interval",
        type=float,
        default=30.0,
        help="seconds between batch status polls (batch mode only)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seed for the sample permutation; --max-functions selects a prefix of it, "
        "so the same seed gives a reproducible, unbiased, nested subset (400 subset of 800)",
    )
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    task_list = ["funcname", "varname", "summarize"] if args.task == "all" else [args.task]

    if args.estimate_cost:
        for task in task_list:
            try:
                estimate_cost(task, args.max_functions, args.model, args.repeats, args.seed)
            except Exception as exc:  # noqa: BLE001
                print(f"[{task}] cost estimate unavailable: {exc}")
        return 0

    all_summaries = []
    for task in task_list:
        results = run_task(
            task,
            args.max_functions,
            args.model,
            out_dir,
            args.repeats,
            batch=not args.sync,
            poll_interval=args.poll_interval,
            seed=args.seed,
        )
        summary = summarize_results(results, task, boilerplate=_boilerplate_tokens(task, args.model))
        print_table(summary)
        all_summaries.append(summary)

    (out_dir / "summary.json").write_text(
        json.dumps(all_summaries, indent=2), encoding="utf-8"
    )
    print(f"\nsummary written to {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())