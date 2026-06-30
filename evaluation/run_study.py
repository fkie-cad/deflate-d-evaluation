"""RQ1 token-savings study: decompiler output x tier x tokenizer -> JSON.

Given a directory of decompiled C laid out as ``<dir>/<backend>/<name>.c`` (as
produced by :mod:`evaluation.decompile.driver`), apply every compression tier
and count tokens with every available provider, emitting one JSON record per
input binary for later plotting.

    python -m evaluation.run_study <decompiled-dir> <out.json>
    python -m evaluation.run_study --bin-dir <bins> <decompiled-dir> <out.json>
    python -m evaluation.run_study --providers openai <decompiled-dir> <out.json>

With ``--bin-dir`` the binaries are decompiled into ``<decompiled-dir>`` first
(Ghidra + Binary Ninja), then the study runs over the result.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from deflated import transform

from .keys import load_keys
from .render_tiers import render_tiers
from .token_counters import build_counters, count_all

TIERS = ("T0", "T1", "T2", "T3", "T4")


def _counts_for(text: str, counters, cache: dict) -> dict[str, int]:
    """Token count per provider for ``text``, memoized on the exact string."""
    if text in cache:
        return cache[text]
    result = {tc.provider: tc.tokens for tc in count_all(text, counters)}
    cache[text] = result
    return result


def study_file(raw: str, counters, cache: dict) -> dict:
    """Compute per-tier char counts and per-provider token counts."""
    out: dict[str, dict] = {}
    for tier in TIERS:
        text = raw if tier == "T0" else transform(raw, tier)
        out[tier] = {"chars": len(text), "tokens": _counts_for(text, counters, cache)}
    return out


def discover(decompiled_dir: Path) -> dict[str, dict[str, Path]]:
    """Map ``binary_name -> {backend: path}`` from ``<dir>/<backend>/<name>.c``."""
    files: dict[str, dict[str, Path]] = {}
    for backend_dir in sorted(p for p in decompiled_dir.iterdir() if p.is_dir()):
        backend = backend_dir.name
        for c_file in sorted(backend_dir.glob("*.c")):
            files.setdefault(c_file.stem, {})[backend] = c_file
    return files


def load_provenance(manifest_paths) -> tuple[dict, list]:
    """Build ``binary_name -> provenance`` from corpus manifests.

    Returns ``(prov_map, corpora_meta)``. Each manifest (as emitted by the
    corpus builders) tags every binary with the corpus's corpus/version/
    opt_level/stripped fields plus the per-binary sha256.
    """
    prov: dict[str, dict] = {}
    corpora: list[dict] = []
    for mp in manifest_paths or []:
        m = json.loads(Path(mp).read_text(encoding="utf-8"))
        tag = {k: m.get(k) for k in ("corpus", "version", "opt_level",
                                     "stripped", "arch")}
        corpora.append(tag)
        for name, info in m.get("binaries", {}).items():
            prov[Path(name).stem] = {**tag, **info}
    return prov, corpora


def run_study(decompiled_dir, out_json, *, providers=None, manifests=None,
              render_dir=None, render=True) -> dict:
    decompiled_dir = Path(decompiled_dir)
    if render:
        # Materialize the per-tier renderings for manual inspection, kept in
        # sync with the corpus on every study run.
        out = render_dir or (decompiled_dir.parent / "deflated")
        render_tiers(decompiled_dir, out, quiet=True)
        print(f"[study] rendered tiers under {out}", file=sys.stderr)
    load_keys()  # populate ANTHROPIC_API_KEY / GEMINI_API_KEY from key files
    counters, errors = build_counters(providers)
    if not counters:
        raise SystemExit(f"no token counters available: {errors}")
    for provider, reason in errors.items():
        print(f"[study] provider skipped: {provider}: {reason}", file=sys.stderr)

    prov, corpora = load_provenance(manifests)

    files = discover(decompiled_dir)
    backends_seen: set[str] = set()
    records = []
    cache: dict = {}  # text -> {provider: tokens}, shared across all files
    for name, by_backend in files.items():
        record = {"binary": name}
        if name in prov:
            record["provenance"] = prov[name]
        for backend, path in sorted(by_backend.items()):
            backends_seen.add(backend)
            raw = path.read_text(encoding="utf-8", errors="replace")
            record[backend] = study_file(raw, counters, cache)
        records.append(record)
        print(f"[study] {name}: {', '.join(sorted(by_backend))}", file=sys.stderr)

    result = {
        "meta": {
            "providers": {c.provider: c.model for c in counters},
            "provider_errors": errors,
            "tiers": list(TIERS),
            "decompilers": sorted(backends_seen),
            "corpora": corpora,
        },
        "files": records,
    }
    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[study] wrote {out_path} ({len(records)} files)", file=sys.stderr)
    return result


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("decompiled_dir", help="dir of <backend>/<name>.c outputs")
    ap.add_argument("out_json", help="output JSON path")
    ap.add_argument("--bin-dir", help="decompile these binaries first")
    ap.add_argument("--manifest", action="append", default=[],
                    help="corpus_manifest.json to tag records with provenance "
                         "(repeatable; one per corpus)")
    ap.add_argument("--providers",
                    help="comma-separated subset of openai/claude/gemini "
                         "(default: all)")
    ap.add_argument("--render-dir",
                    help="where to write per-tier renderings for inspection "
                         "(default: sibling 'deflated' of the decompiled dir)")
    ap.add_argument("--no-render", action="store_true",
                    help="skip materializing the per-tier renderings")
    args = ap.parse_args(argv[1:])

    providers = (
        [p.strip() for p in args.providers.split(",") if p.strip()]
        if args.providers else None
    )

    manifests = list(args.manifest)
    if args.bin_dir:
        from .decompile.driver import decompile_dir
        decompile_dir(args.bin_dir, args.decompiled_dir)
        # A corpus builder drops corpus_manifest.json in the bin dir; use it.
        auto = Path(args.bin_dir) / "corpus_manifest.json"
        if auto.exists() and str(auto) not in manifests:
            manifests.append(str(auto))

    run_study(args.decompiled_dir, args.out_json,
              providers=providers, manifests=manifests,
              render_dir=args.render_dir, render=not args.no_render)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
