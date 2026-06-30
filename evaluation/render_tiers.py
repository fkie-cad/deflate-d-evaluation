"""Materialize the DEFLATE-D tier renderings to disk for manual inspection.

Given a directory of decompiled C laid out as ``<dir>/<backend>/<name>.c`` (the
same layout :mod:`evaluation.run_study` consumes), write every tier rendering of
every file to::

    <out-dir>/<backend>/<name>/T0.c   # raw decompiler output (reference)
    <out-dir>/<backend>/<name>/T1.c   # cosmetic
    <out-dir>/<backend>/<name>/T2.c   # + structural (lossless)
    <out-dir>/<backend>/<name>/T3.c   # + contextual (lossy)
    <out-dir>/<backend>/<name>/T4.c   # + reductive (lossy)

so that the tiers of a single binary sit side by side for diffing.

This is regenerated automatically by :mod:`evaluation.run_study` on every study
run (pass ``--no-render`` to skip it), and can be run standalone::

    python -m evaluation.render_tiers evaluation/decompiled evaluation/deflated
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from deflated import transform

# T0 is the untouched decompiler output, kept alongside the tiers as a diff base.
TIERS = ("T0", "T1", "T2", "T3", "T4")


def discover(decompiled_dir: Path) -> dict[str, dict[str, Path]]:
    """Map ``binary_name -> {backend: path}`` from ``<dir>/<backend>/<name>.c``."""
    files: dict[str, dict[str, Path]] = {}
    for backend_dir in sorted(p for p in decompiled_dir.iterdir() if p.is_dir()):
        backend = backend_dir.name
        for c_file in sorted(backend_dir.glob("*.c")):
            files.setdefault(c_file.stem, {})[backend] = c_file
    return files


def render_file(raw: str) -> dict[str, str]:
    """Return ``{tier: rendered_text}`` for one raw decompiler output."""
    return {tier: (raw if tier == "T0" else transform(raw, tier)) for tier in TIERS}


def render_tiers(decompiled_dir, out_dir, *, quiet: bool = False) -> int:
    """Write every tier of every file under ``out_dir``; return file count.

    The output directory is wiped of stale ``<backend>/<name>`` folders that no
    longer correspond to an input, so reruns stay in sync with the corpus.
    """
    decompiled_dir = Path(decompiled_dir)
    out_dir = Path(out_dir)
    files = discover(decompiled_dir)

    written = 0
    valid: set[Path] = set()
    for name, by_backend in files.items():
        for backend, path in sorted(by_backend.items()):
            raw = path.read_text(encoding="utf-8", errors="replace")
            dest = out_dir / backend / name
            dest.mkdir(parents=True, exist_ok=True)
            valid.add(dest)
            for tier, text in render_file(raw).items():
                (dest / f"{tier}.c").write_text(text, encoding="utf-8")
                written += 1
        if not quiet:
            print(f"[render] {name}: {', '.join(sorted(by_backend))}", file=sys.stderr)

    _prune_stale(out_dir, valid)
    if not quiet:
        print(f"[render] wrote {written} files under {out_dir}", file=sys.stderr)
    return written


def _prune_stale(out_dir: Path, valid: set[Path]) -> None:
    """Remove ``<backend>/<name>`` dirs under ``out_dir`` not in ``valid``."""
    if not out_dir.exists():
        return
    for backend_dir in out_dir.iterdir():
        if not backend_dir.is_dir():
            continue
        for name_dir in backend_dir.iterdir():
            if name_dir.is_dir() and name_dir not in valid:
                for child in name_dir.iterdir():
                    child.unlink()
                name_dir.rmdir()


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("decompiled_dir", help="dir of <backend>/<name>.c outputs")
    ap.add_argument("out_dir", nargs="?", default=None,
                    help="output dir (default: sibling 'deflated' of the input)")
    args = ap.parse_args(argv[1:])
    out_dir = args.out_dir or (Path(args.decompiled_dir).parent / "deflated")
    render_tiers(args.decompiled_dir, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
