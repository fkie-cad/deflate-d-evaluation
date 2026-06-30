"""Decompile a directory of binaries with every available backend.

For each input binary ``<bin>`` this writes one concatenated translation unit
per backend to ``<out>/<backend>/<bin>.c``:

* ``ghidra`` --- via ``analyzeHeadless`` (see :func:`ghidra_runner.find_ghidra`).
* ``binja``  --- via the Binary Ninja headless API (if importable + licensed).
* ``hexrays`` is **not** produced here (needs an IDA/Hex-Rays license). Run it
  separately with :mod:`evaluation.decompile.run_hexrays` (or drop externally-
  produced ``<out>/hexrays/<bin>.c`` files in); the study picks them up
  automatically.

A backend that fails on one binary is logged and skipped; the run continues.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

from . import binja_export, ghidra_runner

# Inputs that are clearly not binaries to decompile.
_SKIP_SUFFIXES = {
    ".c", ".h", ".cpp", ".cc", ".txt", ".md", ".json", ".log",
    ".py", ".sh", ".o", ".a", ".zip", ".tar", ".gz",
}


def iter_binaries(bin_dir: Path):
    """Yield regular-file binaries under ``bin_dir`` (sorted, filtered)."""
    for p in sorted(bin_dir.iterdir()):
        if not p.is_file() or p.name.startswith("."):
            continue
        if p.suffix.lower() in _SKIP_SUFFIXES:
            continue
        yield p


def decompile_dir(bin_dir, out_dir, *, backends=("ghidra", "binja")) -> dict:
    """Decompile every binary in ``bin_dir`` with ``backends``.

    Returns a manifest ``{binary_name: {backend: out_path | None}}``.
    """
    bin_dir = Path(bin_dir)
    out_dir = Path(out_dir)
    binaries = list(iter_binaries(bin_dir))
    if not binaries:
        print(f"[driver] no binaries found in {bin_dir}", file=sys.stderr)

    manifest: dict[str, dict[str, str | None]] = {}
    for binary in binaries:
        name = binary.stem
        manifest[name] = {}
        for backend in backends:
            out_path = out_dir / backend / f"{name}.c"
            try:
                if backend == "ghidra":
                    ghidra_runner.decompile(binary, out_path)
                elif backend == "binja":
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_text(binja_export.decompile(str(binary)),
                                        encoding="utf-8")
                else:
                    print(f"[driver] unknown backend {backend!r}", file=sys.stderr)
                    manifest[name][backend] = None
                    continue
                manifest[name][backend] = str(out_path)
                print(f"[driver] {backend}: {name} -> {out_path}", file=sys.stderr)
            except Exception as exc:  # one failure must not abort the batch
                manifest[name][backend] = None
                print(f"[driver] {backend} FAILED on {name}: {exc}", file=sys.stderr)
                traceback.print_exc()
    return manifest


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print("usage: python -m evaluation.decompile.driver <bin-dir> <out-dir> "
              "[backend ...]", file=sys.stderr)
        return 2
    backends = tuple(argv[3:]) or ("ghidra", "binja")
    decompile_dir(argv[1], argv[2], backends=backends)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
