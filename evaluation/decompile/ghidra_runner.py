"""Drive Ghidra headless decompilation via ``analyzeHeadless``.

Imports a binary into a throwaway project, runs auto-analysis, then runs the
``ExportDecompiledC`` post-script to write one concatenated C file. No Ghidra
Python bindings are needed in our interpreter --- we shell out to the bundled
``analyzeHeadless`` launcher, which carries its own Jython.

Ghidra is located via ``$GHIDRA_INSTALL_DIR`` if set, else the newest
``ghidra_*_PUBLIC`` found under ``~/Downloads``.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent / "ghidra_scripts"
_SCRIPT_NAME = "ExportDecompiledC.java"


def _version_key(path: Path) -> tuple:
    nums = re.findall(r"\d+", path.name)
    return tuple(int(n) for n in nums) if nums else (0,)


def find_ghidra() -> Path:
    """Return the Ghidra install directory, or raise with guidance."""
    env = os.environ.get("GHIDRA_INSTALL_DIR")
    if env:
        p = Path(env).expanduser()
        if (p / "support" / "analyzeHeadless").exists():
            return p
        raise FileNotFoundError(
            f"GHIDRA_INSTALL_DIR={env!r} has no support/analyzeHeadless"
        )
    candidates = sorted(
        (p for p in (Path.home() / "Downloads").glob("ghidra_*_PUBLIC")
         if (p / "support" / "analyzeHeadless").exists()),
        key=_version_key,
    )
    if not candidates:
        raise FileNotFoundError(
            "No Ghidra install found. Set GHIDRA_INSTALL_DIR or place a "
            "ghidra_*_PUBLIC under ~/Downloads."
        )
    return candidates[-1]  # newest by version


def decompile(binary: str | os.PathLike, out_path: str | os.PathLike,
              *, ghidra_dir: Path | None = None, timeout: int = 1800) -> str:
    """Decompile ``binary`` to one C file at ``out_path``; return its text."""
    ghidra_dir = ghidra_dir or find_ghidra()
    headless = ghidra_dir / "support" / "analyzeHeadless"
    binary = Path(binary).resolve()
    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="ghidra_proj_") as proj:
        cmd = [
            str(headless), proj, "deflated_eval",
            "-import", str(binary),
            "-scriptPath", str(_SCRIPT_DIR),
            "-postScript", _SCRIPT_NAME, str(out_path),
            "-deleteProject",
        ]
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
    if not out_path.exists():
        raise RuntimeError(
            "Ghidra produced no output for %s\n--- stdout tail ---\n%s\n"
            "--- stderr tail ---\n%s"
            % (binary, proc.stdout[-2000:], proc.stderr[-2000:])
        )
    return out_path.read_text(encoding="utf-8", errors="replace")


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: python -m evaluation.decompile.ghidra_runner <binary> <out.c>",
              file=sys.stderr)
        return 2
    text = decompile(argv[1], argv[2])
    print(f"wrote {argv[2]} ({len(text)} bytes)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
