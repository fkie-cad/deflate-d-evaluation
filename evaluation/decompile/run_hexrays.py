#!/usr/bin/env python3
"""Batch-decompile a directory of binaries with Hex-Rays, headless.

The IDA/Hex-Rays counterpart to :mod:`evaluation.decompile.ghidra_runner`: for
every binary it runs IDA in batch mode with ``export_hexrays.py`` and writes
``<out>/<name>.c`` (one translation unit per binary, all functions concatenated,
matching the Ghidra/Binary Ninja outputs in the RQ1 corpus). By default it reads
the coreutils corpus assembled by ``evaluation/corpora/coreutils.sh`` and writes
into ``evaluation/decompiled/hexrays/``, where ``evaluation.run_study`` picks the
files up alongside the ghidra/ and binja/ backends.

Usage (from the repository root):
    python -m evaluation.decompile.run_hexrays                 # coreutils_bin -> decompiled/hexrays
    python -m evaluation.decompile.run_hexrays --bins DIR --out DIR
    python -m evaluation.decompile.run_hexrays --idat /path/to/idat64  # if auto-detect fails

What it does for each binary:
  1. skips anything that is not an ELF/Mach-O object (the set is all ELF),
  2. optionally verifies the binary's sha256 against corpus_manifest.json,
  3. runs:  <idat> -A -S"export_hexrays.py <out.c>" -L<log> <binary>
  4. records success/failure in a summary printed at the end.

These binaries are **x86-64 ELF**: a standard IDA x64 Hex-Rays license suffices.
Verify once on a single binary before the full run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent  # the evaluation/ package root
EXPORT_SCRIPT = HERE / "export_hexrays.py"


def find_idat(explicit: str | None) -> str:
    """Locate an idat/idat64 executable (batch-mode IDA)."""
    if explicit:
        return explicit
    # Common names and locations across platforms.
    for name in ("idat64", "idat", "idat64.exe", "idat.exe"):
        found = shutil.which(name)
        if found:
            return found
    candidates = [
        "/Applications/IDA Professional*.app/Contents/MacOS/idat64",
        "/Applications/IDA Pro*.app/Contents/MacOS/idat64",
        str(Path.home() / "idapro*/idat64"),
        "C:/Program Files/IDA*/idat64.exe",
    ]
    for pat in candidates:
        matches = sorted(Path("/").glob(pat.lstrip("/"))) if pat.startswith("/") else sorted(Path(pat).parent.glob(Path(pat).name))
        if matches:
            return str(matches[-1])
    raise SystemExit(
        "Could not find idat/idat64. Pass it explicitly with --idat /path/to/idat64"
    )


def is_binary_object(path: Path) -> bool:
    """True if the file is an ELF (or Mach-O) executable object IDA can load.

    The corpus is x86-64 ELF; Mach-O magics are kept too so the same driver
    works if the binary set is ever swapped back.
    """
    try:
        with path.open("rb") as fh:
            magic = fh.read(4)
    except OSError:
        return False
    if magic == b"\x7fELF":
        return True
    # 32/64-bit LE/BE Mach-O and fat/universal magics.
    return magic in {
        b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe",
        b"\xfe\xed\xfa\xcf", b"\xfe\xed\xfa\xce",
        b"\xca\xfe\xba\xbe", b"\xbe\xba\xfe\xca",
    }


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bins", default=str(EVAL / "coreutils_bin"), help="directory of input binaries")
    ap.add_argument("--out", default=str(EVAL / "decompiled" / "hexrays"), help="output directory for <name>.c")
    ap.add_argument("--idat", default=None, help="path to idat/idat64 (auto-detected if omitted)")
    ap.add_argument("--manifest", default=str(EVAL / "coreutils_bin" / "corpus_manifest.json"),
                    help="corpus_manifest.json for sha256 verification (optional)")
    ap.add_argument("--no-verify", action="store_true", help="skip sha256 verification")
    ap.add_argument("--timeout", type=int, default=1800, help="per-binary timeout in seconds")
    args = ap.parse_args()

    idat = find_idat(args.idat)
    bins_dir = Path(args.bins)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    if not EXPORT_SCRIPT.exists():
        raise SystemExit(f"missing {EXPORT_SCRIPT}")

    manifest_hashes: dict[str, str] = {}
    if not args.no_verify and Path(args.manifest).exists():
        data = json.loads(Path(args.manifest).read_text())
        manifest_hashes = {k: v.get("sha256", "") for k, v in data.get("binaries", {}).items()}

    print(f"idat:      {idat}")
    print(f"binaries:  {bins_dir}")
    print(f"output:    {out_dir}")
    print(f"verify:    {'off' if args.no_verify else f'against {args.manifest}'}")
    print()

    done, skipped, failed, mismatched = [], [], [], []
    entries = sorted(p for p in bins_dir.iterdir() if p.is_file())
    for binp in entries:
        name = binp.name
        if name == "corpus_manifest.json":
            continue
        if not is_binary_object(binp):
            print(f"[skip] {name}: not an ELF/Mach-O object")
            skipped.append(name)
            continue
        if manifest_hashes:
            want = manifest_hashes.get(name)
            if want and sha256(binp) != want:
                print(f"[WARN] {name}: sha256 mismatch vs manifest -- decompiling anyway")
                mismatched.append(name)

        out_c = out_dir / f"{name}.c"
        log = logs_dir / f"{name}.log"
        # -A: autonomous/batch (no dialogs).  -S: run script with arg.  -L: log file.
        cmd = [idat, "-A", f'-S{EXPORT_SCRIPT} {out_c}', f"-L{log}", str(binp)]
        print(f"[run ] {name} ...", end=" ", flush=True)
        try:
            subprocess.run(cmd, timeout=args.timeout, check=False)
        except subprocess.TimeoutExpired:
            print("TIMEOUT")
            failed.append(name)
            continue
        # IDA leaves a .i64/.idb database next to the binary; clean it up.
        for ext in (".i64", ".idb", ".id0", ".id1", ".id2", ".nam", ".til"):
            db = binp.with_suffix(binp.suffix + ext)
            if db.exists():
                db.unlink()
        if out_c.exists() and out_c.stat().st_size > 0:
            print(f"ok ({out_c.stat().st_size} bytes)")
            done.append(name)
        else:
            print("FAILED (no/empty output -- check log)")
            failed.append(name)

    print("\n==================== summary ====================")
    print(f"decompiled: {len(done)}")
    print(f"skipped (non-object): {len(skipped)}  {skipped or ''}")
    if mismatched:
        print(f"sha256 mismatches: {len(mismatched)}  {mismatched}")
    if failed:
        print(f"FAILED: {len(failed)}  {failed}")
        print("  -> inspect hexrays/logs/<name>.log; common cause is a missing")
        print("     Hex-Rays decompiler license or an unsupported function.")
    print(f"\noutputs in: {out_dir}")
    print("The study uses the hexrays/ directory (the .c files; logs/ optional).")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
