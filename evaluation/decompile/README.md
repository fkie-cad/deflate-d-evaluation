# Decompilation + token-savings study (RQ1)

Turn a directory of binaries into the RQ1 dataset: decompile each with every
available backend, apply all compression tiers, and count tokens across
providers into one JSON for plotting.

## Pipeline

```
binaries/  --decompile-->  decompiled/<backend>/<name>.c  --study-->  study.json
```

Each backend concatenates the decompiled output of **all functions** in a binary
into one translation unit, the same form as `deflated/examples/vtables_*.c`.

| Backend | How | Requirements |
|---------|-----|--------------|
| **Ghidra** | `analyzeHeadless` + the `ExportDecompiledC.java` post-script (Decompiler C) | Auto-found: `$GHIDRA_INSTALL_DIR`, else newest `~/Downloads/ghidra_*_PUBLIC` |
| **Binary Ninja** | headless API, **Pseudo C** rendering (not raw HLIL) | Needs `binaryninja` importable + a license permitting headless |
| **Hex-Rays / IDA** | `idat -A` batch + the `export_hexrays.py` IDAPython script, driven by `run_hexrays.py` | Needs an IDA x64 + Hex-Rays license (the corpus is x86-64 ELF); `idat64` auto-detected, or pass `--idat` |

> Ghidra 12 dropped Jython, so the post-script is **Java** (runs headless with no
> PyGhidra). Binary Ninja's "Pseudo C" is the decompiler-output view comparable
> to Ghidra/Hex-Rays; "HLIL" is an intermediate language and is *not* used.

> **Hex-Rays needs a license, so it runs separately** from `driver.py`/`run_study`
> (which do Ghidra + Binary Ninja). Run it once to populate `decompiled/hexrays/`,
> then the study ingests those files alongside the other backends:
> ```bash
> python -m evaluation.decompile.run_hexrays          # coreutils_bin/ -> decompiled/hexrays/
> python -m evaluation.decompile.run_hexrays --bins path/to/bins --out decompiled/hexrays --idat /path/to/idat64
> ```
> `export_hexrays.py` is the IDAPython exporter (decompiles every function, sorted
> by entry address, blank line between, matching `ExportDecompiledC.java`); it can
> also be run in-GUI via *File > Script file*.

> **Versions used in the paper.** Ghidra 12.1.2, Binary Ninja 5.3, and IDA/Hex-Rays 9.3. Backend auto-discovery selects the newest install, so pin these exact versions to reproduce the reported token counts.

## One-shot

Run from the repository root so both `deflated` and `evaluation` resolve:

```bash
python -m evaluation.run_study --bin-dir path/to/binaries decompiled/ study.json
```

Or in two steps (decompile once, re-run the study cheaply):

```bash
python -m evaluation.decompile.driver path/to/binaries decompiled/
python -m evaluation.run_study decompiled/ study.json
python -m evaluation.run_study --providers openai decompiled/ study.json   # GPT-only, offline
```

## Setup

- **Binary Ninja API:** link it into your interpreter once:
  ```bash
  python "/Applications/Binary Ninja.app/Contents/Resources/scripts/install_api.py"
  ```
  In a venv that excludes user-site, instead add a `.pth` pointing at
  `…/Binary Ninja.app/Contents/Resources/python` to the venv's `site-packages`.
- **API keys:** `evaluation/run_study` loads `evaluation/CLAUDE_API_KEY` and
  `evaluation/GEMINI_API_KEY` into `ANTHROPIC_API_KEY` / `GEMINI_API_KEY`
  automatically (an already-set env var wins). GPT counts are local/offline.

## Output schema

```jsonc
{
  "meta": {
    "providers": {"openai": "...", "anthropic": "...", "google": "..."},
    "provider_errors": {},               // providers skipped (missing key/SDK)
    "tiers": ["T0", "T1", "T2", "T3", "T4"],  // T0 = raw decompiler output
    "decompilers": ["binja", "ghidra"]        // present in this run
  },
  "files": [
    {
      "binary": "mathy",
      "ghidra": { "T0": {"chars": N, "tokens": {"openai": N, "anthropic": N, "google": N}},
                  "T1": {...}, "T2": {...}, "T3": {...}, "T4": {...} },
      "binja":  { ... }
      // "hexrays": { ... }  // only if decompiled/hexrays/mathy.c was provided
    }
  ]
}
```

One object per input binary; savings for any provider are `T0 − Tk`. A backend
that fails on a binary is logged and omitted from that record rather than
aborting the batch.
