# Export the Hex-Rays decompiled C of every function to one file.
#
# An IDAPython script for headless (idat -A -S) or in-GUI use. It decompiles
# every function with Hex-Rays and concatenates each function's pseudocode into
# a single translation unit, mirroring the Ghidra exporter used for the rest of
# the RQ1 corpus (all functions, sorted by entry address, blank line between).
#
# Headless (one binary):
#   idat64 -A -S"export_hexrays.py <out.c>" <binary>
#
# The batch driver run_hexrays.py calls this for every binary in a directory.
# If no output-path arg is given (e.g. run from the GUI File>Script menu), it
# writes "<input-file>.c" next to the analyzed binary.
#
# Output format (must match Ghidra's ExportDecompiledC.java):
#   - every function the decompiler can handle, in ascending entry-address order
#   - each function's C, separated by a single blank line
#   - UTF-8, no address column, no extra headers
#
# Requires the Hex-Rays decompiler for the binary's architecture. These binaries
# are x86-64 ELF, so a standard IDA x64 Hex-Rays license suffices.

import idaapi
import idautils
import idc
import ida_hexrays
import ida_auto
import ida_pro
import sys


def _decompile_all():
    """Return concatenated pseudocode for all functions, addr-sorted, plus (ok, total)."""
    if not ida_hexrays.init_hexrays_plugin():
        raise RuntimeError(
            "Hex-Rays decompiler not available for this binary's architecture "
            "(x86-64 ELF needs the IDA x64 Hex-Rays decompiler)."
        )

    funcs = sorted(idautils.Functions())  # ascending by entry address
    chunks = []
    ok = 0
    for ea in funcs:
        try:
            cfunc = ida_hexrays.decompile(ea)
        except Exception:
            cfunc = None
        if cfunc is None:
            continue
        text = str(cfunc)
        if text and text.strip():
            chunks.append(text)
            ok += 1
    body = "\n".join(chunks)
    if body and not body.endswith("\n"):
        body += "\n"
    return body, ok, len(funcs)


def main():
    # Wait for auto-analysis to finish before decompiling.
    ida_auto.auto_wait()

    # Output path: first script arg, else "<input>.c" next to the binary.
    args = idc.ARGV[1:] if hasattr(idc, "ARGV") else []
    if args:
        out_path = args[0]
    else:
        out_path = idaapi.get_input_file_path() + ".c"

    body, ok, total = _decompile_all()
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(body)

    msg = "export_hexrays: %d/%d functions -> %s (%d chars)\n" % (
        ok, total, out_path, len(body),
    )
    sys.stderr.write(msg)
    print(msg)

    # Whether an explicit output path was passed (the batch driver always does);
    # used below to decide whether to close IDA.
    return bool(args)


_had_arg = main()

# In headless/batch mode exit IDA so the driver can move on. Detect batch via
# IDA's own flag; fall back to "an output-path arg was passed" (GUI File>Script
# runs pass none, so this leaves the GUI open).
_is_batch = False
try:
    _is_batch = bool(idaapi.cvar.batch)
except Exception:
    _is_batch = False
if _is_batch or _had_arg:
    ida_pro.qexit(0)
