"""Binary Ninja headless decompilation.

Opens a binary, runs full analysis, and concatenates the **Pseudo C** rendering
of every function into a single translation unit --- the decompiler-output view
(the direct analogue of Ghidra's C and Hex-Rays pseudocode), not raw HLIL.
``single_function_language_representation`` is BN's default language renderer,
which is Pseudo C (real C syntax: braces, ``;``, C casts); ``single_function_hlil``
would instead give the indentation-based IL. Matches the bundled
``deflated/examples/vtables_binja.c`` sample.

Requires the ``binaryninja`` Python API to be importable (link it into the
environment once with the bundled ``install_api.py``) and a license that permits
headless automation. Import is lazy so this module can be inspected without it.

Run as a module so it can be driven in a subprocess with a clean interpreter::

    python -m evaluation.decompile.binja_export <binary> <out.c>
"""

from __future__ import annotations

import sys


def decompile(path: str) -> str:
    """Return the concatenated HLIL pseudo-C for every function in ``path``."""
    import binaryninja as bn

    bv = bn.load(path)
    try:
        bv.update_analysis_and_wait()

        settings = bn.DisassemblySettings()
        # Strip per-line addresses and IL opcode columns: we want C-like text,
        # not an address-annotated listing (those bytes are pure token overhead
        # and the address annotations are exactly what T2 'comments' would drop).
        settings.set_option(bn.DisassemblyOption.ShowAddress, False)
        settings.set_option(bn.DisassemblyOption.ShowOpcode, False)

        # UI "tags" (e.g. the no-entry marker BN auto-attaches to unimplemented
        # instructions) are rendered into the linear view as TagToken glyphs.
        # They are editor annotations, not part of the C, so drop them while
        # keeping every other token verbatim (the textual "/* unimplemented */"
        # comment itself is plain TextTokens and is preserved).
        tag_token = bn.InstructionTextTokenType.TagToken

        def line_text(line) -> str:
            return "".join(
                t.text for t in line.contents.tokens if t.type != tag_token
            )

        # Stable, reproducible order: by entry address.
        funcs = sorted(bv.functions, key=lambda f: f.start)

        blocks: list[str] = []
        for func in funcs:
            # Materialise HLIL first: the Pseudo C language representation is
            # generated lazily on a worker thread, and reading the linear view
            # before it finishes yields placeholder "Loading..." lines. Touching
            # func.hlil forces that generation to complete synchronously.
            try:
                _ = func.hlil
            except Exception:
                pass
            obj = bn.LinearViewObject.single_function_language_representation(
                func, settings
            )
            cursor = bn.LinearViewCursor(obj)
            cursor.seek_to_begin()
            lines: list[str] = []
            while cursor.valid:
                for line in cursor.lines:
                    lines.append(line_text(line))
                if not cursor.next():
                    break
            text = "\n".join(lines).rstrip()
            if text:
                blocks.append(text)
        return "\n\n".join(blocks) + "\n"
    finally:
        # Release the analysis/file handle so batch runs don't leak views.
        if hasattr(bv, "file") and bv.file is not None:
            bv.file.close()


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: python -m evaluation.decompile.binja_export <binary> <out.c>",
              file=sys.stderr)
        return 2
    binary, out_path = argv[1], argv[2]
    text = decompile(binary)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"wrote {out_path} ({len(text)} bytes)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
