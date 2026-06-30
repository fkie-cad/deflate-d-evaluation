"""Fixed, minimal prompts for each RQ2 task.

The prompt is held constant across tiers within a task so that the only thing
that varies between conditions is the decompiler rendering. Output formats are
strictly constrained for parseability, and we ask for no explanation (to keep
output tokens --- and Opus thinking-driven verbosity --- bounded).
"""

from __future__ import annotations

# T-a: function-name prediction. SymGen's own instruction is reused verbatim
# (including the "to replace [MASK]" clause, since the [MASK] token is present in
# the code at every tier) for comparability with the dataset's reported numbers.
# The only addition is the trailing output-format line, which constrains the
# response for parseability and bounds Opus's thinking-driven output verbosity.
FUNCNAME = (
    "Suppose you are an expert in software reverse engineering. "
    "Here is a piece of decompiled code, you should infer code semantics and "
    "tell me the original function name from the contents of the function to "
    "replace [MASK]. Now the decompiled codes are as follows:\n\n"
    "{code}\n\n"
    "Reply with exactly one line of the form "
    "'The predicted function name is <name>' and nothing else."
)

# T-b: variable-name recovery. The model names every local/parameter variable in
# the function; we parse 'varname: name' lines and map back through the tier's
# placeholder map to score against DWARF.
VARNAME = (
    "You are an expert reverse engineer. Below is a decompiled C function in "
    "which the decompiler gave variables meaningless placeholder names. Infer a "
    "concise, descriptive name for each variable from how it is used. Output one "
    "line per variable in the form '<placeholder>: <name>' and nothing else.\n\n"
    "{code}"
)

# T-c: summarization. BinaryLLMs-Eval's own summarization instruction (<=96 words),
# so our numbers are comparable to theirs.
SUMMARIZE = (
    "Please imagine you are an experienced binary reverse engineer. The "
    "following is a stripped decompiled C function, your task is to understand "
    "it and generate a short comment to the function describing its "
    "functionality. No more than 96 words.\n\n"
    "```C\n{code}\n```\n\n"
    "Output only the comment."
)

# T-c reference generation: a summary of the *source* (ground-truth semantics),
# produced once per function and held fixed across all tiers. Same length bound.
SUMMARIZE_REF = (
    "You are an experienced reverse engineer. Below is the original C source of "
    "a function. Write a short comment describing its functionality. No more than "
    "96 words. Output only the comment.\n\n"
    "```C\n{scode}\n```"
)


PROMPTS = {
    "funcname": FUNCNAME,
    "varname": VARNAME,
    "summarize": SUMMARIZE,
    "summarize_ref": SUMMARIZE_REF,
}


def render(template: str, **kw) -> str:
    return template.format(**kw)