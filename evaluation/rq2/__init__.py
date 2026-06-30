"""RQ2 --- downstream task quality across compression tiers.

Tests whether DEFLATE-D's cumulative tiers (T0--T4) preserve LLM task quality on
three published binary-analysis datasets, with a single fixed model (Claude
Opus 4.8) and a single fixed prompt per task so that only the *input format*
varies:

  T-a  function naming     SymGen          (Ghidra)   sub-token P/R/F1
  T-b  variable naming     ReSym           (IDA)      token-F1 vs DWARF
  T-c  summarization       BinaryLLMs-Eval (IDA)      embedding cosine + BLEU

Design: paired within function (raw -> T1 -> T2 -> T3 -> T4), bootstrap CIs on
per-tier means and pairwise deltas vs raw. Run::

    python -m evaluation.rq2.run_rq2 --task funcname --max-functions 100
"""

from __future__ import annotations