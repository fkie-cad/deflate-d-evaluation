# DEFLATE-D evaluation

This repository contains the evaluation harness for
[**DEFLATE-D**](https://github.com/fkie-cad/deflate-d), the deterministic
decompiler-output reformatter introduced in our SURE '26 paper
*"Stop Paying for Whitespace: Token-Efficient Decompiler Output for LLM-Assisted
Binary Analysis"* (see
[How to reference](#how-to-reference-this-approach-or-prototype-implementation)).
It holds the code behind the paper's two research questions:

- **RQ1: token savings.** Decompile a directory of binaries (Ghidra, Binary
  Ninja, and Hex-Rays/IDA), apply every compression tier, and count tokens across the OpenAI,
  Anthropic, and Google tokenizers. Counting is decoupled from generation, so
  RQ1 costs ≈ $0.
- **RQ2: functional check.** Run downstream tasks (function naming, variable
  naming, summarization) on raw vs. reformatted code and score against ground
  truth, to confirm the tiers preserve task-relevant information.

This repo ships **source only**. Datasets, decompiler output, and result
artifacts are not redistributed; see *Datasets* below and the pointers in each
subdirectory's README.

## Layout

```
evaluation/
  token_counters/   provider-agnostic token counting (OpenAI/Claude/Gemini)
  decompile/        RQ1 decompilers: Ghidra + Binary Ninja + Hex-Rays/IDA -> decompiled C
  rq2/              RQ2 functional study (tasks, prompts, metrics, runner)
  corpora/          script to assemble the binary corpus (coreutils)
  datasets/         loaders + where to obtain third-party data (README)
  run_study.py            RQ1 driver: decompile x tier x tokenizer -> JSON
  run_function_study*.py  per-function RQ1 study (_concurrent = full corpus)
  baselines.py            off-the-shelf baseline comparison for RQ1
  render_tiers.py         dump every tier rendering to disk for inspection
  count_tokens.py         token-counting CLI (RQ1 primitive)
  README.md               token-counting layer reference
```

## Install

Requires **Python 3.10+**. The harness reformats with the DEFLATE-D package, so
install that first:

```bash
pip install -e ../deflate-d          # the DEFLATE-D package
pip install -r evaluation/requirements.txt
```

The token-counting backends (`tiktoken`, `anthropic`, `google-genai`) are each
optional and imported lazily; install only the providers you measure. OpenAI
counting is fully local and offline.

Run everything from the repository root so the `evaluation` package resolves
(`python -m evaluation.<module>`).

### Decompilers (RQ1 only)

RQ1 decompilation needs the backends installed locally: **Ghidra 12.1.2**,
**Binary Ninja 5.3** (commercial, headless license), and **IDA/Hex-Rays 9.3**
(commercial). Token counting and RQ2 need none of them. Ghidra + Binary Ninja
run via `run_study`; Hex-Rays runs separately (it needs a license). See
[`evaluation/decompile/README.md`](evaluation/decompile/README.md) for setup and
the exact versions to pin.

## API keys

Only the **Claude** and **Gemini** counting/RQ2 endpoints need keys (OpenAI uses
the local `tiktoken`). Provide them either as environment variables:

```bash
export ANTHROPIC_API_KEY=...
export GEMINI_API_KEY=...
```

or by dropping them into `evaluation/CLAUDE_API_KEY` and
`evaluation/GEMINI_API_KEY` (both git-ignored, so they are never committed).
`evaluation/keys.py` loads the files without overwriting anything already set in
the environment.

## Quick start

```bash
# Count tokens for one file across all available providers (RQ1 primitive)
python -m evaluation.count_tokens path/to/func.c

# OpenAI only, no key, no network
python -m evaluation.count_tokens --providers openai a.c b.c
```

See [`evaluation/README.md`](evaluation/README.md) for the token-counting layer,
[`evaluation/decompile/README.md`](evaluation/decompile/README.md) for the RQ1
decompile pipeline, and [`evaluation/rq2/README.md`](evaluation/rq2/README.md)
for the functional study.

## Datasets

The RQ2 task datasets (SymGen, ReSym, CAPYBARA) are third-party and are **not**
included, nor is the BinaryLLMs-Eval resource used for CodeWordNet synonyms and
baselines. See [`evaluation/datasets/README.md`](evaluation/datasets/README.md)
for download locations, licenses, and where each file goes.

## How to reference this approach or prototype implementation

DEFLATE-D is described in the following paper, to appear at the 2nd Workshop on
Software Understanding and Reverse Engineering (SURE '26), co-located with ACM
CCS 2026. If you use it in academic work, please cite:

```bibtex
@inproceedings{enders2026deflated,
  title     = {Stop Paying for Whitespace: Token-Efficient Decompiler Output
               for LLM-Assisted Binary Analysis},
  author    = {Enders, Steffen and Behner, Eva-Maria C. and Padilla, Elmar},
  booktitle = {Proceedings of the 2nd Workshop on Software Understanding and
               Reverse Engineering (SURE), co-located with ACM CCS},
  year      = {2026},
}
```

## Development

This project was developed with the assistance of Anthropic's Claude, under the
authors' direction and review.

## License

Released under the MIT License. See [`LICENSE`](LICENSE).
