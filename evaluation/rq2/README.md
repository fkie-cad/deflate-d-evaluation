# RQ2: downstream task quality across compression tiers

Tests whether DEFLATE-D's cumulative tiers (T0–T4) preserve LLM task quality on
**three published binary-analysis datasets**, with one fixed model (Claude
Opus 4.8) and one fixed prompt per task, so the only thing that varies between
conditions is the decompiler rendering.

| task | dataset | decompiler | ground truth | metric |
|------|---------|-----------|--------------|--------|
| T-a function naming | SymGen (NDSS '25, Apache-2.0) | Ghidra | DWARF names | sub-token P/R/F1 (CodeWordNet) |
| T-b variable naming | ReSym (CCS '24) | IDA | DWARF var names | macro sub-token F1 |
| T-c summarization | CAPYBARA (BinT5) | Ghidra | human source comment | embedding cosine + BLEU |

Design is **paired within function** (raw → T1 → T2 → T3 → T4) with
non-parametric bootstrap CIs on per-tier means and pairwise deltas vs raw.

## Layout

```
rq2/
  client.py    Opus 4.8 generation (Anthropic API) + Gemini embedder
  tiers.py     apply T0–T4; capture T3's placeholder→short-name map (for var scoring)
  metrics.py   sub-token F1 + CodeWordNet synonyms, embedding cosine, BLEU, bootstrap CIs
  prompts.py   fixed per-task prompts (held constant across tiers)
  data.py      dataset loaders → uniform per-function records
  tasks.py     the three task runners (apply tiers → query Opus → parse → score)
  run_rq2.py   CLI driver
datasets/
  symgen/test_set.json                 SymGen 400-function test set
  capybara/dedup_stripped.jsonl        CAPYBARA dedup stripped (Ghidra, human comments)
  binllm/codewordnet_synonyms.txt      CodeWordNet synonym clusters
  resym/ReSym_data/...                 ReSym VarDecoder test JSONL  ← SEE BELOW
```

## Endpoints

`client.OpusClient` reads its key from `evaluation/CLAUDE_API_KEY` (loaded by
`evaluation.keys`). The free `count_tokens` preflight (`--estimate-cost`) uses
the same endpoint.

Embeddings use the Gemini `gemini-embedding-001` endpoint (free tier) with
`evaluation/GEMINI_API_KEY`, the same key family used for token counting.

## Getting the datasets

SymGen and CAPYBARA ship their usable test data in-repo (already vendored
under `datasets/`). **ReSym's VarDecoder test data is on Zenodo**, which can
block automated download on some networks, so it may need to be fetched by hand:

1. Download `https://zenodo.org/records/13923982/files/ReSym_data.zip` (≈120 MB).
2. Extract so the VarDecoder test `.jsonl` lives under
   `evaluation/datasets/resym/ReSym_data/` (`data.load_varname` searches that
   tree for `*.jsonl`, preferring one whose name contains `test`).

`load_varname` raises a clear error pointing here if the data is absent.

## Running

```bash
# Free pre-flight: input-token cost per tier (no generation, not billed):
python -m evaluation.rq2.run_rq2 --task funcname --max-functions 100 --estimate-cost

# Run one task (checkpoints to results/rq2/<task>.jsonl; safe to interrupt/resume):
python -m evaluation.rq2.run_rq2 --task funcname --max-functions 100

# All three tasks:
python -m evaluation.rq2.run_rq2 --task all --max-functions 120
```

Results checkpoint incrementally (a crash or rate-limit never loses finished
work; re-running resumes). The summary table reports per-tier mean ± 95% CI and
Δ-vs-raw ± 95% CI.

## Token accounting (input vs output cost)

Every generation call records the **real billed usage** from the API response
(`resp.usage`), not just the free `count_tokens` estimate. Each per-(function,
tier) record in `results/rq2/<task>.jsonl` carries:

- `input_tokens`, `output_tokens`, `total_tokens`: mean over `--repeats` for
  that function+tier (input is constant across repeats; output is the
  answer text and varies run to run);
- `usages`: the raw per-request list (one entry per repeat) for variance
  analysis, each with `input_tokens` / `output_tokens` / `cache_*` / `total_tokens`.

Reference-summary calls log their usage too (`summarize_refs.jsonl`, `usage`
field). The summary (`summary.json` + console) adds, per tier:

- `tokens`: mean input/output/total tokens;
- `token_savings_vs_t0`: paired-within-function change vs T0. `pct_saved` is
  **positive when the tier uses fewer tokens** and **negative when it costs
  more**. This is what answers the intro's open question: input savings (T1+)
  may be offset by *higher* output/thinking tokens, and this column shows the
  sign and size of that trade-off directly.

## Honest caveats (disclosed in the paper)

- **Two decompilers**: Ghidra for function naming (SymGen) and summarization
  (CAPYBARA), IDA for variable naming (ReSym). Consistent with RQ1's
  multi-decompiler framing; strengthens generality. (No published clean-licensed
  dataset covers all three tasks under one decompiler.)
- **Summarization references**: CAPYBARA pairs each Ghidra-decompiled function
  with its original source-comment summary (`ref_summary`, human-written), held
  fixed across tiers. These comments are short and uneven in quality, which lowers
  the absolute cosine level but applies uniformly across tiers, so the *relative*
  tier comparison is valid. Reported as a limitation.
- **Name tokenization**: we use the dependency-free camel/snake splitter
  (ported from BinaryLLMs-Eval's `split_func_name`) rather than their
  sentencepiece + suffix-merge variant; the tier-vs-tier comparison is invariant
  to this choice.
- **Opus 4.8 extended thinking**: opt-in and never enabled here, so
  `thinking_tokens` is 0. Prompts are minimal and outputs format-constrained
  regardless.