# Datasets

The RQ2 functional study uses three third-party task datasets (SymGen, ReSym,
CAPYBARA), plus the BinaryLLMs-Eval resource for CodeWordNet synonyms and
baselines. **None of them are redistributed here**; they carry their own
licenses and live with their original authors. This file tells you where to get each one and where to
put it; the loaders in [`../rq2/data.py`](../rq2/data.py) read from the paths
below (resolved relative to `evaluation/datasets/`).

Only our own preprocessing script is vendored:
`capybara/build_capybara_jsonl.py`.

| Task (RQ2) | Dataset | Decompiler | Goes in |
|------------|---------|------------|---------|
| function naming | SymGen | Ghidra | `symgen/test_set.json` |
| variable naming | ReSym | IDA | `resym/ReSym_data/` |
| summarization | CAPYBARA | Ghidra | `capybara/dedup_stripped.jsonl` |
| (baselines) | BinLLM / BinaryLLMs-Eval | n/a | `binllm/` |

## SymGen: function naming

Jiang, Jin, Lin, *Beyond Classification: Inferring Function Names in Stripped
Binaries via Domain Adapted LLMs*, NDSS 2025 (Computer Security Laboratory,
OSU). Licensed **Apache-2.0**.

Obtain the released test split (`test_set.json`, 400 functions) from the SymGen
artifact and place it at `symgen/test_set.json`.

## ReSym: variable naming

Xie et al., *ReSym: Harnessing LLMs to Recover Variable and Data Structure
Symbols from Stripped Binaries*, ACM CCS 2024.

Download `ReSym_data.zip` from
<https://zenodo.org/records/13923982/files/ReSym_data.zip> and extract it into
`resym/ReSym_data/`. (Zenodo may block automated download from some networks;
fetch it in a browser if so.)

## CAPYBARA: summarization

Al-Kaswan et al., *Extending Source Code Pre-Trained Language Models to
Summarise Decompiled Binaries* (BinT5), SANER 2023. Published on Hugging Face as
`AISE-TUDelft/Capybara`.

Regenerate the vendored JSONL we use (the `dedup_stripped` variant, whose
reference summaries are the original human-written source comments):

```bash
pip install huggingface_hub pyarrow          # build-time only
python -m evaluation.datasets.capybara.build_capybara_jsonl
```

This writes `capybara/dedup_stripped.jsonl`. The eval loader then reads that
JSONL with the standard library alone.

## BinLLM / BinaryLLMs-Eval

Used by the baseline comparisons. Obtain `dataset_x64_o2.json` and
`codewordnet_synonyms.txt` from the BinaryLLMs-Eval project and place them under
`binllm/`.
