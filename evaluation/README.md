# DEFLATE-D evaluation

Tooling for the token-savings study (RQ1) and the functional check (RQ2).

The end-to-end RQ1 pipeline (decompile a directory of binaries with Ghidra,
Binary Ninja, and IDA/Hex-Rays, apply all compression levels, count tokens across
providers into one JSON) lives in [`decompile/`](decompile/README.md). The
token-counting layer it builds on is documented below.

## `token_counters/`: count tokens across providers

A provider-agnostic layer that maps a raw string to a token count for each
provider. Counting is decoupled from generation, so the RQ1 study costs ~$0:

| Provider | Backend | Network? | Cost | Key |
|----------|---------|----------|------|-----|
| OpenAI / GPT | `tiktoken` (local) | No | Free | none |
| Claude | `messages.count_tokens` | Yes | Free (not billed) | `ANTHROPIC_API_KEY` |
| Gemini | `models.count_tokens` | Yes | Free | `GEMINI_API_KEY` / `GOOGLE_API_KEY` |

Each backend is imported lazily, so the package loads even with none installed;
a counter raises a clear error only when you actually construct/use it.

> We tokenize **raw decompiler-output strings**. Any per-message/role overhead a
> counting endpoint adds is a constant that cancels out of a raw-vs-compressed
> delta, so the counts are directly comparable across levels.

### Install

```bash
pip install -r evaluation/requirements.txt   # or just the providers you need
```

OpenAI-only (fully offline) needs just `tiktoken`.

### CLI

Run from the repository root so the `evaluation` package resolves:

```bash
# All available providers (missing SDK/key are skipped with a warning)
python -m evaluation.count_tokens path/to/func.c

# OpenAI only, no key, no network
python -m evaluation.count_tokens --providers openai a.c b.c

# Override models; JSON output; stdin
python -m evaluation.count_tokens --model claude=claude-opus-4-8 --json -
```

### Library

```python
from evaluation.token_counters import build_counters, count_all

counters, errors = build_counters(["openai", "claude", "gemini"])
for provider, reason in errors.items():
    print("skipped", provider, reason)

text = open("func.c").read()
for tc in count_all(text, counters):
    print(tc.provider, tc.model, tc.tokens)
```

Accepted provider names: `openai`/`gpt`, `anthropic`/`claude`, `google`/`gemini`.

### Notes

- **Default models:** OpenAI `gpt-5.1`, Claude `claude-opus-4-8`, Gemini
  `gemini-3.1-pro-preview`. Claude's tokenizer is shared across Opus
  4.7/4.8/Fable 5 and differs from older models, so pick the model you report.
- Open-weight tokenizers (Llama/Qwen/DeepSeek via HuggingFace) are also local
  and free; a backend for them can be added under `token_counters/` next.
