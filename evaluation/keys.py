"""Load provider API keys from local key files into the environment.

The counting endpoints read keys from environment variables. For convenience you
may drop your keys into ``evaluation/CLAUDE_API_KEY`` / ``evaluation/GEMINI_API_KEY``
(both git-ignored — never commit them); this maps those files onto the variables
the SDKs expect, *without* overwriting a value already set in the environment. If
the files are absent, the corresponding environment variable is simply left as-is,
so exporting ``ANTHROPIC_API_KEY`` / ``GEMINI_API_KEY`` works just as well. Key
values are never logged.
"""

from __future__ import annotations

import os
from pathlib import Path

_DIR = Path(__file__).resolve().parent

# key file -> environment variable the SDK reads.
_KEY_FILES = {
    "CLAUDE_API_KEY": "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY": "GEMINI_API_KEY",
}


def load_keys() -> list[str]:
    """Populate provider env vars from key files. Returns the vars set (names)."""
    loaded: list[str] = []
    for filename, env_var in _KEY_FILES.items():
        if os.environ.get(env_var):
            continue  # respect an explicitly-set environment
        path = _DIR / filename
        if not path.exists():
            continue
        value = path.read_text(encoding="utf-8").strip()
        if value:
            os.environ[env_var] = value
            loaded.append(env_var)
    return loaded
