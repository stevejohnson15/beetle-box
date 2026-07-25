# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""Credential loading for rich-mode (frontier API) agents.

The Anthropic API key is resolved from, in order:
  1. the ``ANTHROPIC_API_KEY`` environment variable;
  2. an ``anthropic.key`` file (searched from the current directory upward),
     accepting either a bare key or a ``ANTHROPIC_API_KEY=...`` / ``KEY: value``
     line.

The key file is gitignored. This module never logs or prints the key.
"""

from __future__ import annotations

import os
from pathlib import Path

KEY_FILENAME = "anthropic.key"


def _parse_key_file(text: str) -> str | None:
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # Accept "sk-...", "ANTHROPIC_API_KEY=sk-...", or "key: sk-...".
        for sep in ("=", ":"):
            if sep in line:
                line = line.split(sep, 1)[1].strip()
                break
        return line.strip().strip('"').strip("'")
    return None


def find_key_file(start: str | Path | None = None) -> Path | None:
    """Search for ``anthropic.key`` from ``start`` (default cwd) up to the root."""
    here = Path(start or Path.cwd()).resolve()
    for directory in (here, *here.parents):
        candidate = directory / KEY_FILENAME
        if candidate.is_file():
            return candidate
    return None


def load_anthropic_key(start: str | Path | None = None) -> str:
    """Return the Anthropic API key, or raise if none is configured."""
    env = os.environ.get("ANTHROPIC_API_KEY")
    if env:
        return env
    key_file = find_key_file(start)
    if key_file is not None:
        key = _parse_key_file(key_file.read_text(encoding="utf-8"))
        if key:
            return key
    raise RuntimeError(
        "No Anthropic API key found. Set ANTHROPIC_API_KEY or place an "
        f"{KEY_FILENAME!r} file in the project (it is gitignored)."
    )
