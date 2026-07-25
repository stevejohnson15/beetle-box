# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""Loader for frozen pre-registrations.

Scoring must use criteria fixed *before* the run (``plan/beetle-box.md`` §3.2).
This loads a versioned prereg YAML and refuses to proceed unless it is marked
``frozen: true`` -- a small guard against silently scoring against a draft.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_PREREG = "prereg/e1_signaling.yaml"


def load_prereg(path: str | Path = DEFAULT_PREREG) -> dict[str, Any]:
    """Load a frozen pre-registration document."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"pre-registration not found at {p!s}. Scoring requires a frozen prereg."
        )
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not data.get("frozen", False):
        raise ValueError(
            f"pre-registration {p!s} is not marked 'frozen: true'; refusing to score "
            "against an unfrozen prereg."
        )
    return data
