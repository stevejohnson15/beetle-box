# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""Deterministic seeding.

Reproducibility is a guardrail, not a nicety (see ``plan/beetle-box.md`` §3):
every run must be reproducible from a ``seed`` + config. :func:`seed_everything`
seeds Python, NumPy, and (if available) PyTorch, and requests deterministic
algorithms. The chosen seed is recorded in every run manifest.
"""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int, *, deterministic: bool = True) -> int:
    """Seed all relevant RNGs. Returns the seed for convenient logging."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:  # torch is a heavy import; keep seeding usable without it.
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - GPU-only, not exercised in CI
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:  # pragma: no cover - torch is a hard dependency here
        pass

    return seed
