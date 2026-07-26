# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""Mechanistic sub-stack for E4 (quus / rule-following).

Small transformers trained from scratch on modular arithmetic in the grokking
regime, plus a direct Fourier read-out of the learned circuit — the interpretability
backbone linking Beetle-Box to the grokking literature (plan §4.4, §9).

- :class:`~beetlebox.mech.transformer.ModularAdditionTransformer` — the model.
- :class:`~beetlebox.mech.grokking.GrokkingRun` — the memorize→generalize training
  loop with its accuracy + description-length curves.
- :mod:`beetlebox.mech.circuits` — Fourier power spectrum, key frequencies, and the
  cross-seed **algorithmic agreement** ("Clock and the Pizza") metric.
"""

from beetlebox.mech.grokking import GrokkingRun
from beetlebox.mech.transformer import ModularAdditionTransformer

__all__ = ["ModularAdditionTransformer", "GrokkingRun"]
