# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""The quus task for E4 (Kripke's rule-following skepticism).

Kripke's ``quus`` operator agrees with ``plus`` on every pair seen so far and
diverges only past a **bend-point** ``k``::

    a quus b = a + b        if max(a, b) < k
             = quus_value    otherwise

If a learner has only ever seen pairs with ``max(a, b) < k`` (all *below the
bend*), the data is exactly consistent with both ``plus`` and ``quus`` (and
infinitely many other rules). Nothing in the examples fixes which rule is being
followed — the skeptical point. We make it manipulable: train on the below-bend
pairs, then look at what the learner extrapolates on the *above-bend* pairs, where
plus and quus finally disagree.

Outputs live in ``[0, modulus)`` so the task is a finite classification (labels are
taken ``mod modulus``); with a large enough modulus this is ordinary addition.
NumPy only; the God's-eye ``plus``/``quus`` targets are used only for scoring.
"""

from __future__ import annotations

import numpy as np

from beetlebox.config import QuusConfig


class QuusTask:
    """Generates below-bend training pairs and above-bend test pairs."""

    def __init__(self, cfg: QuusConfig) -> None:
        if cfg.bend < 2:
            raise ValueError("bend must be >= 2")
        if cfg.max_operand <= cfg.bend:
            raise ValueError("max_operand must exceed bend (need above-bend pairs)")
        if cfg.encoding not in ("scalar", "onehot"):
            raise ValueError(f"unknown encoding: {cfg.encoding!r}")
        self.bend = cfg.bend
        self.max_operand = cfg.max_operand
        self.modulus = cfg.modulus
        self.quus_value = cfg.quus_value
        self.encoding = cfg.encoding
        self._pairs_below = self._enumerate(below=True)
        self._pairs_above = self._enumerate(below=False)

    def _enumerate(self, *, below: bool) -> np.ndarray:
        """All (a, b) pairs with max(a,b) below (or at/above) the bend-point."""
        pairs = []
        for a in range(self.max_operand):
            for b in range(self.max_operand):
                is_below = max(a, b) < self.bend
                if is_below == below:
                    pairs.append((a, b))
        return np.array(pairs, dtype=np.int64)

    def plus(self, pairs: np.ndarray) -> np.ndarray:
        """The ``plus`` target: ``(a + b) mod modulus``."""
        return (pairs[:, 0] + pairs[:, 1]) % self.modulus

    def quus(self, pairs: np.ndarray) -> np.ndarray:
        """The ``quus`` target: plus below the bend, ``quus_value`` at/above it."""
        below = np.max(pairs, axis=1) < self.bend
        return np.where(below, self.plus(pairs), self.quus_value % self.modulus)

    @property
    def train_pairs(self) -> np.ndarray:
        """All below-bend pairs — the (rule-underdetermining) training set."""
        return self._pairs_below

    @property
    def test_pairs(self) -> np.ndarray:
        """All above-bend pairs — where plus and quus finally disagree."""
        return self._pairs_above

    @property
    def feature_dim(self) -> int:
        """Width of the feature vector for the configured encoding."""
        return 2 if self.encoding == "scalar" else 2 * self.max_operand

    def features(self, pairs: np.ndarray) -> np.ndarray:
        """Encode operand pairs as ``[N, feature_dim]`` float features.

        ``scalar``: each operand as its value / ``max_operand`` (extrapolable, so a
        simplicity prior can extend plus past the bend). ``onehot``: independent
        per-value units, so above-bend operand values are unconstrained by training.
        """
        n = pairs.shape[0]
        if self.encoding == "scalar":
            return (pairs.astype(np.float32) / self.max_operand)
        feats = np.zeros((n, 2 * self.max_operand), dtype=np.float32)
        feats[np.arange(n), pairs[:, 0]] = 1.0
        feats[np.arange(n), self.max_operand + pairs[:, 1]] = 1.0
        return feats
