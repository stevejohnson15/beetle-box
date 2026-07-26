# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""The E1 signaling-game environment.

A referent space of ``K`` objects. In ``flat`` mode the objects are unstructured
one-hot referents. In ``grid`` mode they are points in an attribute x value grid,
so that referents carry a distance structure (Hamming over attributes) against
which the emergent message code can be compared -- this is what makes
topographic-similarity analysis of *compositional* structure meaningful when the
channel carries more than one symbol.

NumPy only; no natural-language content anywhere in the referent representation.
"""

from __future__ import annotations

import itertools

import numpy as np

from beetlebox.config import EnvConfig


class SignalingEnv:
    """Referent generator + scorer for the Lewis signaling game."""

    def __init__(self, cfg: EnvConfig) -> None:
        self.mode = cfg.mode
        if cfg.mode == "grid":
            self.num_attributes = cfg.num_attributes
            self.num_values = cfg.num_values
            self._attribute_table = np.array(
                list(itertools.product(range(cfg.num_values), repeat=cfg.num_attributes)),
                dtype=np.int64,
            )  # shape [K, num_attributes]
            self._num_classes = self._attribute_table.shape[0]
            self._features = self._grid_features(self._attribute_table)
        elif cfg.mode == "flat":
            self.num_attributes = 1
            self.num_values = cfg.num_referents
            self._num_classes = cfg.num_referents
            self._attribute_table = np.arange(cfg.num_referents, dtype=np.int64).reshape(-1, 1)
            self._features = np.eye(cfg.num_referents, dtype=np.float32)
        else:
            raise ValueError(f"unknown env mode: {cfg.mode!r}")

    # -- geometry ---------------------------------------------------------- #
    @property
    def num_classes(self) -> int:
        """Number of distinct referents ``K``."""
        return self._num_classes

    @property
    def feature_dim(self) -> int:
        """Dimension of a referent's feature vector (the sender's input width)."""
        return self._features.shape[1]

    def _grid_features(self, attrs: np.ndarray) -> np.ndarray:
        """Concatenated per-attribute one-hots."""
        k = attrs.shape[0]
        feats = np.zeros((k, self.num_attributes * self.num_values), dtype=np.float32)
        for a in range(self.num_attributes):
            feats[np.arange(k), a * self.num_values + attrs[:, a]] = 1.0
        return feats

    def features_for(self, indices: np.ndarray) -> np.ndarray:
        """Deterministic index -> feature-vector lookup."""
        return self._features[indices]

    def attributes_for(self, index: int) -> np.ndarray:
        """Return the attribute vector for a referent (a single value in flat mode)."""
        return self._attribute_table[index]

    def referent_distance(self, i: int, j: int) -> int:
        """Hamming distance over attributes.

        In ``grid`` mode this is the meaningful semantic distance. In ``flat``
        mode referents are unstructured, so this collapses to 0/1 (same/other)
        and topographic-similarity analysis is not meaningful -- the analysis
        layer guards against reporting it there.
        """
        return int(np.sum(self._attribute_table[i] != self._attribute_table[j]))

    # -- sampling ---------------------------------------------------------- #
    def sample_batch(
        self, rng: np.random.Generator, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample referent indices and their feature vectors."""
        indices = rng.integers(0, self._num_classes, size=batch_size)
        return indices, self.features_for(indices)
