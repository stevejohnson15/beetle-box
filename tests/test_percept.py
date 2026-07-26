# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""Tests for the E2 private percept stream."""

import numpy as np
import pytest

from beetlebox.config import PerceptConfig
from beetlebox.envs.percept import PerceptStream


def test_sample_shapes_and_type_range():
    stream = PerceptStream(PerceptConfig(num_types=6, dim=8, noise=0.25),
                           np.random.default_rng(0))
    types, percepts = stream.sample(50)
    assert types.shape == (50,)
    assert percepts.shape == (50, 8)
    assert types.min() >= 0 and types.max() < 6
    assert stream.prototypes.shape == (6, 8)


def test_sampling_is_seeded():
    a = PerceptStream(PerceptConfig(), np.random.default_rng(0)).sample(20)
    b = PerceptStream(PerceptConfig(), np.random.default_rng(0)).sample(20)
    assert np.array_equal(a[0], b[0]) and np.allclose(a[1], b[1])


def test_clean_regime_same_type_closer_than_cross_type():
    # With small noise, two percepts of the same type are closer than two of
    # different types -- the geometry that makes "same again" learnable.
    cfg = PerceptConfig(num_types=6, dim=8, noise=0.2)
    stream = PerceptStream(cfg, np.random.default_rng(0))
    types, percepts = stream.sample(400)
    same, cross = [], []
    for i in range(0, 400, 2):
        for j in range(i + 1, min(i + 6, 400)):
            d = float(np.linalg.norm(percepts[i] - percepts[j]))
            (same if types[i] == types[j] else cross).append(d)
    assert np.mean(same) < np.mean(cross)


def test_invalid_config_raises():
    with pytest.raises(ValueError):
        PerceptStream(PerceptConfig(num_types=0), np.random.default_rng(0))
    with pytest.raises(ValueError):
        PerceptStream(PerceptConfig(dim=0), np.random.default_rng(0))
