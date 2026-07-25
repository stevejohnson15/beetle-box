# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
import numpy as np

from beetlebox.config import EnvConfig
from beetlebox.envs import SignalingEnv


def test_flat_env_shapes():
    env = SignalingEnv(EnvConfig(mode="flat", num_referents=8))
    assert env.num_classes == 8
    assert env.feature_dim == 8
    feats = env.features_for(np.arange(8))
    assert feats.shape == (8, 8)
    assert np.array_equal(feats, np.eye(8))


def test_grid_env_shapes_and_distance():
    env = SignalingEnv(EnvConfig(mode="grid", num_attributes=2, num_values=4))
    assert env.num_classes == 16
    assert env.feature_dim == 2 * 4
    # Referent 0 = (0,0); the distance to itself is 0, to a one-attribute change is 1.
    d_same = env.referent_distance(0, 0)
    assert d_same == 0
    # Find a referent that differs in exactly one attribute from referent 0.
    diffs = [env.referent_distance(0, j) for j in range(16)]
    assert 1 in diffs and 2 in diffs


def test_sampling_is_seeded():
    env = SignalingEnv(EnvConfig(mode="flat", num_referents=6))
    idx_a, _ = env.sample_batch(np.random.default_rng(0), 10)
    idx_b, _ = env.sample_batch(np.random.default_rng(0), 10)
    assert np.array_equal(idx_a, idx_b)
