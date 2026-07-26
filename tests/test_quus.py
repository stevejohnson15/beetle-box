# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""Tests for the E4 quus task."""

import numpy as np
import pytest

from beetlebox.config import QuusConfig
from beetlebox.envs.quus import QuusTask


def test_train_pairs_are_below_bend_and_test_above():
    task = QuusTask(QuusConfig(max_operand=12, bend=8))
    assert (np.max(task.train_pairs, axis=1) < 8).all()
    assert (np.max(task.test_pairs, axis=1) >= 8).all()
    # The two sets partition the full grid.
    assert len(task.train_pairs) + len(task.test_pairs) == 12 * 12


def test_plus_and_quus_agree_below_but_differ_above():
    task = QuusTask(QuusConfig(max_operand=12, bend=8, modulus=24, quus_value=0))
    below = task.train_pairs
    assert np.array_equal(task.plus(below), task.quus(below))  # agree below the bend
    above = task.test_pairs
    assert not np.array_equal(task.plus(above), task.quus(above))  # disagree above
    assert (task.quus(above) == 0).all()  # quus returns its constant above the bend


def test_scalar_and_onehot_feature_dims():
    scalar = QuusTask(QuusConfig(max_operand=12, encoding="scalar"))
    onehot = QuusTask(QuusConfig(max_operand=12, encoding="onehot"))
    assert scalar.feature_dim == 2
    assert onehot.feature_dim == 24
    assert scalar.features(scalar.train_pairs).shape[1] == 2
    assert onehot.features(onehot.train_pairs).shape[1] == 24


def test_invalid_config_raises():
    with pytest.raises(ValueError):
        QuusTask(QuusConfig(bend=1))
    with pytest.raises(ValueError):
        QuusTask(QuusConfig(bend=8, max_operand=8))  # no above-bend pairs
    with pytest.raises(ValueError):
        QuusTask(QuusConfig(encoding="bogus"))
