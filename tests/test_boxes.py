# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
import numpy as np

from beetlebox.boxes import BoxCondition, BoxScheme


def _scheme(cond):
    return BoxScheme(cond, num_agents=2, num_states=4, box_dim=8,
                     rng=np.random.default_rng(0))


def test_shared_agents_get_identical_codes():
    s = _scheme("shared")
    states = np.array([0, 1, 2, 3])
    assert np.array_equal(s.signal(0, states), s.signal(1, states))
    assert s.is_informative


def test_divergent_agents_get_different_codes():
    s = _scheme("divergent")
    states = np.array([0, 1, 2, 3])
    assert not np.array_equal(s.signal(0, states), s.signal(1, states))
    # ...but each agent's own code is stable across calls (informative).
    assert np.array_equal(s.signal(0, states), s.signal(0, states))
    assert s.is_informative


def test_empty_box_is_zero_and_uninformative():
    s = _scheme("empty")
    sig = s.signal(0, np.array([0, 1, 2]))
    assert sig.shape == (3, 8)
    assert np.all(sig == 0.0)
    assert not s.is_informative


def test_noise_box_is_random_and_uninformative():
    s = _scheme("noise")
    a = s.signal(0, np.array([0, 0, 0]))
    b = s.signal(0, np.array([0, 0, 0]))
    # Same state, different draws -> carries no stable info about the state.
    assert not np.array_equal(a, b)
    assert not s.is_informative


def test_shared_encodes_state_not_identity():
    s = _scheme("shared")
    # Different states -> different codes (the box does encode the state).
    assert not np.array_equal(s.signal(0, np.array([0])), s.signal(0, np.array([1])))


def test_condition_enum_roundtrip():
    assert BoxCondition("shared") == BoxCondition.SHARED
