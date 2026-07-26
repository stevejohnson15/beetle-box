# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""Tests for the E2 diarist naming policies."""

import collections

import numpy as np
import pytest

from beetlebox.agents.diarist import (
    FixedQuantizerDiarist,
    NoisyImpressionDiarist,
    PrototypeDiarist,
    make_diarist,
)
from beetlebox.config import DiaristConfig, PerceptConfig
from beetlebox.envs.percept import PerceptStream


def _stream(noise=0.25, n=400, seed=0):
    s = PerceptStream(PerceptConfig(num_types=6, dim=8, noise=noise),
                      np.random.default_rng(seed))
    return s.sample(n)


def _consistency(types, terms):
    by = collections.defaultdict(list)
    for t, w in zip(types, terms, strict=True):
        by[t].append(w)
    return sum(collections.Counter(ws).most_common(1)[0][1] / len(ws)
               for ws in by.values()) / len(by)


def test_full_memory_manufactures_stability():
    types, percepts = _stream()
    d = PrototypeDiarist(memory_mode="full", threshold=2.0)
    terms = [d.assign_term(p) for p in percepts]
    assert _consistency(types, terms) >= 0.9


def test_no_memory_names_every_percept_afresh():
    types, percepts = _stream()
    d = PrototypeDiarist(memory_mode="none", threshold=2.0)
    terms = [d.assign_term(p) for p in percepts]
    assert len(set(terms)) == len(terms)  # a fresh term every step
    assert _consistency(types, terms) <= 0.1


def _name_all(diarist, percepts):
    return [diarist.assign_term(p) for p in percepts]


def test_windowed_memory_between_full_and_none():
    types, percepts = _stream()
    full = _name_all(PrototypeDiarist(memory_mode="full", threshold=2.0), percepts)
    win = _name_all(PrototypeDiarist(memory_mode="windowed", window=20, threshold=2.0),
                    percepts)
    none = _name_all(PrototypeDiarist(memory_mode="none", threshold=2.0), percepts)
    assert _consistency(types, none) < _consistency(types, win) < _consistency(types, full)


def test_fixed_quantizer_is_deterministic():
    d1 = FixedQuantizerDiarist(num_bins=4, scale=3.0)
    d2 = FixedQuantizerDiarist(num_bins=4, scale=3.0)
    x = np.array([0.1, -1.2, 2.0, 0.5, -0.3, 1.1, -2.5, 0.0], dtype=np.float32)
    assert d1.assign_term(x) == d2.assign_term(x)  # same rule, same percept


def test_noisy_impression_greedy_is_deterministic():
    d = NoisyImpressionDiarist(num_types=6, dim=8, temperature=0.0,
                               rng=np.random.default_rng(0))
    x = np.ones(8, dtype=np.float32)
    assert d.assign_term(x) == d.assign_term(x)  # temperature 0 -> argmax


def test_reset_parameters_clears_diary():
    types, percepts = _stream(n=50)
    d = PrototypeDiarist(memory_mode="full", threshold=2.0)
    for p in percepts:
        d.assign_term(p)
    assert len(d.memory) > 0
    d.reset_parameters()
    assert len(d.memory) == 0
    assert d._next_term == 0


def test_make_diarist_dispatch_and_unknown_policy():
    assert isinstance(make_diarist(DiaristConfig(policy="prototype"),
                                   num_types=6, dim=8), PrototypeDiarist)
    assert isinstance(make_diarist(DiaristConfig(policy="fixed_quantizer"),
                                   num_types=6, dim=8), FixedQuantizerDiarist)
    assert isinstance(make_diarist(DiaristConfig(policy="noisy_impression"),
                                   num_types=6, dim=8), NoisyImpressionDiarist)
    with pytest.raises(ValueError):
        make_diarist(DiaristConfig(policy="bogus"), num_types=6, dim=8)
