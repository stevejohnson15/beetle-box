# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
import numpy as np

from beetlebox.seeding import seed_everything


def test_numpy_determinism():
    seed_everything(42)
    a = np.random.rand(5)
    seed_everything(42)
    b = np.random.rand(5)
    assert np.array_equal(a, b)


def test_torch_determinism():
    import torch

    seed_everything(7)
    a = torch.rand(5)
    seed_everything(7)
    b = torch.rand(5)
    assert torch.equal(a, b)


def test_returns_seed():
    assert seed_everything(99) == 99


def test_seed_everything_non_deterministic_flag():
    # The deterministic=False path should still seed reproducibly.
    seed_everything(5, deterministic=False)
    a = np.random.rand(3)
    seed_everything(5, deterministic=False)
    b = np.random.rand(3)
    assert np.array_equal(a, b)
