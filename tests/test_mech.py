# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""Tests for the E4 mechanistic sub-stack (tiny/fast configs, no real grokking)."""

import torch

from beetlebox.config import GrokkingConfig
from beetlebox.mech import circuits
from beetlebox.mech.grokking import GrokkingRun, _all_pairs
from beetlebox.mech.transformer import ModularAdditionTransformer


def test_transformer_forward_shapes():
    model = ModularAdditionTransformer(modulus=13, d_model=32, n_heads=4)
    a = torch.tensor([0, 5, 12])
    b = torch.tensor([1, 5, 12])
    tokens = model.input_tokens(a, b)
    assert tokens.shape == (3, 3)
    assert tokens[:, -1].tolist() == [13, 13, 13]  # the '=' token id == modulus
    logits = model(tokens)
    assert logits.shape == (3, 13)


def test_all_pairs_covers_grid_with_correct_labels():
    a, b, labels = _all_pairs(7)
    assert len(labels) == 49
    assert torch.equal(labels, (a + b) % 7)


def test_grokking_run_trains_and_records_history():
    cfg = GrokkingConfig(modulus=11, d_model=32, n_heads=4, num_steps=300,
                         eval_every=100, weight_decay=1.0, batch_size=0,
                         train_frac=0.8, seed=0)
    run = GrokkingRun(cfg)
    s = run.run()
    assert len(s["history"]) == 3
    # It should at least fit the training set (memorize) on this tiny problem.
    assert s["final_train_acc"] > 0.5
    assert set(s["history"][0]) >= {"step", "train_acc", "val_acc", "weight_norm"}


def test_circuits_readout_shapes_and_bounds():
    model = ModularAdditionTransformer(modulus=17, d_model=32, n_heads=4)
    ps = circuits.fourier_power_spectrum(model)
    assert ps.shape == (17 // 2,)              # frequencies 1..p//2
    assert abs(ps.sum() - 1.0) < 1e-5          # normalized
    assert 0.0 <= circuits.concentration(model, top_k=3) <= 1.0
    assert all(1 <= f <= 17 // 2 for f in circuits.key_frequencies(model, top_k=5))


def test_algorithmic_agreement_bounds_and_self_identity():
    m = ModularAdditionTransformer(modulus=17, d_model=32, n_heads=4)
    assert circuits.algorithmic_agreement(m, m, top_k=5) == 1.0  # identical to itself
    m2 = ModularAdditionTransformer(modulus=17, d_model=32, n_heads=4)
    assert 0.0 <= circuits.algorithmic_agreement(m, m2, top_k=5) <= 1.0


def test_load_model_roundtrip(tmp_path):
    m = ModularAdditionTransformer(modulus=13, d_model=16, n_heads=2)
    path = tmp_path / "model.pt"
    torch.save(m.state_dict(), path)
    loaded = circuits.load_model(str(path), modulus=13, d_model=16, n_heads=2)
    x = m.input_tokens(torch.tensor([1]), torch.tensor([2]))
    assert torch.allclose(m(x), loaded(x))
