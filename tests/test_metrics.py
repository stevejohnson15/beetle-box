# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
from beetlebox.analysis import metrics
from beetlebox.channels import SymbolChannel
from beetlebox.config import ChannelConfig, EnvConfig
from beetlebox.envs import SignalingEnv


def test_convention_stability_frozen_vs_wandering():
    frozen = [{"mapping": [[1], [2], [3]]} for _ in range(4)]
    assert metrics.convention_stability(frozen, window=3) == 1.0
    wandering = [
        {"mapping": [[1], [2], [3]]},
        {"mapping": [[1], [0], [3]]},
        {"mapping": [[4], [0], [3]]},
    ]
    assert metrics.convention_stability(wandering, window=3) < 1.0


def test_transmission_fidelity_split_on_turnover():
    evs = [
        {"step": 100, "accuracy": 0.9},
        {"step": 200, "accuracy": 0.95},
        {"step": 300, "accuracy": 0.4},
        {"step": 400, "accuracy": 0.92},
    ]
    fid = metrics.transmission_fidelity(evs, turnover_step=250)
    assert fid["applicable"]
    assert fid["pre_turnover_accuracy"] == 0.95
    assert fid["post_turnover_accuracy"] == 0.92


def test_transmission_fidelity_not_applicable_without_turnover():
    assert metrics.transmission_fidelity([], None) == {"applicable": False}


def test_topsim_not_applicable_for_flat_single_symbol():
    env = SignalingEnv.from_config(EnvConfig(mode="flat", num_referents=4))
    ch = SymbolChannel.from_config(ChannelConfig(vocab_size=4, message_length=1))
    out = metrics.topographic_similarity([[0], [1], [2], [3]], env, ch)
    assert out == {"applicable": False}


def test_topsim_perfect_correlation_on_grid():
    # 2 attributes x 2 values -> 4 referents; a compositional code where each
    # symbol position encodes one attribute yields perfect topographic similarity.
    env = SignalingEnv.from_config(EnvConfig(mode="grid", num_attributes=2, num_values=2))
    ch = SymbolChannel.from_config(ChannelConfig(vocab_size=2, message_length=2))
    mapping = [list(env.attributes_for(i)) for i in range(env.num_classes)]
    out = metrics.topographic_similarity(mapping, env, ch)
    assert out["applicable"]
    assert out["rho"] > 0.99


def test_convention_stability_nan_when_too_few_snapshots():
    import math
    val = metrics.convention_stability([{"mapping": [[0]]}], window=5)
    assert math.isnan(val)


def test_mapping_agreement_handles_length_mismatch():
    # Defensive: mismatched mapping lengths -> 0.0 rather than crashing.
    assert metrics._mapping_agreement([[0], [1]], [[0]]) == 0.0
