# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
from beetlebox.config import RunConfig, config_hash, from_dict, to_dict


def test_hash_is_stable_and_seed_independent():
    a = RunConfig(seed=0)
    b = RunConfig(seed=123)  # different seed only
    assert config_hash(a) == config_hash(b)  # seed excluded from hash


def test_hash_changes_with_condition():
    a = RunConfig()
    b = RunConfig()
    b.channel.vocab_size = a.channel.vocab_size + 1
    assert config_hash(a) != config_hash(b)


def test_hash_excludes_device_and_output_dir():
    a = RunConfig(device="cpu", output_dir="results")
    b = RunConfig(device="cuda", output_dir="/tmp/x")
    assert config_hash(a) == config_hash(b)


def test_roundtrip_dict():
    cfg = RunConfig(seed=7)
    cfg.experiment.feedback = False
    cfg.env.mode = "grid"
    rebuilt = from_dict(to_dict(cfg))
    assert rebuilt.seed == 7
    assert rebuilt.experiment.feedback is False
    assert rebuilt.env.mode == "grid"
    assert config_hash(rebuilt) == config_hash(cfg)


def test_from_dict_ignores_unknown_and_fills_defaults():
    cfg = from_dict({"seed": 3, "bogus": 1, "channel": {"vocab_size": 12}})
    assert cfg.seed == 3
    assert cfg.channel.vocab_size == 12
    assert cfg.channel.message_length == 1  # default preserved
