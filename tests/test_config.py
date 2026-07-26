# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
from beetlebox.config import (
    ChannelConfig,
    E3RunConfig,
    EnvConfig,
    RunConfig,
    config_hash,
    from_dict,
    from_dict_e3,
    to_dict,
)


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


def test_channel_bandwidth_and_env_num_classes():
    assert ChannelConfig(vocab_size=6, message_length=2).bandwidth == 36
    assert EnvConfig(mode="flat", num_referents=8).num_classes == 8
    assert EnvConfig(mode="grid", num_attributes=2, num_values=4).num_classes == 16


def test_e3_roundtrip_and_hash_seed_independent():
    cfg = E3RunConfig(seed=3)
    cfg.experiment.game = "sensation_matching"
    cfg.box.condition = "divergent"
    rebuilt = from_dict_e3(to_dict(cfg))
    assert rebuilt.experiment.game == "sensation_matching"
    assert rebuilt.box.condition == "divergent"
    assert config_hash(rebuilt) == config_hash(cfg)
    # seed excluded from the condition hash
    other = E3RunConfig(seed=99)
    other.experiment.game = "sensation_matching"
    other.box.condition = "divergent"
    assert config_hash(other) == config_hash(cfg)


def test_e3_from_dict_ignores_unknown_keys():
    cfg = from_dict_e3({"seed": 1, "bogus": 2, "box": {"condition": "empty", "extra": 9}})
    assert cfg.seed == 1
    assert cfg.box.condition == "empty"


def test_e2_roundtrip_and_hash():
    from beetlebox.config import E2RunConfig, from_dict_e2
    cfg = E2RunConfig(seed=2)
    cfg.diarist.policy = "prototype"
    cfg.diarist.memory = "windowed"
    cfg.percept.noise = 0.9
    rebuilt = from_dict_e2(to_dict(cfg))
    assert rebuilt.diarist.memory == "windowed"
    assert rebuilt.percept.noise == 0.9
    assert config_hash(rebuilt) == config_hash(cfg)


def test_e2_from_dict_ignores_unknown_keys():
    from beetlebox.config import from_dict_e2
    cfg = from_dict_e2({"seed": 1, "bogus": 2, "diarist": {"memory": "none", "x": 9}})
    assert cfg.seed == 1
    assert cfg.diarist.memory == "none"


def test_e4_roundtrip_and_hash():
    from beetlebox.config import E4RunConfig, from_dict_e4
    cfg = E4RunConfig(seed=4)
    cfg.quus.encoding = "onehot"
    cfg.quus.bend = 6
    cfg.experiment.num_seeds = 5
    rebuilt = from_dict_e4(to_dict(cfg))
    assert rebuilt.quus.encoding == "onehot"
    assert rebuilt.quus.bend == 6
    assert rebuilt.experiment.num_seeds == 5
    assert config_hash(rebuilt) == config_hash(cfg)


def test_e4_from_dict_ignores_unknown_keys():
    from beetlebox.config import from_dict_e4
    cfg = from_dict_e4({"seed": 1, "bogus": 2, "quus": {"encoding": "scalar", "x": 9}})
    assert cfg.seed == 1
    assert cfg.quus.encoding == "scalar"
