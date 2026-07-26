# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""E5 tests: grounded world, manager (both regimes), selectivity metrics, scoring."""

import pytest

from beetlebox.analysis import e5
from beetlebox.config import E5RunConfig, EnvConfig, config_hash, to_dict
from beetlebox.envs import SignalingEnv
from beetlebox.envs.grounded import GroundedWorld
from beetlebox.harness import E5RunManager
from beetlebox.runlog import RunLogger, run_dir


def _grid_env():
    return SignalingEnv.from_config(EnvConfig(mode="grid", num_attributes=2, num_values=4))


# -- grounded world --------------------------------------------------------- #
def test_grounded_world_payoff_structure():
    world = GroundedWorld(_grid_env())
    assert world.num_actions == 4
    assert world.payoff.shape == (16, 4)
    # The correct action pays +stake; every other action pays -stake.
    for r in range(16):
        a = world.correct_action[r]
        assert world.payoff[r, a] == pytest.approx(world.stake[r])
        others = [world.payoff[r, x] for x in range(4) if x != a]
        assert all(o == pytest.approx(-world.stake[r]) for o in others)
    assert world.max_mean_payoff == pytest.approx(float(world.stake.mean()))


def test_grounded_world_requires_grid():
    with pytest.raises(ValueError):
        GroundedWorld(SignalingEnv.from_config(EnvConfig(mode="flat", num_referents=8)))


# -- manager ---------------------------------------------------------------- #
def _cfg(grounded, steps=2500, seed=0):
    cfg = E5RunConfig(seed=seed)
    cfg.experiment.grounded = grounded
    cfg.experiment.num_steps = steps
    cfg.experiment.eval_every = 500
    cfg.env.mode = "grid"
    cfg.env.num_attributes = 2
    cfg.env.num_values = 4
    cfg.channel.vocab_size = 6
    cfg.channel.message_length = 2
    return cfg


def test_grounded_run_reaches_high_payoff():
    s = E5RunManager(_cfg(grounded=True)).run()
    assert s["grounded"] is True
    assert s["performance"] > 0.8  # normalized payoff near optimal
    assert s["final_mapping"] is not None


def test_ungrounded_run_identifies_referents():
    s = E5RunManager(_cfg(grounded=False)).run()
    assert s["grounded"] is False
    assert s["performance"] > 0.7  # identification accuracy


def test_run_is_deterministic():
    a = E5RunManager(_cfg(grounded=True, steps=1500)).run()
    b = E5RunManager(_cfg(grounded=True, steps=1500)).run()
    assert a["final_mapping"] == b["final_mapping"]


# -- metrics + scoring ------------------------------------------------------ #
def test_recoverability_full_vs_selective():
    env = _grid_env()
    # A full-identity mapping (unique message per referent) -> both attrs recoverable.
    full = [[i // 4, i % 4] for i in range(16)]
    recover, lexicon = e5.recoverability(full, env)
    assert lexicon == 16
    assert recover[0] == 1.0 and recover[1] == 1.0
    # A selective mapping keyed only on attribute 0 -> attr0 recoverable, attr1 not.
    sel = [[env.attributes_for(i)[0], 0] for i in range(16)]
    recover, lexicon = e5.recoverability(sel, env)
    assert lexicon == 4
    assert recover[0] == 1.0 and recover[1] == 0.0


def _run(tmp_path, cfg):
    cfg.output_dir = str(tmp_path)
    directory = run_dir(cfg.output_dir, config_hash(cfg), cfg.seed)
    with RunLogger(directory) as logger:
        logger.write_manifest(to_dict(cfg), seed=cfg.seed, config_hash=config_hash(cfg))
        E5RunManager(cfg, logger=logger).run()
    return str(directory)


def test_score_grounded_is_selective(tmp_path):
    result = e5.score_run(_run(tmp_path, _cfg(grounded=True)))
    assert result["checks"]["grounding_is_selective"]["pass"] is True
    assert result["selectivity_gap"] >= 0.5
    assert "E5 report" in e5._format_report(result)


def test_score_ungrounded_encodes_identity(tmp_path):
    result = e5.score_run(_run(tmp_path, _cfg(grounded=False)))
    assert result["checks"]["encodes_full_identity"]["pass"] is True
    assert result["selectivity_gap"] <= 0.2


def test_e5_main(tmp_path, monkeypatch, capsys):
    d = _run(tmp_path, _cfg(grounded=True, steps=1500))
    monkeypatch.setattr("sys.argv", ["e5", d])
    e5.main()
    assert "E5 report" in capsys.readouterr().out
