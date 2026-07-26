# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""E3 harness + analysis tests: earnable cancellation, determinism, scoring."""

from beetlebox.analysis.e3 import score_game
from beetlebox.config import BoxConfig, E3ExperimentConfig, E3RunConfig
from beetlebox.harness import E3RunManager
from beetlebox.runlog import RunLogger


def _cfg(condition: str, game: str = "private_referent", seed: int = 0) -> E3RunConfig:
    cfg = E3RunConfig(seed=seed)
    cfg.experiment = E3ExperimentConfig(game=game, num_steps=800, eval_every=400)
    cfg.box = BoxConfig(condition=condition, box_dim=16)
    return cfg


def test_private_referent_earnable_cancellation():
    # Informative boxes must beat empty ones: the box CAN influence output.
    shared = E3RunManager(_cfg("shared")).run()["final_accuracy"]
    empty = E3RunManager(_cfg("empty")).run()["final_accuracy"]
    assert shared > 0.5
    assert empty <= 0.30  # near chance (1/8)
    assert shared - empty >= 0.25


def test_private_referent_no_leak_message_is_load_bearing():
    # The fix: the receiver cannot solve the task from its own boxes alone.
    # With the public message zeroed, accuracy must collapse to chance -- i.e. the
    # sender->receiver channel is doing the work (no receiver-side leak).
    cfg = _cfg("shared")
    cfg.experiment.num_steps = 2000
    mgr = E3RunManager(cfg)
    mgr.run()
    abl = mgr.channel_ablation()
    assert abl["full"] > 0.6
    assert abl["message_zeroed"] <= abl["chance"] + 0.10  # message removed -> ~chance


def test_e3_run_is_deterministic():
    a = E3RunManager(_cfg("divergent")).run()
    b = E3RunManager(_cfg("divergent")).run()
    assert a["final_accuracy"] == b["final_accuracy"]


def test_matching_game_runs_and_beats_chance_when_shared():
    summary = E3RunManager(_cfg("shared", game="sensation_matching")).run()
    assert summary["chance"] == 0.5
    assert summary["final_accuracy"] > 0.6


def _write_run(tmp_path, condition, game, accuracy, chance):
    directory = tmp_path / condition
    with RunLogger(directory) as logger:
        cfg = {"box": {"condition": condition}, "experiment": {"game": game}}
        logger.write_manifest(cfg, seed=0, config_hash="x")
        logger.log("run_end", game=game, condition=condition,
                   final_accuracy=accuracy, chance=chance)
    return str(directory)


def test_score_game_beetle_drops_out(tmp_path):
    dirs = [
        _write_run(tmp_path, "shared", "private_referent", 1.0, 0.125),
        _write_run(tmp_path, "divergent", "private_referent", 1.0, 0.125),
        _write_run(tmp_path, "empty", "private_referent", 0.12, 0.125),
        _write_run(tmp_path, "noise", "private_referent", 0.13, 0.125),
    ]
    result = score_game(dirs)
    assert result["checks"]["earnable"]["pass"] is True
    assert result["checks"]["beetle_gap"]["verdict"] == "beetle_drops_out"


def test_score_game_shared_helps(tmp_path):
    dirs = [
        _write_run(tmp_path, "shared", "sensation_matching", 0.95, 0.5),
        _write_run(tmp_path, "divergent", "sensation_matching", 0.75, 0.5),
    ]
    result = score_game(dirs)
    assert result["checks"]["beetle_gap"]["verdict"] == "shared_inner_state_helps"


def test_unknown_game_raises():
    import pytest
    cfg = _cfg("shared", game="bogus_game")
    with pytest.raises(ValueError):
        E3RunManager(cfg)


def test_greedy_convention_rejects_matching_game():
    import pytest
    mgr = E3RunManager(_cfg("shared", game="sensation_matching"))
    with pytest.raises(ValueError):
        mgr.greedy_convention()


def test_channel_ablation_rejects_non_private_game():
    import pytest
    mgr = E3RunManager(_cfg("shared", game="public_referent_aux"))
    with pytest.raises(ValueError):
        mgr.channel_ablation()


def test_public_referent_aux_runs_and_beats_chance():
    summary = E3RunManager(_cfg("shared", game="public_referent_aux")).run()
    assert summary["chance"] == 0.125
    assert summary["final_accuracy"] > 0.5


def test_greedy_convention_shapes_for_private_referent():
    mgr = E3RunManager(_cfg("divergent"))
    conv = mgr.greedy_convention()
    assert len(conv["mapping"]) == mgr.num_states
    assert len(conv["guesses"]) == mgr.num_states


def test_score_game_rejects_mixed_games(tmp_path):
    import pytest
    dirs = [
        _write_run(tmp_path, "shared", "private_referent", 1.0, 0.125),
        _write_run(tmp_path, "divergent", "sensation_matching", 0.9, 0.5),
    ]
    with pytest.raises(ValueError):
        score_game(dirs)


def test_score_game_report_renders(tmp_path):
    from beetlebox.analysis.e3 import _format_report
    dirs = [
        _write_run(tmp_path, "shared", "private_referent", 1.0, 0.125),
        _write_run(tmp_path, "divergent", "private_referent", 0.9, 0.125),
        _write_run(tmp_path, "empty", "private_referent", 0.12, 0.125),
        _write_run(tmp_path, "noise", "private_referent", 0.13, 0.125),
    ]
    report = _format_report(score_game(dirs))
    assert "E3 report" in report
    assert "earnable-cancellation" in report
    assert "beetle gap" in report
