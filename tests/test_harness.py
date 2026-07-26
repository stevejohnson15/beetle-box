# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""End-to-end harness tests: emergence, determinism, and the feedback ablation."""

from beetlebox.config import RunConfig
from beetlebox.harness import RunManager


def _fast_cfg(seed: int = 0, feedback: bool = True) -> RunConfig:
    cfg = RunConfig(seed=seed)
    cfg.experiment.num_steps = 800
    cfg.experiment.eval_every = 200
    cfg.experiment.feedback = feedback
    return cfg


def test_convention_emerges_with_feedback():
    summary = RunManager(_fast_cfg(feedback=True)).run()
    # Convention from use: accuracy climbs far above chance (1/8 = 0.125).
    assert summary["final_accuracy"] > 0.5
    assert summary["final_accuracy"] > 3 * summary["chance"]


def test_ablation_stays_near_chance():
    summary = RunManager(_fast_cfg(feedback=False)).run()
    # No success/correction signal -> no learning -> near chance.
    assert summary["final_accuracy"] <= 0.30


def test_run_is_deterministic():
    a = RunManager(_fast_cfg(seed=1)).run()
    b = RunManager(_fast_cfg(seed=1)).run()
    assert a["final_accuracy"] == b["final_accuracy"]
    assert a["final_mapping"] == b["final_mapping"]


def test_different_seeds_are_independent():
    a = RunManager(_fast_cfg(seed=1)).run()
    b = RunManager(_fast_cfg(seed=2)).run()
    # Same condition, different seed: both should still learn a working convention.
    assert a["final_accuracy"] > 0.5 and b["final_accuracy"] > 0.5


def test_turnover_convention_survives_founders():
    cfg = RunConfig(seed=0)
    cfg.experiment.num_steps = 2000
    cfg.experiment.eval_every = 200
    cfg.experiment.turnover = True
    cfg.experiment.turnover_at = 0.5
    summary = RunManager(cfg).run()
    assert summary["turnover_step"] == 1000
    # A fresh receiver re-adopts the incumbent convention -> high final accuracy.
    assert summary["final_accuracy"] > 0.5


def test_sample_exchanges_greedy_is_deterministic():
    cfg = RunConfig(seed=0)
    cfg.experiment.num_steps = 0
    mgr = RunManager(cfg)
    a = mgr.sample_exchanges(4, greedy=True)
    b = mgr.sample_exchanges(4, greedy=True)
    assert [e["message"] for e in a] == [e["message"] for e in b]
