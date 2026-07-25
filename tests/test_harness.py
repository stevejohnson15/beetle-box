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
