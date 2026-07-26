# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""Tests for the E1 scorer (scores a run against the frozen pre-registration)."""

from beetlebox.analysis.e1 import _format_report, score_run
from beetlebox.config import RunConfig, config_hash, to_dict
from beetlebox.runlog import RunLogger, run_dir


def _run(tmp_path, *, feedback=True, mode="flat", steps=1500, seed=0):
    cfg = RunConfig(seed=seed, output_dir=str(tmp_path))
    cfg.experiment.feedback = feedback
    cfg.experiment.num_steps = steps
    cfg.experiment.eval_every = 300
    if mode == "grid":
        cfg.env.mode = "grid"
        cfg.env.num_attributes = 2
        cfg.env.num_values = 4
        cfg.channel.vocab_size = 6
        cfg.channel.message_length = 2
    directory = run_dir(cfg.output_dir, config_hash(cfg), cfg.seed)
    with RunLogger(directory) as logger:
        logger.write_manifest(to_dict(cfg), seed=cfg.seed, config_hash=config_hash(cfg))
        from beetlebox.harness import RunManager
        RunManager(cfg, logger=logger).run()
    return str(directory)


def test_score_run_emergence_passes_with_feedback(tmp_path):
    result = score_run(_run(tmp_path, feedback=True))
    assert result["feedback"] is True
    assert result["checks"]["emergence"]["pass"] is True
    assert result["final_accuracy"] > result["chance"]
    # The report renders without error and includes the licenses text.
    report = _format_report(result)
    assert "E1 report" in report
    assert "does / does not license" in report


def test_score_run_ablation_stays_at_chance(tmp_path):
    result = score_run(_run(tmp_path, feedback=False))
    assert result["feedback"] is False
    assert "ablation_at_chance" in result["checks"]
    assert result["checks"]["ablation_at_chance"]["pass"] is True


def test_score_run_reports_topographic_similarity_on_grid(tmp_path):
    result = score_run(_run(tmp_path, mode="grid", steps=2500))
    ts = result["topographic_similarity"]
    assert ts["applicable"] is True
    assert "topographic_similarity" in result["checks"]
    assert "rho" in _format_report(result)
