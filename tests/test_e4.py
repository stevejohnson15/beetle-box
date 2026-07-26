# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""E4 behavioral tests: rule learner, harness, metrics, and scoring."""

import pytest

from beetlebox.agents.rule_learner import RuleLearner
from beetlebox.analysis import e4
from beetlebox.config import (
    E4RunConfig,
    QuusConfig,
    RuleLearnerConfig,
    config_hash,
    to_dict,
)
from beetlebox.envs.quus import QuusTask
from beetlebox.harness import E4RunManager
from beetlebox.runlog import RunLogger, run_dir
from beetlebox.seeding import seed_everything


def _cfg(encoding="scalar", num_seeds=4, steps=2500, seed=0):
    cfg = E4RunConfig(seed=seed)
    cfg.experiment.num_seeds = num_seeds
    cfg.quus = QuusConfig(max_operand=12, bend=8, modulus=24, quus_value=0, encoding=encoding)
    cfg.learner = RuleLearnerConfig(num_steps=steps)
    return cfg


# -- learner ---------------------------------------------------------------- #
def test_rule_learner_fits_training_data():
    seed_everything(0)
    task = QuusTask(QuusConfig(encoding="scalar"))
    x, y = task.features(task.train_pairs), task.plus(task.train_pairs).astype(float)
    m = RuleLearner(task.feature_dim, RuleLearnerConfig(num_steps=2000))
    m.fit(x, y)
    preds = m.predict(x)
    assert (preds == task.plus(task.train_pairs)).mean() > 0.9  # learns the seen data
    m.reset_parameters()


# -- harness + metrics ------------------------------------------------------ #
def _run(tmp_path, cfg):
    cfg.output_dir = str(tmp_path)
    directory = run_dir(cfg.output_dir, config_hash(cfg), cfg.seed)
    with RunLogger(directory) as logger:
        logger.write_manifest(to_dict(cfg), seed=cfg.seed, config_hash=config_hash(cfg))
        E4RunManager(cfg, logger=logger).run()
    return str(directory)


def test_scalar_encoding_resolves_toward_plus(tmp_path):
    result = e4.score_run(_run(tmp_path, _cfg("scalar")))
    assert result["checks"]["shared_prior_resolves"]["pass"] is True
    assert result["plus_rate"] >= 0.6
    assert "E4 report" in e4._format_report(result)


def test_onehot_encoding_leaves_underdetermination(tmp_path):
    result = e4.score_run(_run(tmp_path, _cfg("onehot")))
    assert result["checks"]["underdetermination_visible"]["pass"] is True
    assert result["plus_rate"] <= 0.2


def test_cross_seed_agreement_extremes():
    assert e4.cross_seed_agreement([[1, 2, 3], [1, 2, 3]]) == 1.0
    assert e4.cross_seed_agreement([[1, 2, 3], [4, 5, 6]]) == 0.0


def test_score_run_missing_predictions_raises(tmp_path):
    directory = tmp_path / "bad"
    with RunLogger(directory) as logger:
        logger.write_manifest({"quus": {"encoding": "scalar"}}, seed=0, config_hash="x")
        logger.log("run_end", encoding="scalar")
    with pytest.raises(ValueError):
        e4.score_run(str(directory))


def test_e4_main(tmp_path, monkeypatch, capsys):
    d = _run(tmp_path, _cfg("scalar", num_seeds=3, steps=1500))
    monkeypatch.setattr("sys.argv", ["e4", d])
    e4.main()
    assert "E4 report" in capsys.readouterr().out
