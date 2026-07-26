# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""E2 harness + analysis tests: the memory-manufactures-stability result, metrics,
determinism, and scoring."""

import pytest

from beetlebox.analysis import e2
from beetlebox.config import E2RunConfig, config_hash, to_dict
from beetlebox.harness import E2RunManager
from beetlebox.runlog import RunLogger, run_dir


def _cfg(memory="full", noise=0.25, steps=800, seed=0):
    cfg = E2RunConfig(seed=seed)
    cfg.experiment.num_steps = steps
    cfg.diarist.memory = memory
    cfg.percept.noise = noise
    return cfg


def _run(tmp_path, cfg):
    directory = run_dir(str(tmp_path), config_hash(cfg), cfg.seed)
    cfg.output_dir = str(tmp_path)
    directory = run_dir(cfg.output_dir, config_hash(cfg), cfg.seed)
    with RunLogger(directory) as logger:
        logger.write_manifest(to_dict(cfg), seed=cfg.seed, config_hash=config_hash(cfg))
        summary = E2RunManager(cfg, logger=logger).run()
    return str(directory), summary


# -- harness ---------------------------------------------------------------- #
def test_run_summary_has_sequence():
    summary = E2RunManager(_cfg(steps=200)).run()
    assert len(summary["types"]) == 200
    assert len(summary["terms"]) == 200
    assert len(summary["gaps"]) == 200


def test_run_is_deterministic():
    a = E2RunManager(_cfg(steps=300)).run()
    b = E2RunManager(_cfg(steps=300)).run()
    assert a["terms"] == b["terms"]


# -- metrics ---------------------------------------------------------------- #
def test_consistency_and_purity_extremes():
    types = [0, 0, 1, 1]
    assert e2.consistency(types, [7, 7, 9, 9]) == 1.0        # each type one term
    assert e2.consistency(types, [0, 1, 2, 3]) == 0.5        # all distinct
    assert e2.purity([0, 0, 1, 1], [5, 5, 5, 5]) == 0.5      # one term, two types


def test_drift_and_gap_conditioned():
    types = [0, 1, 0, 1, 0, 1]
    terms = [0, 1, 0, 1, 0, 1]
    d = e2.drift(types, terms, split=0.5)
    assert d["early"] == 1.0 and d["late"] == 1.0
    gaps = [-1, -1, 2, 2, 2, 2]
    g = e2.gap_conditioned_consistency(types, terms, gaps, gap_threshold=20)
    assert g["short_gap"] == 1.0


# -- scoring ---------------------------------------------------------------- #
def test_score_full_memory_passes_stability(tmp_path):
    d, _ = _run(tmp_path, _cfg(memory="full"))
    result = e2.score_run(d)
    assert result["checks"]["stability_with_memory"]["pass"] is True
    assert result["consistency"] >= 0.9
    assert "E2 report" in e2._format_report(result)


def test_score_no_memory_passes_chance(tmp_path):
    d, _ = _run(tmp_path, _cfg(memory="none"))
    result = e2.score_run(d)
    assert result["checks"]["chance_without_memory"]["pass"] is True
    assert result["consistency"] <= 0.1


def test_score_run_missing_sequence_raises(tmp_path):
    directory = tmp_path / "bad"
    with RunLogger(directory) as logger:
        logger.write_manifest({"diarist": {"policy": "prototype", "memory": "full"}},
                              seed=0, config_hash="x")
        logger.log("run_end", policy="prototype", memory="full")
    with pytest.raises(ValueError):
        e2.score_run(str(directory))


def test_e2_main(tmp_path, monkeypatch, capsys):
    d, _ = _run(tmp_path, _cfg(memory="full", steps=200))
    monkeypatch.setattr("sys.argv", ["e2", d])
    e2.main()
    assert "E2 report" in capsys.readouterr().out
