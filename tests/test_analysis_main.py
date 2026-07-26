# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""Exercise the analysis module CLI entrypoints (main())."""

from beetlebox.config import RunConfig, config_hash, to_dict
from beetlebox.harness import RunManager
from beetlebox.runlog import RunLogger, run_dir


def _e1_run(tmp_path):
    cfg = RunConfig(seed=0, output_dir=str(tmp_path))
    cfg.experiment.num_steps = 200
    cfg.experiment.eval_every = 100
    d = run_dir(cfg.output_dir, config_hash(cfg), cfg.seed)
    with RunLogger(d) as logger:
        logger.write_manifest(to_dict(cfg), seed=cfg.seed, config_hash=config_hash(cfg))
        RunManager(cfg, logger=logger).run()
    return str(d)


def test_e1_main(tmp_path, monkeypatch, capsys):
    from beetlebox.analysis import e1
    monkeypatch.setattr("sys.argv", ["e1", _e1_run(tmp_path)])
    e1.main()
    assert "E1 report" in capsys.readouterr().out


def test_e3_main(tmp_path, monkeypatch, capsys):
    from beetlebox.analysis import e3
    from beetlebox.runlog import RunLogger
    dirs = []
    for cond, acc in [("shared", 1.0), ("divergent", 0.9), ("empty", 0.12), ("noise", 0.13)]:
        d = tmp_path / cond
        with RunLogger(d) as logger:
            logger.write_manifest({"box": {"condition": cond},
                                   "experiment": {"game": "private_referent"}},
                                  seed=0, config_hash="x")
            logger.log("run_end", game="private_referent", condition=cond,
                       final_accuracy=acc, chance=0.125)
        dirs.append(str(d))
    monkeypatch.setattr("sys.argv", ["e3", *dirs])
    e3.main()
    assert "E3 report" in capsys.readouterr().out
