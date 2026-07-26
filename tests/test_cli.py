# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""Tests for the ``beetlebox`` command-line entrypoint."""

import pytest

from beetlebox import __version__, cli
from beetlebox.config import RunConfig, config_hash, to_dict
from beetlebox.harness import RunManager
from beetlebox.runlog import RunLogger, run_dir


def _make_run(tmp_path) -> str:
    cfg = RunConfig(seed=0, output_dir=str(tmp_path))
    cfg.experiment.num_steps = 200
    cfg.experiment.eval_every = 100
    directory = run_dir(cfg.output_dir, config_hash(cfg), cfg.seed)
    with RunLogger(directory) as logger:
        logger.write_manifest(to_dict(cfg), seed=cfg.seed, config_hash=config_hash(cfg))
        RunManager(cfg, logger=logger).run()
    return str(directory)


def test_analyze_prints_report(tmp_path, capsys):
    directory = _make_run(tmp_path)
    rc = cli.main(["analyze", directory])
    out = capsys.readouterr().out
    assert rc == 0
    assert "E1 report" in out
    assert "pre-registered checks" in out


def test_no_command_prints_help_and_returns_nonzero(capsys):
    rc = cli.main([])
    assert rc == 1
    assert "beetlebox" in capsys.readouterr().out.lower()


def test_version_flag_exits(capsys):
    # argparse's --version raises SystemExit after printing.
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out
