# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
from beetlebox.analysis import transcript
from beetlebox.config import RunConfig
from beetlebox.harness import RunManager


def test_convention_rows_flags_collisions():
    # Referents 0 and 1 both map to message [3] -> a collision.
    mapping = [[3], [3], [5]]
    guesses = [0, 0, 2]
    rows = transcript.convention_rows(mapping, guesses)
    assert rows[0]["collision_with"] is None
    assert rows[1]["collision_with"] == 0
    assert rows[2]["collision_with"] is None
    assert rows[0]["correct"] is True
    assert rows[1]["correct"] is False  # guessed 0, is 1


def test_format_convention_reports_count():
    text = transcript.format_convention([[0], [1]], [0, 1], step=800, accuracy=1.0)
    assert "2/2 referents correctly communicated" in text
    assert "step 800" in text


def test_sample_exchanges_shape():
    cfg = RunConfig(seed=0)
    cfg.experiment.num_steps = 0
    mgr = RunManager(cfg)
    ex = mgr.sample_exchanges(6)
    assert len(ex) == 6
    for e in ex:
        assert set(e) == {"referent", "message", "guess", "correct"}
        assert e["correct"] == (e["referent"] == e["guess"])


def test_evaluate_includes_guesses():
    cfg = RunConfig(seed=0)
    cfg.experiment.num_steps = 0
    ev = RunManager(cfg).evaluate()
    assert "guesses" in ev and "mapping" in ev
    assert len(ev["guesses"]) == cfg.env.num_referents


def test_render_run_from_events(tmp_path):
    # A short real run logs guesses; the transcript reconstructs from events alone.
    cfg = RunConfig(seed=0, output_dir=str(tmp_path))
    cfg.experiment.num_steps = 200
    cfg.experiment.eval_every = 100
    from beetlebox.config import config_hash, to_dict
    from beetlebox.runlog import RunLogger, run_dir
    directory = run_dir(cfg.output_dir, config_hash(cfg), cfg.seed)
    with RunLogger(directory) as logger:
        logger.write_manifest(to_dict(cfg), seed=cfg.seed, config_hash=config_hash(cfg))
        RunManager(cfg, logger=logger).run()
    text = transcript.render_run(str(directory))
    assert "convention snapshot" in text
    assert "referents correctly communicated" in text


def test_format_exchanges_renders_trials():
    exchanges = [
        {"referent": 3, "message": [6], "guess": 3, "correct": True},
        {"referent": 1, "message": [2], "guess": 5, "correct": False},
    ]
    text = transcript.format_exchanges(exchanges)
    assert "trial" in text
    assert "referent 3" in text and "s6" in text


def test_render_run_filters_steps(tmp_path):
    from beetlebox.config import RunConfig, config_hash, to_dict
    from beetlebox.runlog import RunLogger, run_dir
    cfg = RunConfig(seed=0, output_dir=str(tmp_path))
    cfg.experiment.num_steps = 300
    cfg.experiment.eval_every = 100
    directory = run_dir(cfg.output_dir, config_hash(cfg), cfg.seed)
    with RunLogger(directory) as logger:
        logger.write_manifest(to_dict(cfg), seed=cfg.seed, config_hash=config_hash(cfg))
        RunManager(cfg, logger=logger).run()
    text = transcript.render_run(str(directory), steps=[300])
    assert "step 300" in text
    assert "step 100" not in text


def test_render_run_without_transcript_data(tmp_path):
    from beetlebox.runlog import RunLogger
    directory = tmp_path / "empty_run"
    with RunLogger(directory) as logger:
        logger.write_manifest({"channel": {"vocab_size": 8, "message_length": 1}},
                              seed=0, config_hash="x")
        logger.log("eval", step=100, accuracy=0.5, mapping=[[0]])  # no 'guesses'
    text = transcript.render_run(str(directory))
    assert "No transcript data" in text


def test_transcript_main(tmp_path, monkeypatch, capsys):
    from beetlebox.analysis import transcript as tmod
    from beetlebox.config import RunConfig, config_hash, to_dict
    from beetlebox.runlog import RunLogger, run_dir
    cfg = RunConfig(seed=0, output_dir=str(tmp_path))
    cfg.experiment.num_steps = 200
    cfg.experiment.eval_every = 100
    d = run_dir(cfg.output_dir, config_hash(cfg), cfg.seed)
    with RunLogger(d) as logger:
        logger.write_manifest(to_dict(cfg), seed=cfg.seed, config_hash=config_hash(cfg))
        RunManager(cfg, logger=logger).run()
    monkeypatch.setattr("sys.argv", ["transcript", str(d)])
    tmod.main()
    assert "convention snapshot" in capsys.readouterr().out
