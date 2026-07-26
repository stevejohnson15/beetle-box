# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""E6 (reflexive layer) tests -- injected fake agents, no network.

The design is what matters: the metric is the behavioral pre->post delta contrasted
against a matched control, and the reflection transcript is never scored. These
tests verify that plumbing deterministically.
"""

import re

import pytest

from beetlebox.analysis import e6
from beetlebox.config import E6Config, config_hash, to_dict
from beetlebox.harness.rich_e6 import RichReflexiveRunner
from beetlebox.runlog import RunLogger, run_dir

_ZERO = {"input_tokens": 0, "output_tokens": 0,
         "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}


class _Fake:
    def __init__(self, chooser, reply="a calm lake at dawn"):
        self._chooser = chooser
        self._reply = reply
        self.usage = dict(_ZERO)
        self.num_calls = 0
        self.respond_calls = 0

    def choose(self, system, user, choices):
        self.num_calls += 1
        return self._chooser(user, list(choices))

    def respond(self, system, user):
        self.num_calls += 1
        self.respond_calls += 1
        return self._reply


def _perfect_sender(user, choices):
    r = int(re.search(r"hidden object is (\d+)", user).group(1))
    return r % len(choices)


def _perfect_receiver(user, choices):
    return int(re.search(r"sent symbol (\d+)", user).group(1))  # symbol == object here


def _run(intervention, sender, receiver, rounds=6, seed=0):
    cfg = E6Config(intervention=intervention, num_referents=4, vocab_size=6,
                   rounds_per_block=rounds, seed=seed)
    return RichReflexiveRunner(cfg, sender=sender, receiver=receiver).run()


def test_reflect_and_control_add_one_interlude_turn_each():
    snd, rcv = _Fake(_perfect_sender), _Fake(_perfect_receiver)
    _run("reflect", snd, rcv)
    assert snd.respond_calls == 1 and rcv.respond_calls == 1  # one intervention turn each


def test_none_intervention_has_no_interlude_and_no_transcript():
    snd, rcv = _Fake(_perfect_sender), _Fake(_perfect_receiver)
    s = _run("none", snd, rcv)
    assert snd.respond_calls == 0
    assert s["reflection_transcript"] == {}


def test_delta_is_post_minus_pre():
    s = _run("reflect", _Fake(_perfect_sender), _Fake(_perfect_receiver))
    assert s["pre_accuracy"] == 1.0 and s["post_accuracy"] == 1.0
    assert s["delta"] == pytest.approx(0.0)


def test_bad_intervention_raises():
    with pytest.raises(ValueError):
        RichReflexiveRunner(E6Config(intervention="bogus"),
                            sender=_Fake(_perfect_sender), receiver=_Fake(_perfect_receiver))


def test_transcript_recorded_but_separate():
    snd, rcv = _Fake(_perfect_sender, reply="REFLECTION TEXT"), _Fake(_perfect_receiver, reply="R")
    s = _run("reflect", snd, rcv)
    assert s["reflection_transcript"]["sender"] == "REFLECTION TEXT"


# -- analysis --------------------------------------------------------------- #
def _write_run(tmp_path, intervention, pre, post):
    d = tmp_path / intervention
    cfg = E6Config(intervention=intervention, seed=0)
    directory = run_dir(str(d), config_hash(cfg), cfg.seed)
    with RunLogger(directory) as logger:
        logger.write_manifest(to_dict(cfg), seed=cfg.seed, config_hash=config_hash(cfg))
        logger.log("run_end", intervention=intervention, pre_accuracy=pre,
                   post_accuracy=post, delta=post - pre, chance=0.25)
    return str(directory)


def test_compare_story_only_when_effect_small(tmp_path):
    reflect = _write_run(tmp_path, "reflect", 0.6, 0.65)   # delta +0.05
    control = _write_run(tmp_path, "control", 0.6, 0.62)   # delta +0.02
    result = e6.compare(reflect, control)
    assert result["verdict"] == "reflection_adds_a_story_only"
    assert "E6 report" in e6._format_comparison(result)  # sanity: renders


def test_compare_loadbearing_when_effect_large(tmp_path):
    reflect = _write_run(tmp_path, "reflect", 0.5, 0.95)   # delta +0.45
    control = _write_run(tmp_path, "control", 0.5, 0.55)   # delta +0.05
    result = e6.compare(reflect, control)
    assert result["verdict"] == "reflection_changes_coordination"
    assert result["reflection_effect"] == pytest.approx(0.40)


def test_compare_aggregates_multiple_seeds(tmp_path):
    # Multiple same-condition runs are averaged so one noisy run can't drive it.
    reflect = [_write_run(tmp_path / "r0", "reflect", 0.5, 0.9),   # +0.4
               _write_run(tmp_path / "r1", "reflect", 0.5, 0.5)]   # +0.0  -> mean +0.2
    control = [_write_run(tmp_path / "c0", "control", 0.5, 0.55),  # +0.05
               _write_run(tmp_path / "c1", "control", 0.5, 0.55)]  # +0.05 -> mean +0.05
    result = e6.compare(reflect, control)
    assert result["reflect"]["n"] == 2
    assert result["reflection_effect"] == pytest.approx(0.15)


def test_score_run_and_main(tmp_path, monkeypatch, capsys):
    d = _write_run(tmp_path, "reflect", 0.5, 0.7)
    assert e6.score_run(d)["delta"] == pytest.approx(0.2)
    monkeypatch.setattr("sys.argv", ["e6", d])
    e6.main()
    assert "E6 run" in capsys.readouterr().out
