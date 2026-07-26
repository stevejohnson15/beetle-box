# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""Rich (frontier) rule-learner tests — injected fake agent (no network)."""

import re

from beetlebox.harness.rich_e4 import RichRuleLearnerRunner

_ZERO_USAGE = {"input_tokens": 0, "output_tokens": 0,
               "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}


class _Fake:
    def __init__(self, policy):
        self._policy = policy
        self.usage = dict(_ZERO_USAGE)
        self.num_calls = 0

    def choose(self, system, user, choices):
        self.num_calls += 1
        return self._policy(user, list(choices))


def _plus_answer(user, choices):
    # A "plus" learner: parse "a # b = ?" and return a + b.
    a, b = map(int, re.findall(r"(\d+) # (\d+) = \?", user)[0])
    return a + b


def _quus_answer(user, choices):
    return 0  # always the quus constant


def test_plus_learner_scores_plus():
    r = RichRuleLearnerRunner(num_queries=8, seed=0, agent=_Fake(_plus_answer))
    s = r.run()
    assert s["plus_rate"] == 1.0
    assert s["api_calls"] == 8


def test_quus_learner_scores_quus():
    r = RichRuleLearnerRunner(num_queries=6, seed=0, quus_value=0, agent=_Fake(_quus_answer))
    s = r.run()
    assert s["quus_rate"] == 1.0
    assert s["plus_rate"] == 0.0


def test_examples_block_is_below_bend():
    # The examples shown to the model are all below the bend (plus == quus there).
    r = RichRuleLearnerRunner(max_operand=12, bend=8, seed=0, agent=_Fake(_plus_answer))
    block = r._examples_block()
    assert "8 #" not in block and "# 8" not in block  # nothing at/above the bend
