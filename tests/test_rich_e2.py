# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""Rich (frontier) diarist tests — run with an injected fake agent (no network)."""

import pytest

from beetlebox.harness.rich_e2 import RichDiaristRunner

_ZERO_USAGE = {"input_tokens": 0, "output_tokens": 0,
               "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}


class _Fake:
    def __init__(self, policy):
        self._policy = policy
        self.usage = dict(_ZERO_USAGE)
        self.num_calls = 0
        self.last_user = None

    def choose(self, system, user, choices):
        self.num_calls += 1
        self.last_user = user
        return self._policy(user, list(choices))


def test_always_new_coins_every_step():
    # A diarist that always picks the last option (coin new) -> all distinct terms.
    r = RichDiaristRunner(memory="none", num_types=3, dim=3, num_steps=10, seed=0,
                          agent=_Fake(lambda u, c: c[-1]))
    s = r.run()
    assert s["distinct_terms"] == 10
    assert s["api_calls"] == 10


def test_always_reuse_first_gives_one_term():
    # Always reuse option 0 when it exists (else coin) -> a single term after step 1.
    def policy(user, choices):
        return 0
    r = RichDiaristRunner(memory="full", num_types=3, dim=3, num_steps=8, seed=0,
                          agent=_Fake(policy))
    s = r.run()
    assert s["distinct_terms"] == 1


def test_none_memory_prompt_hides_diary():
    fake = _Fake(lambda u, c: c[-1])
    RichDiaristRunner(memory="none", num_types=3, dim=3, num_steps=3, seed=0,
                      agent=fake).run()
    assert "no diary" in fake.last_user.lower() or "no past entries" in fake.last_user.lower()


def test_full_memory_prompt_shows_diary():
    # Reuse-first so a diary entry exists; later prompts should show past readings.
    fake = _Fake(lambda u, c: 0)
    RichDiaristRunner(memory="full", num_types=3, dim=3, num_steps=4, seed=0,
                      agent=fake).run()
    assert "reading" in fake.last_user.lower()


def test_invalid_memory_mode_raises():
    with pytest.raises(ValueError):
        RichDiaristRunner(memory="bogus", agent=_Fake(lambda u, c: 0))
