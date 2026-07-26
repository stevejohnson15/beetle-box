# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""Leak-free property of the rich (frontier) private_referent runner.

These tests inject fake agents, so they run without any API access. The point is
structural: the receiver's prompt never reveals the target -- only the sender's
symbol can -- so a receiver that ignores the message cannot beat chance.
"""

import re

from beetlebox.harness.rich_e3 import RichPrivateReferentRunner

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


def _runner(condition, sender, receiver, k=4, v=4, rounds=40, seed=0):
    return RichPrivateReferentRunner(
        condition, num_referents=k, vocab_size=v, num_rounds=rounds, seed=seed,
        sender=sender, receiver=receiver)


def test_receiver_candidates_are_target_independent_and_complete():
    r = _runner("shared", _Fake(lambda u, c: 0), _Fake(lambda u, c: 0))
    cands = r.receiver_candidates()
    assert len(cands) == r.k
    assert len(set(cands)) == r.k  # shared: every object has a distinct code
    # Called again it is identical -> it cannot encode a per-round target.
    assert r.receiver_candidates() == cands


def test_empty_condition_candidates_indistinguishable():
    r = _runner("empty", _Fake(lambda u, c: 0), _Fake(lambda u, c: 0))
    cands = r.receiver_candidates()
    assert len(set(cands)) == 1  # nothing to tell objects apart


def test_receiver_prompt_does_not_depend_on_the_target():
    # The prompt is built only from (history, symbol, candidates) -- not the target.
    r = _runner("shared", _Fake(lambda u, c: 0), _Fake(lambda u, c: 0))
    cands = r.receiver_candidates()
    p1 = r._receiver_user(history=[], symbol=2, candidates=cands)
    p2 = r._receiver_user(history=[], symbol=2, candidates=cands)
    assert p1 == p2
    assert "withheld" in r._receiver_user(history=[], symbol=None, candidates=cands)


def _smart_sender(code_list):
    # Sender encodes its private sensation as the symbol = that code's object index.
    def policy(user, choices):
        m = re.search(r"private sensation is: (\S+?)\.", user)
        token = m.group(1)
        return code_list.index(token) if token in code_list else 0
    return policy


def _symbol_reading_receiver(user, choices):
    # Competent receiver: decode the symbol (== target index under the test scheme).
    m = re.search(r"sent symbol (\d+)", user)
    return int(m.group(1)) if m else 0


def _message_ignoring_receiver(user, choices):
    return 0  # ignores the symbol entirely


def test_message_is_load_bearing_smart_vs_ignoring():
    # A receiver that reads the symbol succeeds; one that ignores it is stuck at
    # chance -- proving the public message, not a leaked box, carries the signal.
    sender = _Fake(_smart_sender(list(RichPrivateReferentRunner(
        "shared", num_referents=4, vocab_size=4, num_rounds=1,
        sender=_Fake(lambda u, c: 0), receiver=_Fake(lambda u, c: 0))._code_s)))
    smart = _runner("shared", sender, _Fake(_symbol_reading_receiver))
    assert smart.run()["final_accuracy"] > 0.9

    blind = _runner("shared", _Fake(_smart_sender(smart._code_s)),
                    _Fake(_message_ignoring_receiver))
    assert blind.run()["final_accuracy"] <= 0.5  # ~chance (1/4)


def test_drop_message_forces_chance():
    # With the symbol withheld, even a symbol-reading receiver has nothing to use.
    sender = _Fake(_smart_sender(list(_runner(
        "shared", _Fake(lambda u, c: 0), _Fake(lambda u, c: 0))._code_s)))
    r = _runner("shared", sender, _Fake(_symbol_reading_receiver))
    summary = r.run(drop_message=True)
    assert summary["drop_message"] is True
    assert summary["final_accuracy"] <= 0.5


def test_divergent_candidates_disjoint_from_sender_codes():
    r = _runner("divergent", _Fake(lambda u, c: 0), _Fake(lambda u, c: 0))
    cands = r.receiver_candidates()
    assert len(set(cands)) == r.k
    assert set(cands).isdisjoint(set(r._code_s))  # divergent -> different code block


def test_noise_candidates_are_not_stable():
    r = _runner("noise", _Fake(lambda u, c: 0), _Fake(lambda u, c: 0), rounds=1)
    # Fresh random draws each call -> no stable object->code map to exploit.
    first = r.receiver_candidates()
    second = r.receiver_candidates()
    assert first != second or len(set(first)) < r.k


def test_noise_run_stays_near_chance_with_symbol_reader():
    # Under noise there is no stable code, so even a symbol-reading receiver
    # cannot do better than chance -- the run still completes cleanly.
    sender = _Fake(lambda u, c: 0)
    r = _runner("noise", sender, _Fake(_symbol_reading_receiver), rounds=20)
    s = r.run()
    assert 0.0 <= s["final_accuracy"] <= 1.0
    assert s["api_calls"] == 2 * 20


def test_too_many_referents_raises():
    import pytest
    with pytest.raises(ValueError):
        _runner("shared", _Fake(lambda u, c: 0), _Fake(lambda u, c: 0), k=20)
