# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""Shape and reinitialization tests for the from-scratch neural agents."""

import torch

from beetlebox.agents import (
    DiscriminationReceiver,
    MatchingAgent,
    NeuralReceiver,
    NeuralSender,
)


def test_neural_sender_shapes_and_reset():
    s = NeuralSender(feature_dim=8, vocab_size=6, message_length=2)
    feats = torch.randn(4, 8)
    msg, logprob, entropy = s.act(feats)
    assert msg.shape == (4, 2)
    assert logprob.shape == (4,) and entropy.shape == (4,)
    assert (msg >= 0).all() and (msg < 6).all()
    s.reset_parameters()  # exercised by turnover; must not error


def test_neural_receiver_shapes_and_reset():
    r = NeuralReceiver(vocab_size=6, message_length=2, num_classes=8)
    msg = torch.randint(0, 6, (4, 2))
    assert r(msg).shape == (4, 8)
    assert r.predict(msg).shape == (4,)
    r.reset_parameters()


def test_discrimination_receiver_shapes_and_reset():
    r = DiscriminationReceiver(vocab_size=6, message_length=2, box_dim=16)
    msg = torch.randint(0, 6, (4, 2))
    candidate_boxes = torch.randn(4, 8, 16)  # [B, K, box_dim]
    logits = r(msg, candidate_boxes)
    assert logits.shape == (4, 8)
    assert r.predict(msg, candidate_boxes).shape == (4,)
    r.reset_parameters()


def test_matching_agent_speak_and_judge_and_reset():
    a = MatchingAgent(vocab_size=6, message_length=2, box_dim=16)
    box = torch.randn(4, 16)
    msg, logprob, entropy = a.speak(box)
    assert msg.shape == (4, 2) and logprob.shape == (4,)
    assert a.judge(box, msg).shape == (4, 2)  # same/different logits
    a.reset_parameters()
