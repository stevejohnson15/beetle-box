# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""From-scratch neural agents for the E1 signaling game.

Both agents are small PyTorch modules initialized from scratch -- they arrive
with *no* inherited natural language, which is what makes E1 a clean-room study
of convention emerging from use alone. The sender is trained by REINFORCE (its
message is a discrete sample); the receiver by cross-entropy reconstruction of
the referent. Training itself lives in the harness, keeping agents to policy and
initialization.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

from beetlebox.agents.base import Agent
from beetlebox.memory import Memory


class NeuralSender(nn.Module, Agent):
    """Referent -> message policy. Emits ``L`` symbols from a vocab of size ``V``."""

    def __init__(self, feature_dim: int, vocab_size: int, message_length: int,
                 embed_dim: int = 32, hidden_dim: int = 64, *, name: str = "sender",
                 memory: Memory | None = None) -> None:
        nn.Module.__init__(self)
        Agent.__init__(self, name=name, memory=memory)
        self.vocab_size = vocab_size
        self.message_length = message_length
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, message_length * vocab_size),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return per-position symbol logits, shape ``[B, L, V]``."""
        logits = self.net(features)
        return logits.view(-1, self.message_length, self.vocab_size)

    def act(self, features: torch.Tensor, *, greedy: bool = False):
        """Produce a message and (for training) its log-prob and entropy.

        Returns ``(message[B, L], logprob[B], entropy[B])``. In ``greedy`` mode
        the argmax message is returned (used for deterministic evaluation).
        """
        logits = self.forward(features)
        dist = Categorical(logits=logits)  # batched over [B, L]
        if greedy:
            message = logits.argmax(dim=-1)
        else:
            message = dist.sample()
        logprob = dist.log_prob(message).sum(dim=-1)  # sum over positions
        entropy = dist.entropy().sum(dim=-1)
        return message, logprob, entropy

    def reset_parameters(self) -> None:
        for module in self.net:
            if isinstance(module, nn.Linear):
                module.reset_parameters()


class NeuralReceiver(nn.Module, Agent):
    """Message -> referent classifier over the ``K`` referents (reconstruction)."""

    def __init__(self, vocab_size: int, message_length: int, num_classes: int,
                 embed_dim: int = 32, hidden_dim: int = 64, *, name: str = "receiver",
                 memory: Memory | None = None) -> None:
        nn.Module.__init__(self)
        Agent.__init__(self, name=name, memory=memory)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(message_length * embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, message: torch.Tensor) -> torch.Tensor:
        """Return referent-class logits, shape ``[B, K]``."""
        emb = self.embedding(message)  # [B, L, embed_dim]
        flat = emb.reshape(emb.shape[0], -1)
        return self.net(flat)

    def predict(self, message: torch.Tensor) -> torch.Tensor:
        return self.forward(message).argmax(dim=-1)

    def reset_parameters(self) -> None:
        self.embedding.reset_parameters()
        for module in self.net:
            if isinstance(module, nn.Linear):
                module.reset_parameters()


class BoxReceiver(nn.Module, Agent):
    """Message + private box -> referent classifier (E3 private-referent game).

    The receiver's own box is a real input, so a *shared* inner state can help
    it decode where a *divergent* one cannot -- that difference is the measured
    beetle-box result.
    """

    def __init__(self, vocab_size: int, message_length: int, num_classes: int,
                 box_dim: int, embed_dim: int = 32, hidden_dim: int = 64, *,
                 name: str = "box_receiver", memory: Memory | None = None) -> None:
        nn.Module.__init__(self)
        Agent.__init__(self, name=name, memory=memory)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(message_length * embed_dim + box_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, message: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(message).reshape(message.shape[0], -1)
        return self.net(torch.cat([emb, box], dim=-1))

    def predict(self, message: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
        return self.forward(message, box).argmax(dim=-1)

    def reset_parameters(self) -> None:
        self.embedding.reset_parameters()
        for module in self.net:
            if isinstance(module, nn.Linear):
                module.reset_parameters()


class DiscriminationReceiver(nn.Module, Agent):
    """Message + candidate boxes -> which candidate is the target (E3 private-referent).

    The receiver is shown *all* candidates symmetrically -- each represented only by
    the receiver's private box for it -- and must use the sender's public message to
    pick the target. It scores candidate ``c`` by matching a query built from the
    message against a key built from ``box_R(c)``. Crucially there is no single
    "this is the target" box: with the message removed the candidates are
    indistinguishable, so the public channel is *forced* to be load-bearing (no
    leak -- unlike a plain classifier handed the target's own box).
    """

    def __init__(self, vocab_size: int, message_length: int, box_dim: int,
                 embed_dim: int = 32, hidden_dim: int = 64, *,
                 name: str = "discrimination_receiver", memory: Memory | None = None) -> None:
        nn.Module.__init__(self)
        Agent.__init__(self, name=name, memory=memory)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.msg_mlp = nn.Sequential(
            nn.Linear(message_length * embed_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.box_mlp = nn.Linear(box_dim, hidden_dim)

    def forward(self, message: torch.Tensor, candidate_boxes: torch.Tensor) -> torch.Tensor:
        """message ``[B, L]``, candidate_boxes ``[B, K, box_dim]`` -> logits ``[B, K]``."""
        q = self.msg_mlp(self.embedding(message).reshape(message.shape[0], -1))  # [B, H]
        k = self.box_mlp(candidate_boxes)  # [B, K, H]
        return torch.einsum("bh,bkh->bk", q, k)  # score each candidate against the message

    def predict(self, message: torch.Tensor, candidate_boxes: torch.Tensor) -> torch.Tensor:
        return self.forward(message, candidate_boxes).argmax(dim=-1)

    def reset_parameters(self) -> None:
        self.embedding.reset_parameters()
        self.box_mlp.reset_parameters()
        for module in self.msg_mlp:
            if isinstance(module, nn.Linear):
                module.reset_parameters()


class MatchingAgent(nn.Module, Agent):
    """Symmetric agent for the sensation same/different game (E3).

    A ``speak`` head turns the agent's private sensation (box) into a public
    symbol; a ``judge`` head decides, from the agent's own box plus the partner's
    public symbol, whether the two sensations are the same TYPE.
    """

    def __init__(self, vocab_size: int, message_length: int, box_dim: int,
                 embed_dim: int = 32, hidden_dim: int = 64, *, name: str = "matcher",
                 memory: Memory | None = None) -> None:
        nn.Module.__init__(self)
        Agent.__init__(self, name=name, memory=memory)
        self.vocab_size = vocab_size
        self.message_length = message_length
        self.speaker = nn.Sequential(
            nn.Linear(box_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, message_length * vocab_size),
        )
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.judge_net = nn.Sequential(
            nn.Linear(box_dim + message_length * embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # same / different
        )

    def speak(self, box: torch.Tensor, *, greedy: bool = False):
        logits = self.speaker(box).view(-1, self.message_length, self.vocab_size)
        dist = Categorical(logits=logits)
        message = logits.argmax(dim=-1) if greedy else dist.sample()
        logprob = dist.log_prob(message).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return message, logprob, entropy

    def judge(self, box: torch.Tensor, partner_message: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(partner_message).reshape(partner_message.shape[0], -1)
        return self.judge_net(torch.cat([box, emb], dim=-1))

    def reset_parameters(self) -> None:
        for net in (self.speaker, self.judge_net):
            for module in net:
                if isinstance(module, nn.Linear):
                    module.reset_parameters()
        self.embedding.reset_parameters()
