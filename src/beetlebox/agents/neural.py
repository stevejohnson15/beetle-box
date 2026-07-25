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
