# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""Agent backends for Beetle-Box.

``Agent`` is the framework-free base; ``NeuralSender``/``NeuralReceiver`` are the
from-scratch clean-room agents used by E1. Importing the neural agents pulls in
PyTorch; import :mod:`beetlebox.agents.base` directly to avoid that.
"""

from beetlebox.agents.base import Agent
from beetlebox.agents.neural import (
    BoxReceiver,
    DiscriminationReceiver,
    MatchingAgent,
    NeuralReceiver,
    NeuralSender,
)

__all__ = [
    "Agent",
    "NeuralSender",
    "NeuralReceiver",
    "BoxReceiver",
    "DiscriminationReceiver",
    "MatchingAgent",
]
