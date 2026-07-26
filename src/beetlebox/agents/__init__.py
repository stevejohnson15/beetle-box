# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""Agent backends for Beetle-Box.

``Agent`` is the framework-free base. The rest are from-scratch (clean-room)
PyTorch agents: ``NeuralSender``/``NeuralReceiver`` play E1 (referential signaling);
``DiscriminationReceiver`` and ``MatchingAgent`` add E3's beetle-box games. The
frontier (rich-mode) backend lives separately in :mod:`beetlebox.agents.api_model`,
imported on demand so this module works without the ``anthropic`` SDK.

Importing the neural agents pulls in PyTorch; import :mod:`beetlebox.agents.base`
directly to avoid that.
"""

from beetlebox.agents.base import Agent
from beetlebox.agents.neural import (
    DiscriminationReceiver,
    MatchingAgent,
    NeuralReceiver,
    NeuralSender,
)

__all__ = [
    "Agent",
    "NeuralSender",
    "NeuralReceiver",
    "DiscriminationReceiver",
    "MatchingAgent",
]
