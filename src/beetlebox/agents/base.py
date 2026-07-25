# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""Agent abstractions.

The base :class:`Agent` is deliberately light and framework-free so that future
backends (frontier API models for rich mode, other local models) can implement
it. E1's concrete agents live in :mod:`beetlebox.agents.neural` and are PyTorch
modules that also satisfy this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from beetlebox.memory import Memory, NoMemory


class Agent(ABC):
    """A participant in an experiment.

    ``reset_parameters`` supports the E1 *turnover* manipulation: a fresh agent
    is introduced mid-run to test whether a convention survives its founders.
    """

    def __init__(self, name: str, memory: Memory | None = None) -> None:
        self.name = name
        self.memory: Memory = memory if memory is not None else NoMemory()

    @abstractmethod
    def reset_parameters(self) -> None:
        """Reinitialize learnable state (used for agent turnover)."""
