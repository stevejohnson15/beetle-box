# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""Agent abstractions -- re-exported from the `agentharness` library.

The framework-free ``Agent`` base was extracted into the standalone, Apache-2.0
``agentharness`` package (https://github.com/stevejohnson15/agentharness); it is
re-exported here so existing ``beetlebox.agents.base`` imports keep working. E1's
concrete PyTorch agents live in :mod:`beetlebox.agents.neural` and satisfy this
interface.
"""

from __future__ import annotations

from agentharness import Agent, ChoiceAgent

__all__ = ["Agent", "ChoiceAgent"]
