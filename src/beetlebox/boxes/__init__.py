# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""The beetle-box mechanism (Philosophical Investigations §293).

Every agent has a private "box"; on each trial its box yields a private inner
signal that only that agent sees. We (the experimenters) control the boxes,
which is the whole point of §293: the agents cannot look into each other's boxes,
but we can, so "whether two agents' beetles are the same" has a God's-eye answer
the agents lack.

Four conditions control what the boxes contain, relative to a God's-eye state
``r`` (e.g. a referent index):

- ``shared``   : every agent's box encodes ``r`` with the SAME code -> boxes are
                 literally comparable across agents (a shared inner state).
- ``divergent``: each agent encodes ``r`` with its OWN fixed private code -> the
                 information is there but idiosyncratic and incomparable (the
                 literal beetle-box: same game, private forms differ).
- ``empty``    : the box is constant (carries nothing about ``r``).
- ``noise``    : the box is random and independent of ``r``.

**Earnable cancellation (§3.3).** The box signal is delivered as a real input to
the agents' networks, so it *can* propagate to behavior. Whether it actually
does — and whether the private *form* (shared vs. divergent) matters — is then a
measured result, not an assumption. Informative boxes (shared/divergent) must be
able to beat empty ones, or the wiring would be rigged.

NumPy only; an extraction candidate for a reusable emergent-communication kit.
"""

from __future__ import annotations

from enum import StrEnum

import numpy as np


class BoxCondition(StrEnum):
    SHARED = "shared"
    DIVERGENT = "divergent"
    EMPTY = "empty"
    NOISE = "noise"


class BoxScheme:
    """Produces per-agent private box signals for a run.

    Code tables are fixed for the run (built once from the run seed), so a given
    ``(agent, r)`` pair yields a stable signal within a run — except in the
    ``noise`` condition, where a fresh random vector is drawn each call (the box
    carries no stable information about ``r``).
    """

    def __init__(self, condition: str | BoxCondition, num_agents: int, num_states: int,
                 box_dim: int, rng: np.random.Generator) -> None:
        self.condition = BoxCondition(condition)
        self.num_agents = num_agents
        self.num_states = num_states
        self.box_dim = box_dim
        self._rng = rng

        if self.condition == BoxCondition.SHARED:
            # One code table, shared by all agents: box_agent(r) is identical.
            shared = rng.standard_normal((num_states, box_dim)).astype(np.float32)
            self._tables = [shared for _ in range(num_agents)]
        elif self.condition == BoxCondition.DIVERGENT:
            # Per-agent code tables: same r, different private encodings.
            self._tables = [
                rng.standard_normal((num_states, box_dim)).astype(np.float32)
                for _ in range(num_agents)
            ]
        else:  # EMPTY / NOISE carry no information about r; no tables needed.
            self._tables = None

    def signal(self, agent_idx: int, states: np.ndarray) -> np.ndarray:
        """Return box signals of shape ``[len(states), box_dim]`` for one agent.

        ``states`` is an array of God's-eye state indices (e.g. referent ids).
        """
        states = np.asarray(states)
        n = states.shape[0]
        if self.condition == BoxCondition.EMPTY:
            return np.zeros((n, self.box_dim), dtype=np.float32)
        if self.condition == BoxCondition.NOISE:
            return self._rng.standard_normal((n, self.box_dim)).astype(np.float32)
        return self._tables[agent_idx][states]

    @property
    def is_informative(self) -> bool:
        """True when the box carries information about the God's-eye state."""
        return self.condition in (BoxCondition.SHARED, BoxCondition.DIVERGENT)
