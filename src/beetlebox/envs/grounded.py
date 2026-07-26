# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""The grounded 'form of life' for E5 (*Philosophical Investigations* on forms of
life; capstone).

E1's signaling game is *ungrounded*: a symbol earns reward only by letting the
receiver name the referent — it connects to nothing beyond the identification game.
E5 adds a **grounded** regime in which words drive **real consequences**. Each
referent is an object in a small resource/survival world: the receiver must choose
an **action**, and the payoff depends on the referent's properties, with per-referent
**stakes** (getting some objects right matters far more than others).

This module supplies that world on top of a grid :class:`~beetlebox.envs.SignalingEnv`:

- the **correct action** for a referent is the value of its first ("relevant")
  attribute — so the message must convey that distinction to act well;
- the **stakes** grow with the second ("irrelevant-to-the-action") attribute — so a
  mistake on a high-stakes object is far costlier, giving the language a functional
  pressure the ungrounded game lacks;
- ``payoff[referent, action] = +stake`` if the action is correct, ``-stake`` otherwise.

The point of E5 is to compare the *character* of the language that emerges under
this grounding versus under bare identification. NumPy only.
"""

from __future__ import annotations

import numpy as np

from beetlebox.envs import SignalingEnv


class GroundedWorld:
    """Per-referent action payoffs with stakes, derived from grid attributes."""

    def __init__(self, env: SignalingEnv) -> None:
        if env.mode != "grid":
            raise ValueError("GroundedWorld requires a grid SignalingEnv (needs attributes)")
        self.env = env
        self.num_actions = env.num_values  # one action per value of the relevant attribute
        attrs = np.stack([env.attributes_for(r) for r in range(env.num_classes)])  # [K, A]
        self._correct = attrs[:, 0].astype(np.int64)  # relevant attribute -> correct action
        # Stakes grow with the second attribute (uniform +1 baseline) so some objects
        # matter much more than others; normalized so mean stake is ~1.
        if attrs.shape[1] > 1:
            raw = 1.0 + attrs[:, 1].astype(np.float32)
        else:
            raw = np.ones(env.num_classes, dtype=np.float32)
        self._stake = (raw / raw.mean()).astype(np.float32)
        self._payoff = self._build_payoff()

    def _build_payoff(self) -> np.ndarray:
        """``[K, num_actions]``: +stake for the correct action, -stake otherwise."""
        k = self.env.num_classes
        payoff = -self._stake[:, None] * np.ones((k, self.num_actions), dtype=np.float32)
        payoff[np.arange(k), self._correct] = self._stake
        return payoff

    @property
    def payoff(self) -> np.ndarray:
        """The full ``[K, num_actions]`` payoff matrix (God's-eye)."""
        return self._payoff

    @property
    def correct_action(self) -> np.ndarray:
        """The payoff-maximizing action per referent, shape ``[K]``."""
        return self._correct

    @property
    def stake(self) -> np.ndarray:
        """Per-referent stake (payoff magnitude), shape ``[K]``."""
        return self._stake

    @property
    def max_mean_payoff(self) -> float:
        """Best achievable mean payoff (always act correctly) = mean stake."""
        return float(self._stake.mean())

    def payoff_for(self, referents: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Realized payoff for chosen ``actions`` on the true ``referents``."""
        return self._payoff[referents, actions]
