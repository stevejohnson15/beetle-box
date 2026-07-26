# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""Diarist naming policies for E2 (*Philosophical Investigations* §258).

The diarist sees a private percept each step and must emit a **term** for it —
naming a recurring sensation — with **no external correction**. Because §258 has
no training signal (there is nothing to be right or wrong against), these are not
trained networks: they are explicit naming *policies* whose behavior we can read
off directly.

Three policies span the philosophical possibilities:

- :class:`PrototypeDiarist` — the diary model. It matches each percept against its
  stored past impressions (its "diary") and reuses the term of the nearest one
  within a threshold, else coins a new term. Its memory is E2's central knob (via
  :func:`beetlebox.memory.make_memory`): ``none`` (no diary — every sensation is
  named afresh), ``windowed`` (only the last W impressions — "same again" fades
  with time gaps), or ``full``. The dilemma of §258 lives here: the diary can
  *manufacture* stability, but it checks impressions against impressions — is that
  an independent check or a longer private impression?
- :class:`FixedQuantizerDiarist` — a deterministic percept->term rule with no diary.
  It stands in for a hypothetical *independent criterion*: perfectly consistent by
  construction, the ceiling a real external check would buy.
- :class:`NoisyImpressionDiarist` — names by an in-the-moment stochastic impression
  with no persistence, modelling "whatever is going to seem right to me is right"
  (§258): the floor.

NumPy only.
"""

from __future__ import annotations

from abc import abstractmethod

import numpy as np

from beetlebox.agents.base import Agent
from beetlebox.config import DiaristConfig
from beetlebox.memory import make_memory


class Diarist(Agent):
    """Base class: maps a percept to a term id, under no external correction."""

    @abstractmethod
    def assign_term(self, percept: np.ndarray) -> int:
        """Return the term id the diarist assigns to ``percept`` this step."""

    def reset_parameters(self) -> None:
        """Diarists carry no trainable parameters; state is cleared per run."""
        self.memory.clear()


class PrototypeDiarist(Diarist):
    """Name by nearest past impression in the diary (memory-toggle driven)."""

    def __init__(self, *, memory_mode: str = "full", window: int = 20,
                 threshold: float = 0.6, name: str = "prototype_diarist") -> None:
        enabled = memory_mode != "none"
        capacity = window if memory_mode == "windowed" else None
        super().__init__(name=name, memory=make_memory(enabled, capacity=capacity))
        self.memory_mode = memory_mode
        self.threshold = threshold
        self._next_term = 0

    def assign_term(self, percept: np.ndarray) -> int:
        """Reuse the term of the nearest diary impression within threshold, else coin."""
        entries = self.memory.read()  # past impressions (empty under NoMemory)
        term: int
        if entries:
            protos = np.stack([e["percept"] for e in entries])
            dists = np.linalg.norm(protos - percept[None, :], axis=1)
            nearest = int(np.argmin(dists))
            if dists[nearest] <= self.threshold:
                term = int(entries[nearest]["term"])  # "same again"
            else:
                term = self._coin()
        else:
            term = self._coin()  # no diary to check against -> name afresh
        # Record this impression so later steps can match it.
        self.memory.write({"percept": np.asarray(percept, dtype=np.float32), "term": term})
        return term

    def _coin(self) -> int:
        term = self._next_term
        self._next_term += 1
        return term

    def reset_parameters(self) -> None:
        """Clear the diary and the coined-term counter."""
        super().reset_parameters()
        self._next_term = 0


class FixedQuantizerDiarist(Diarist):
    """Deterministic percept->term rule (an independent-criterion ceiling)."""

    def __init__(self, *, num_bins: int = 3, scale: float = 1.5,
                 name: str = "fixed_quantizer_diarist") -> None:
        super().__init__(name=name)  # no memory needed: the rule is stateless
        self.num_bins = num_bins
        self.scale = scale

    def assign_term(self, percept: np.ndarray) -> int:
        """Map the percept to a fixed quantization-cell id (stateless, deterministic)."""
        # Quantize each dimension into `num_bins` bins over [-scale, scale], then
        # fold the per-dim bins into one integer cell id -- a fixed function of the
        # percept, identical every time (no diary of past instances).
        edges = np.clip(percept, -self.scale, self.scale)
        bins = np.floor((edges + self.scale) / (2 * self.scale) * self.num_bins)
        bins = np.clip(bins, 0, self.num_bins - 1).astype(np.int64)
        term = 0
        for b in bins:
            term = term * self.num_bins + int(b)
        return term


class NoisyImpressionDiarist(Diarist):
    """Name by an in-the-moment stochastic impression, with no persistence."""

    def __init__(self, *, num_types: int, dim: int, temperature: float = 0.5,
                 rng: np.random.Generator | None = None,
                 name: str = "noisy_impression_diarist") -> None:
        super().__init__(name=name)  # no memory: nothing is retained between steps
        self.temperature = temperature
        self._rng = rng if rng is not None else np.random.default_rng(0)
        # A fixed projection to a term-logit space; the same percept gives the same
        # logits, but sampling adds per-step noise -> "whatever seems right".
        self._proj = self._rng.standard_normal((dim, num_types)).astype(np.float32)

    def assign_term(self, percept: np.ndarray) -> int:
        """Sample a term from a fixed percept projection (temperature 0 = argmax)."""
        logits = percept @ self._proj
        if self.temperature > 0:
            logits = logits / self.temperature
            logits = logits - logits.max()
            probs = np.exp(logits)
            probs /= probs.sum()
            return int(self._rng.choice(len(probs), p=probs))
        return int(np.argmax(logits))


def make_diarist(cfg: DiaristConfig, *, num_types: int, dim: int,
                 rng: np.random.Generator | None = None) -> Diarist:
    """Build the diarist selected by ``cfg`` (``num_types``/``dim``/``rng`` for
    the policies that need them)."""
    if cfg.policy == "prototype":
        return PrototypeDiarist(memory_mode=cfg.memory, window=cfg.window,
                                threshold=cfg.threshold)
    if cfg.policy == "fixed_quantizer":
        return FixedQuantizerDiarist(num_bins=cfg.quantizer_bins, scale=cfg.quantizer_scale)
    if cfg.policy == "noisy_impression":
        return NoisyImpressionDiarist(num_types=num_types, dim=dim,
                                      temperature=cfg.impression_temp, rng=rng)
    raise ValueError(f"unknown diarist policy: {cfg.policy!r}")
