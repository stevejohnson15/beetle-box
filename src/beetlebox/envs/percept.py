# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""The private percept stream for E2 (the diarist, *Philosophical Investigations*
§258).

Wittgenstein's diarist writes a sign "S" whenever a private sensation recurs, with
no public criterion for "same again." We model the sensation source as a stream of
percept vectors, each drawn from one of ``K`` latent **types** (the God's-eye
ground truth the diarist never sees). A percept is its type's prototype plus
Gaussian noise:

    percept = prototype[type] + N(0, noise^2)

The noise level is the "clean vs. noisy same-again" manipulation: with small noise
the types are well separated (a real recurrence is obvious); with large noise the
types overlap and "same again" becomes genuinely ambiguous. Types are drawn i.i.d.,
so a given type recurs after random **time gaps** — which lets the analysis ask
whether "same again" degrades as the gap since a type last appeared grows.

Only the analysis uses the God's-eye ``types``; the diarist sees ``percepts`` alone.
NumPy only.
"""

from __future__ import annotations

import numpy as np

from beetlebox.config import PerceptConfig


class PerceptStream:
    """Generator of a private percept stream over ``K`` latent types."""

    def __init__(self, cfg: PerceptConfig, rng: np.random.Generator) -> None:
        if cfg.num_types < 1:
            raise ValueError("num_types must be >= 1")
        if cfg.dim < 1:
            raise ValueError("dim must be >= 1")
        self.num_types = cfg.num_types
        self.dim = cfg.dim
        self.noise = cfg.noise
        self._rng = rng
        # Fixed prototypes, one per type: standard-normal (seeded via the run rng).
        # In ``dim`` dimensions two independent prototypes sit ~sqrt(2*dim) apart,
        # while two percepts of the SAME type differ by ~noise*sqrt(2*dim). So for
        # small ``noise`` the types are cleanly separated ("same again" is obvious)
        # and for large ``noise`` their clouds overlap ("same again" is ambiguous)
        # -- the clean-vs-noisy manipulation, with no rescaling needed.
        self._prototypes = rng.standard_normal((cfg.num_types, cfg.dim)).astype(np.float32)

    @property
    def prototypes(self) -> np.ndarray:
        """The ``[K, dim]`` prototype matrix (God's-eye; not shown to the diarist)."""
        return self._prototypes

    def sample(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Draw ``n`` timesteps.

        Returns ``(types[n], percepts[n, dim])``. ``types`` is the God's-eye label
        used only for scoring; ``percepts`` is what the diarist observes.
        """
        types = self._rng.integers(0, self.num_types, size=n)
        noise = self._rng.standard_normal((n, self.dim)).astype(np.float32) * self.noise
        percepts = self._prototypes[types] + noise
        return types, percepts
