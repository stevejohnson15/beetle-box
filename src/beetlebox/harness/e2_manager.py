# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""E2 orchestration: the private diarist (*Philosophical Investigations* §258).

A single diarist watches a private percept stream and names each percept with a
self-invented term, under **no external correction**. There is no reward and no
training: the run simply records what the diarist does so the analysis can ask
whether "same again" stabilizes or wanders (:mod:`beetlebox.analysis.e2`).

Each step records the God's-eye type (for scoring only), the term the diarist
emitted, and the **gap** since that type last appeared (for the gap-conditioned
"does same-again fade with time?" metric). The full sequence is written to the
run log so the analysis is reconstructable from the event stream alone.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from beetlebox.agents.diarist import make_diarist
from beetlebox.config import E2RunConfig
from beetlebox.envs.percept import PerceptStream
from beetlebox.runlog import RunLogger
from beetlebox.seeding import seed_everything


class E2RunManager:
    """Owns one E2 run: stream a percept sequence through a diarist, record terms."""

    def __init__(self, cfg: E2RunConfig, logger: RunLogger | None = None) -> None:
        self.cfg = cfg
        self.logger = logger
        seed_everything(cfg.seed)
        self.rng = np.random.default_rng(cfg.seed)
        self.stream = PerceptStream(cfg.percept, self.rng)
        # The diarist gets its own rng stream so stochastic policies are seeded
        # independently of the percept draws.
        self.diarist = make_diarist(cfg.diarist, num_types=cfg.percept.num_types,
                                    dim=cfg.percept.dim,
                                    rng=np.random.default_rng(cfg.seed + 1))

    def run(self) -> dict[str, Any]:
        """Run the diary for ``num_steps`` and return the recorded sequence + summary."""
        exp = self.cfg.experiment
        types, percepts = self.stream.sample(exp.num_steps)
        terms: list[int] = []
        gaps: list[int] = []
        last_seen: dict[int, int] = {}
        for step in range(exp.num_steps):
            t = int(types[step])
            terms.append(self.diarist.assign_term(percepts[step]))
            gaps.append(step - last_seen[t] if t in last_seen else -1)  # -1 = first sight
            last_seen[t] = step

        types_list = [int(x) for x in types]
        summary = {
            "experiment": exp.name,
            "policy": self.cfg.diarist.policy,
            "memory": self.cfg.diarist.memory,
            "num_types": self.cfg.percept.num_types,
            "noise": self.cfg.percept.noise,
            "num_steps": exp.num_steps,
            "distinct_terms": len(set(terms)),
        }
        if self.logger is not None:
            self.logger.log("run_start", **{k: summary[k] for k in
                            ("policy", "memory", "num_types", "noise", "num_steps")})
            # The whole sequence in one event -> analysis reconstructs from the log.
            self.logger.log("sequence", types=types_list, terms=terms, gaps=gaps)
            self.logger.log("run_end", **summary)
        # Attach the sequence for in-process callers (e.g. the notebook).
        summary["types"] = types_list
        summary["terms"] = terms
        summary["gaps"] = gaps
        return summary
