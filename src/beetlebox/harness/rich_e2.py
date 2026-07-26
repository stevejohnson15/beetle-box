# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""Rich-mode E2: the private diarist (§258) with a frontier agent.

The clean-room diarist (:mod:`beetlebox.agents.diarist`) is an explicit policy;
this is the rich-but-confounded arm (plan §3.2): a frontier model names a private
percept stream *in context*, studying how a pretrained agent handles "same again"
when its only reference is its own diary.

Each step the model sees a percept "reading" (the vector, rounded) and must either
reuse one of the terms it has already coined or coin a new one. The **memory
toggle** controls how much of the diary is placed in context:

  full     - all past (reading -> term) entries
  windowed - only the last ``window`` entries
  none     - nothing; each reading is judged with no diary at all

With no diary the model has nothing to check "same again" against — the §258
predicament in the first person. The agent is injectable, so the prompt/coining
logic is unit-tested without any API call.

**Cost warning.** One API call per step; keep ``num_steps`` small.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from beetlebox.config import PerceptConfig
from beetlebox.envs.percept import PerceptStream
from beetlebox.harness.rich_e3 import _Chooser  # shared minimal agent interface
from beetlebox.runlog import RunLogger

_NEW = "coin a brand-new term"


class RichDiaristRunner:
    """In-context private diarist played by a frontier agent."""

    def __init__(self, *, memory: str = "full", window: int = 10, num_types: int = 4,
                 dim: int = 4, noise: float = 0.25, num_steps: int = 24,
                 model: str | None = None, seed: int = 0,
                 logger: RunLogger | None = None, agent: _Chooser | None = None) -> None:
        if memory not in ("full", "windowed", "none"):
            raise ValueError(f"unknown memory mode: {memory!r}")
        self.memory = memory
        self.window = window
        self.num_steps = num_steps
        self.logger = logger
        self.rng = np.random.default_rng(seed)
        self.stream = PerceptStream(
            PerceptConfig(num_types=num_types, dim=dim, noise=noise), self.rng)
        if agent is None:
            from beetlebox.agents.api_model import DEFAULT_MODEL, ApiAgent
            agent = ApiAgent(name="diarist", model=model or DEFAULT_MODEL)
        self.agent = agent
        self.model = model

    @staticmethod
    def _reading(percept: np.ndarray) -> str:
        return "[" + ", ".join(f"{x:.1f}" for x in percept) + "]"

    def _system(self) -> str:
        return (
            "You are keeping a private diary of sensations. Each entry is a numeric "
            "'reading'. Give the SAME term to readings that feel like the same recurring "
            "sensation, and a NEW term to ones that feel different. No one corrects you; "
            "be as consistent as your record allows."
        )

    def _user(self, diary: list[tuple[str, int]], reading: str, coined: list[int]) -> str:
        if self.memory == "none" or not diary:
            diary_text = "(you are keeping no diary; you have no past entries to consult)"
        else:
            shown = diary if self.memory == "full" else diary[-self.window:]
            diary_text = "\n".join(f"  reading {r} -> term {t}" for r, t in shown)
        options = "".join(f"\n  {i}: reuse term {t}" for i, t in enumerate(coined))
        options += f"\n  {len(coined)}: {_NEW}"
        return (f"Your diary so far:\n{diary_text}\n\n"
                f"New reading: {reading}\nChoose one option (reply with its number):{options}")

    def run(self) -> dict[str, Any]:
        """Name ``num_steps`` percepts in context; return the recorded sequence."""
        types_arr, percepts = self.stream.sample(self.num_steps)
        diary: list[tuple[str, int]] = []
        coined: list[int] = []  # term ids coined so far, in coin order
        types: list[int] = []
        terms: list[int] = []
        gaps: list[int] = []
        last_seen: dict[int, int] = {}
        if self.logger is not None:
            self.logger.log("rich_run_start", experiment="e2_diarist", memory=self.memory,
                            num_steps=self.num_steps, model=self.model)
        for step in range(self.num_steps):
            reading = self._reading(percepts[step])
            choice = self.agent.choose(self._system(), self._user(diary, reading, coined),
                                       range(len(coined) + 1))
            if choice >= len(coined):  # coin a new term
                term = len(coined)
                coined.append(term)
            else:
                term = coined[choice]
            diary.append((reading, term))
            t = int(types_arr[step])
            types.append(t)
            terms.append(term)
            gaps.append(step - last_seen[t] if t in last_seen else -1)
            last_seen[t] = step

        summary = {
            "experiment": "e2_diarist", "mode": "rich", "memory": self.memory,
            "num_steps": self.num_steps, "distinct_terms": len(set(terms)),
            "api_calls": self.agent.num_calls,
            "usage": dict(self.agent.usage),
            "types": types, "terms": terms, "gaps": gaps,
        }
        if self.logger is not None:
            self.logger.log("sequence", types=types, terms=terms, gaps=gaps)
            self.logger.log("rich_run_end", **{k: v for k, v in summary.items()
                                                if k not in ("types", "terms", "gaps")})
        return summary
