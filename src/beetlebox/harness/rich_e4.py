# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""Rich-mode E4: the teachable rule (quus) with a frontier agent.

The clean-room students learn from scratch; this is the rich-but-confounded arm
(plan §3.2). A frontier model is shown the same below-bend examples (where plus and
quus agree) and asked to extrapolate above the bend. A pretrained model arrives
with the strongest possible inductive prior for this task -- ordinary arithmetic,
an inherited form of life -- so the question is how forcefully that prior resolves
the underdetermination toward plus.

One API call per queried above-bend pair; the agent is injectable so the
prompt/scoring logic is unit-tested without a network. Keep ``num_queries`` small.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from beetlebox.config import QuusConfig
from beetlebox.envs.quus import QuusTask
from beetlebox.harness.rich_e3 import _Chooser  # shared minimal agent interface
from beetlebox.runlog import RunLogger


class RichRuleLearnerRunner:
    """In-context quus extrapolation by a frontier agent."""

    def __init__(self, *, max_operand: int = 12, bend: int = 8, modulus: int = 24,
                 quus_value: int = 0, num_queries: int = 8, model: str | None = None,
                 seed: int = 0, logger: RunLogger | None = None,
                 agent: _Chooser | None = None) -> None:
        self.task = QuusTask(QuusConfig(max_operand=max_operand, bend=bend,
                                        modulus=modulus, quus_value=quus_value))
        self.modulus = modulus
        self.num_queries = num_queries
        self.logger = logger
        self.rng = np.random.default_rng(seed)
        if agent is None:
            from beetlebox.agents.api_model import DEFAULT_MODEL, ApiAgent
            agent = ApiAgent(name="rule_learner", model=model or DEFAULT_MODEL)
        self.agent = agent
        self.model = model

    def _system(self) -> str:
        return (
            "You are learning a binary operation, written a # b, purely from examples. "
            "Infer the pattern from the examples given and answer each new instance with "
            "the single integer the operation yields. Do not assume it is ordinary "
            "addition unless the examples warrant it."
        )

    def _examples_block(self) -> str:
        pairs = self.task.train_pairs
        vals = self.task.plus(pairs)  # below the bend, plus == quus
        return "\n".join(f"  {a} # {b} = {v}" for (a, b), v in zip(pairs, vals, strict=True))

    def run(self) -> dict[str, Any]:
        """Query ``num_queries`` above-bend pairs and record what the model answers."""
        test = self.task.test_pairs
        idx = self.rng.choice(len(test), size=min(self.num_queries, len(test)), replace=False)
        queried = test[idx]
        plus = self.task.plus(queried)
        quus = self.task.quus(queried)
        if self.logger is not None:
            self.logger.log("rich_run_start", experiment="e4_quus",
                            num_queries=len(queried), model=self.model)
        answers: list[int] = []
        for a, b in queried:
            user = (f"Examples:\n{self._examples_block()}\n\n"
                    f"Now answer: {a} # {b} = ?  (reply with the integer)")
            answers.append(self.agent.choose(self._system(), user, range(self.modulus)))

        plus_rate = float(np.mean([p == t for p, t in zip(answers, plus, strict=True)]))
        quus_rate = float(np.mean([p == t for p, t in zip(answers, quus, strict=True)]))
        summary = {
            "experiment": "e4_quus", "mode": "rich", "num_queries": len(queried),
            "plus_rate": plus_rate, "quus_rate": quus_rate,
            "api_calls": self.agent.num_calls, "usage": dict(self.agent.usage),
            "queried": [[int(a), int(b)] for a, b in queried], "answers": answers,
        }
        if self.logger is not None:
            self.logger.log("rich_run_end", **{k: v for k, v in summary.items()
                                                if k not in ("queried", "answers")})
        return summary
