# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""E4 behavioral layer: the teachable rule (Kripke's quus).

A teacher supplies finite examples that are all *below the bend-point*, where
``plus`` and ``quus`` agree, so the data underdetermines the rule. Several seeded
students then extrapolate to the *above-bend* pairs, and we record what each
predicts. Comparing the students shows whether they **converge** (a shared
inductive prior quietly resolving the underdetermination) or **diverge**
(underdetermination made visible).

Only the students' above-bend predictions are behavior; the God's-eye ``plus`` /
``quus`` targets are used solely by the analysis to label what was extrapolated.
"""

from __future__ import annotations

from typing import Any

from beetlebox.agents.rule_learner import RuleLearner
from beetlebox.config import E4RunConfig
from beetlebox.envs.quus import QuusTask
from beetlebox.runlog import RunLogger
from beetlebox.seeding import seed_everything


class E4RunManager:
    """Trains ``num_seeds`` students on identical below-bend data and reads out
    their above-bend extrapolations."""

    def __init__(self, cfg: E4RunConfig, logger: RunLogger | None = None) -> None:
        self.cfg = cfg
        self.logger = logger
        self.task = QuusTask(cfg.quus)

    def run(self) -> dict[str, Any]:
        """Train each seeded student and return their above-bend predictions."""
        task = self.task
        x_train = task.features(task.train_pairs)
        y_train = task.plus(task.train_pairs).astype(float)  # below bend: plus == quus
        x_test = task.features(task.test_pairs)

        predictions: list[list[int]] = []
        for s in range(self.cfg.experiment.num_seeds):
            seed_everything(self.cfg.seed + s)  # distinct student, identical data
            student = RuleLearner(task.feature_dim, self.cfg.learner)
            student.fit(x_train, y_train, device=self.cfg.device)
            predictions.append([int(p) for p in student.predict(x_test, device=self.cfg.device)])

        summary = {
            "experiment": self.cfg.experiment.name,
            "encoding": self.cfg.quus.encoding,
            "num_seeds": self.cfg.experiment.num_seeds,
            "bend": task.bend,
            "num_train": len(task.train_pairs),
            "num_test": len(task.test_pairs),
        }
        if self.logger is not None:
            self.logger.log("run_start", **summary)
            self.logger.log("predictions", predictions=predictions,
                            plus=[int(v) for v in task.plus(task.test_pairs)],
                            quus=[int(v) for v in task.quus(task.test_pairs)])
            self.logger.log("run_end", **summary)
        summary["predictions"] = predictions
        summary["plus"] = [int(v) for v in task.plus(task.test_pairs)]
        summary["quus"] = [int(v) for v in task.quus(task.test_pairs)]
        return summary
