# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""The from-scratch rule learner for E4's behavioral layer.

A small MLP "student" is trained only on the below-bend quus pairs, where ``plus``
and ``quus`` (and infinitely many other rules) agree. It learns to **regress the
result** (the arithmetic function), and its behavior *above* the bend is then read
out — rounded to the nearest integer — to see which rule it extrapolated. Because
the training data does not fix the rule, different random seeds can extrapolate
differently: that divergence is Kripke's underdetermination made visible, while
convergence across seeds shows a shared inductive prior quietly resolving it (which
is exactly what the operand encoding controls; see :mod:`beetlebox.envs.quus`).

The student is deliberately minimal (one hidden layer) so that whatever regularity
it extends past the bend comes from its inductive bias, not from a built-in rule.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from beetlebox.agents.base import Agent
from beetlebox.config import RuleLearnerConfig


class RuleLearner(nn.Module, Agent):
    """A small MLP student that regresses the result from below-bend examples."""

    def __init__(self, input_dim: int, cfg: RuleLearnerConfig, *,
                 name: str = "rule_learner") -> None:
        nn.Module.__init__(self)
        Agent.__init__(self, name=name)
        self.cfg = cfg
        self.net = nn.Sequential(
            nn.Linear(input_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return the scalar predicted result, shape ``[B]``."""
        return self.net(features).squeeze(-1)

    def fit(self, features: np.ndarray, targets: np.ndarray, *, device: str = "cpu") -> None:
        """Train on the (below-bend) examples via full-batch AdamW + MSE."""
        dev = torch.device(device)
        self.to(dev)
        x = torch.as_tensor(features, dtype=torch.float32, device=dev)
        y = torch.as_tensor(targets, dtype=torch.float32, device=dev)
        opt = torch.optim.AdamW(self.parameters(), lr=self.cfg.learning_rate,
                                weight_decay=self.cfg.weight_decay)
        loss_fn = nn.MSELoss()
        self.train()
        for _ in range(self.cfg.num_steps):
            opt.zero_grad()
            loss = loss_fn(self.forward(x), y)
            loss.backward()
            opt.step()

    @torch.no_grad()
    def predict(self, features: np.ndarray, *, device: str = "cpu") -> np.ndarray:
        """Predicted results, rounded to the nearest integer."""
        self.eval()
        x = torch.as_tensor(features, dtype=torch.float32, device=torch.device(device))
        return np.rint(self.forward(x).cpu().numpy()).astype(np.int64)

    def reset_parameters(self) -> None:
        """Reinitialize the student's weights from scratch."""
        for module in self.net:
            if isinstance(module, nn.Linear):
                module.reset_parameters()
