# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""E5 orchestration: forms of life / grounding (capstone).

The same signaling game as E1, run two ways:

- **ungrounded** (``grounded=False``): reward = identify the referent. Symbols are
  pure labels; they connect to nothing beyond the identification game.
- **grounded** (``grounded=True``): the receiver chooses an **action** and the
  reward is the world **payoff** of that action for the true referent, with
  per-referent **stakes** (:mod:`beetlebox.envs.grounded`). Words now drive real,
  differential consequences — a resource/survival task.

The question is whether grounding changes the *character* of the emergent language.
The run records four things for each regime so the analysis can compare them:
performance, convention stability, compositionality (topographic similarity),
**robustness** to channel noise, and **transfer** to a fresh receiver (turnover).

Reuses E1's :class:`~beetlebox.agents.NeuralSender` /
:class:`~beetlebox.agents.NeuralReceiver` and channel unchanged; only the receiver's
output space and the reward differ between regimes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from beetlebox.agents import NeuralReceiver, NeuralSender
from beetlebox.channels import SymbolChannel
from beetlebox.config import E5RunConfig
from beetlebox.envs import SignalingEnv
from beetlebox.envs.grounded import GroundedWorld
from beetlebox.runlog import RunLogger
from beetlebox.seeding import seed_everything


class E5RunManager:
    """Owns one E5 run for the grounded or ungrounded regime."""

    def __init__(self, cfg: E5RunConfig, logger: RunLogger | None = None) -> None:
        self.cfg = cfg
        self.logger = logger
        seed_everything(cfg.seed)
        self.rng = np.random.default_rng(cfg.seed)
        self.device = torch.device(cfg.device)
        self.grounded = cfg.experiment.grounded

        self.env = SignalingEnv.from_config(cfg.env)
        self.channel = SymbolChannel.from_config(cfg.channel)
        self.num_states = self.env.num_classes
        self.world = GroundedWorld(self.env) if self.grounded else None
        out_dim = self.world.num_actions if self.grounded else self.num_states

        self.sender = NeuralSender(self.env.feature_dim, self.channel.vocab_size,
                                   self.channel.message_length, cfg.agent.embed_dim,
                                   cfg.agent.hidden_dim).to(self.device)
        self.receiver = NeuralReceiver(self.channel.vocab_size, self.channel.message_length,
                                       out_dim, cfg.agent.embed_dim,
                                       cfg.agent.hidden_dim).to(self.device)
        if self.grounded:
            self._payoff = torch.as_tensor(self.world.payoff, dtype=torch.float32,
                                           device=self.device)
        self.optimizer = self._build_optimizer(
            list(self.sender.parameters()) + list(self.receiver.parameters()))
        self.baseline = 0.0

    def _build_optimizer(self, params) -> torch.optim.Optimizer:
        return torch.optim.Adam(params, lr=self.cfg.agent.learning_rate)

    def _log(self, event: str, **fields: Any) -> None:
        if self.logger is not None:
            self.logger.log(event, **fields)

    # ------------------------------------------------------------------ #
    def _train_step(self) -> dict[str, float]:
        idx, feats_np = self.env.sample_batch(self.rng, self.cfg.experiment.batch_size)
        feats = torch.as_tensor(feats_np, dtype=torch.float32, device=self.device)
        target = torch.as_tensor(idx, dtype=torch.long, device=self.device)
        message, logprob, entropy = self.sender.act(feats)
        logits = self.receiver(message)

        if self.grounded:
            probs = logits.softmax(dim=-1)
            payoff_rows = self._payoff[target]  # [B, num_actions]
            reward = (probs * payoff_rows).sum(dim=-1)  # expected payoff (differentiable)
            receiver_loss = -reward.mean()
        else:
            reward = (logits.argmax(dim=-1) == target).float()
            receiver_loss = F.cross_entropy(logits, target)

        advantage = reward - self.baseline
        sender_loss = -(advantage.detach() * logprob).mean() \
            - self.cfg.agent.entropy_coef * entropy.mean()
        loss = sender_loss + receiver_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.baseline = 0.99 * self.baseline + 0.01 * float(reward.mean().detach())
        return {"reward": float(reward.mean().detach())}

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _greedy_messages(self) -> torch.Tensor:
        idx = np.arange(self.num_states)
        feats = torch.as_tensor(self.env.features_for(idx), dtype=torch.float32,
                                device=self.device)
        message, _, _ = self.sender.act(feats, greedy=True)
        return message

    def _apply_noise(self, message: torch.Tensor, eps: float) -> torch.Tensor:
        """Flip each symbol to a uniform-random one with probability ``eps``."""
        if eps <= 0:
            return message
        flip = torch.as_tensor(self.rng.random(message.shape) < eps, device=message.device)
        rand = torch.as_tensor(
            self.rng.integers(0, self.channel.vocab_size, size=tuple(message.shape)),
            dtype=message.dtype, device=message.device)
        return torch.where(flip, rand, message)

    @torch.no_grad()
    def _performance(self, eps: float = 0.0) -> float:
        """Native performance over all referents: id-accuracy (ungrounded) or
        normalized mean payoff (grounded), optionally under channel noise ``eps``."""
        idx = np.arange(self.num_states)
        message = self._apply_noise(self._greedy_messages(), eps)
        logits = self.receiver(message)
        choice = logits.argmax(dim=-1).cpu().numpy()
        if self.grounded:
            mean_payoff = float(self.world.payoff_for(idx, choice).mean())
            return mean_payoff / self.world.max_mean_payoff
        return float((choice == idx).mean())

    def evaluate(self) -> dict[str, Any]:
        """Greedy performance and the referent->message convention map."""
        mapping = self._greedy_messages().cpu().numpy().tolist()
        return {"performance": self._performance(), "mapping": mapping}

    # ------------------------------------------------------------------ #
    def _do_turnover(self, step: int) -> None:
        for p in self.sender.parameters():
            p.requires_grad_(False)
        self.receiver.reset_parameters()
        self.optimizer = self._build_optimizer(list(self.receiver.parameters()))
        self.baseline = 0.0
        self._log("turnover", step=step)

    def run(self) -> dict[str, Any]:
        """Train the regime, then measure performance, robustness (channel noise),
        and transfer (turnover); return the run summary."""
        exp = self.cfg.experiment
        turnover_step = int(exp.turnover_at * exp.num_steps) if exp.turnover else None
        self._log("run_start", grounded=self.grounded, num_classes=self.num_states,
                  bandwidth=self.channel.bandwidth, turnover_step=turnover_step)
        pre_turnover_perf = None
        for step in range(1, exp.num_steps + 1):
            if turnover_step is not None and step == turnover_step + 1:
                pre_turnover_perf = self._performance()
                self._do_turnover(step)
            self._train_step()
            if step % exp.eval_every == 0 or step == exp.num_steps:
                ev = self.evaluate()
                self._log("eval", step=step, performance=ev["performance"],
                          mapping=ev["mapping"])

        final = self.evaluate()
        clean = final["performance"]
        noisy = self._performance(eps=exp.robustness_noise)
        summary = {
            "grounded": self.grounded,
            "performance": clean,
            "robustness_noise": exp.robustness_noise,
            "performance_under_noise": noisy,
            "robustness_ratio": (noisy / clean) if clean > 0 else 0.0,
            "final_mapping": final["mapping"],
            "num_classes": self.num_states,
            "bandwidth": self.channel.bandwidth,
            "turnover_step": turnover_step,
            "pre_turnover_performance": pre_turnover_perf,
            "post_turnover_performance": clean if turnover_step is not None else None,
        }
        self._log("run_end", **summary)
        return summary
