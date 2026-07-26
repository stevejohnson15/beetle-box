# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""E1 orchestration: the signaling-game turn loop.

``RunManager`` builds the environment, channel, and agents from a
:class:`~beetlebox.config.RunConfig`, runs the training loop (sender emits a
message; receiver reconstructs the referent; REINFORCE + cross-entropy updates
under the ``feedback`` toggle), and streams structured events to a
:class:`~beetlebox.runlog.RunLogger`. It also implements the *turnover*
manipulation as a first-class phase: freeze the converged sender, introduce a
fresh receiver, and continue -- testing whether a convention survives its
founders.

The manager is Hydra-independent; it consumes plain dataclass configs so it can
be extracted into a standalone orchestration library.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from beetlebox.agents import NeuralReceiver, NeuralSender
from beetlebox.channels import SymbolChannel
from beetlebox.config import RunConfig
from beetlebox.envs import SignalingEnv
from beetlebox.runlog import RunLogger
from beetlebox.seeding import seed_everything


class RunManager:
    """Owns one E1 run end-to-end."""

    def __init__(self, cfg: RunConfig, logger: RunLogger | None = None) -> None:
        self.cfg = cfg
        self.logger = logger
        # Seed before building nets so weight initialization is reproducible.
        seed_everything(cfg.seed)
        self.rng = np.random.default_rng(cfg.seed)
        self.device = torch.device(cfg.device)

        self.env = SignalingEnv(cfg.env)
        self.channel = SymbolChannel.from_config(cfg.channel)

        self.sender = NeuralSender(
            feature_dim=self.env.feature_dim,
            vocab_size=self.channel.vocab_size,
            message_length=self.channel.message_length,
            embed_dim=cfg.agent.embed_dim,
            hidden_dim=cfg.agent.hidden_dim,
        ).to(self.device)
        self.receiver = NeuralReceiver(
            vocab_size=self.channel.vocab_size,
            message_length=self.channel.message_length,
            num_classes=self.env.num_classes,
            embed_dim=cfg.agent.embed_dim,
            hidden_dim=cfg.agent.hidden_dim,
        ).to(self.device)

        self.optimizer = self._build_optimizer(
            list(self.sender.parameters()) + list(self.receiver.parameters())
        )
        self.baseline = 0.0  # REINFORCE reward baseline (variance reduction)

    # ------------------------------------------------------------------ #
    def _build_optimizer(self, params) -> torch.optim.Optimizer:
        return torch.optim.Adam(params, lr=self.cfg.agent.learning_rate)

    def _log(self, event: str, **fields: Any) -> None:
        if self.logger is not None:
            self.logger.log(event, **fields)

    # ------------------------------------------------------------------ #
    def _train_step(self) -> dict[str, float]:
        exp = self.cfg.experiment
        indices, features = self.env.sample_batch(self.rng, exp.batch_size)
        target = torch.as_tensor(indices, dtype=torch.long, device=self.device)
        feats = torch.as_tensor(features, dtype=torch.float32, device=self.device)

        message, logprob, entropy = self.sender.act(feats)
        logits = self.receiver(message)
        pred = logits.argmax(dim=-1)
        correct = (pred == target).float()
        reward = correct  # shared success signal

        receiver_loss = F.cross_entropy(logits, target)
        advantage = reward - self.baseline
        # REINFORCE with an entropy bonus for exploration.
        sender_loss = -(advantage.detach() * logprob).mean()
        sender_loss = sender_loss - self.cfg.agent.entropy_coef * entropy.mean()
        loss = sender_loss + receiver_loss

        # The `feedback` toggle IS the manipulation: with no success/correction
        # signal, no learning occurs and coordination should stay at chance.
        if exp.feedback:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.baseline = 0.99 * self.baseline + 0.01 * float(reward.mean())

        return {
            "train_acc": float(correct.mean()),
            "sender_loss": float(sender_loss.detach()),
            "receiver_loss": float(receiver_loss.detach()),
            "entropy": float(entropy.mean().detach()),
        }

    @torch.no_grad()
    def evaluate(self) -> dict[str, Any]:
        """Deterministic greedy evaluation over every referent.

        Returns communication accuracy and the greedy referent->message map
        (the emergent convention), used downstream for stability, transmission
        fidelity, and topographic-similarity analysis.
        """
        k = self.env.num_classes
        indices = np.arange(k)
        feats = torch.as_tensor(self.env.features_for(indices), dtype=torch.float32,
                                device=self.device)
        message, _, _ = self.sender.act(feats, greedy=True)
        pred = self.receiver.predict(message)
        target = torch.as_tensor(indices, dtype=torch.long, device=self.device)
        accuracy = float((pred == target).float().mean())
        mapping = message.cpu().numpy().tolist()  # [K][L]  sender: referent -> message
        guesses = pred.cpu().numpy().tolist()  # [K]  receiver: message -> referent guess
        # Logging guesses (not just the mapping) lets a full exchange transcript be
        # reconstructed from the event stream alone -- no model weights needed.
        return {"accuracy": accuracy, "mapping": mapping, "guesses": guesses}

    @torch.no_grad()
    def sample_exchanges(self, n: int, *, greedy: bool = False) -> list[dict[str, Any]]:
        """Sample ``n`` individual sender->receiver trials (the live "chatter").

        Each trial: a referent is drawn, the sender emits a message (stochastic
        unless ``greedy``), the receiver guesses. Returns one dict per trial with
        the referent, the message, the guess, and whether it succeeded. This is
        the per-trial view; :meth:`evaluate` is the whole-convention snapshot.
        """
        idx = self.rng.integers(0, self.env.num_classes, size=n)
        feats = torch.as_tensor(self.env.features_for(idx), dtype=torch.float32,
                                device=self.device)
        message, _, _ = self.sender.act(feats, greedy=greedy)
        guess = self.receiver.predict(message)
        msgs = message.cpu().numpy().tolist()
        guesses = guess.cpu().numpy().tolist()
        return [
            {"referent": int(idx[i]), "message": msgs[i], "guess": int(guesses[i]),
             "correct": int(idx[i]) == int(guesses[i])}
            for i in range(n)
        ]

    # ------------------------------------------------------------------ #
    def _do_turnover(self, step: int) -> None:
        """Freeze the sender, introduce a fresh receiver, keep the sender's code."""
        for p in self.sender.parameters():
            p.requires_grad_(False)
        self.receiver.reset_parameters()
        # Optimizer now trains the fresh receiver only.
        self.optimizer = self._build_optimizer(list(self.receiver.parameters()))
        self.baseline = 0.0
        self._log("turnover", step=step)

    def run(self) -> dict[str, Any]:
        """Train for ``num_steps`` (applying turnover if configured), logging periodic
        evals; return the run summary (final accuracy, mapping, guesses, chance)."""
        exp = self.cfg.experiment
        turnover_step = int(exp.turnover_at * exp.num_steps) if exp.turnover else None
        chance = 1.0 / self.env.num_classes
        self._log("run_start", chance=chance, num_classes=self.env.num_classes,
                  bandwidth=self.channel.bandwidth, feedback=exp.feedback,
                  turnover_step=turnover_step)

        last_train: dict[str, float] = {}
        for step in range(1, exp.num_steps + 1):
            if turnover_step is not None and step == turnover_step + 1:
                self._do_turnover(step)
            last_train = self._train_step()
            if step % exp.eval_every == 0 or step == exp.num_steps:
                ev = self.evaluate()
                self._log("eval", step=step, accuracy=ev["accuracy"],
                          mapping=ev["mapping"], guesses=ev["guesses"], **last_train)

        final = self.evaluate()
        summary = {
            "final_accuracy": final["accuracy"],
            "chance": chance,
            "final_mapping": final["mapping"],
            "final_guesses": final["guesses"],
            "turnover_step": turnover_step,
            "num_classes": self.env.num_classes,
            "bandwidth": self.channel.bandwidth,
        }
        self._log("run_end", **summary)
        return summary
