# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""E3 orchestration: the beetle-box (Philosophical Investigations §293).

A population plays a sensation-language-game -- coordinating through public words
while each agent holds a private "box" whose contents we control. Three
operationalizations are all supported as a config axis (this is a framework for
exploration, not a single design); see ``docs/e3_design.md`` for the full catalog:

- ``private_referent``    : the sender's ONLY access to the referent is its box; a
                            box-aware receiver identifies the referent. Beetle-box
                            result = shared - divergent coordination gap.
- ``sensation_matching``  : two symmetric agents privately sense a "type", exchange
                            public words, and jointly judge same/different.
- ``public_referent_aux`` : standard referential game (referent public to sender)
                            plus a private box as an auxiliary side-channel.

Each runs under the four box conditions (shared / divergent / empty / noise). The
box is a real network input (earnable cancellation, §3.3), so whether the private
signal -- and its private *form* -- matters is measured, not assumed.

Reuses the E1 harness pieces (channel, env, neural agents, logging).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from beetlebox.agents import BoxReceiver, MatchingAgent, NeuralSender
from beetlebox.boxes import BoxScheme
from beetlebox.channels import SymbolChannel
from beetlebox.config import E3RunConfig
from beetlebox.envs import SignalingEnv
from beetlebox.runlog import RunLogger
from beetlebox.seeding import seed_everything

SENDER, RECEIVER = 0, 1  # agent indices into the BoxScheme


class E3RunManager:
    """Owns one E3 run end-to-end for the selected game + box condition."""

    def __init__(self, cfg: E3RunConfig, logger: RunLogger | None = None) -> None:
        self.cfg = cfg
        self.logger = logger
        seed_everything(cfg.seed)
        self.rng = np.random.default_rng(cfg.seed)
        self.device = torch.device(cfg.device)
        self.game = cfg.experiment.game

        self.env = SignalingEnv(cfg.env)
        self.channel = SymbolChannel.from_config(cfg.channel)
        self.num_states = self.env.num_classes
        self.box = BoxScheme(cfg.box.condition, num_agents=2, num_states=self.num_states,
                             box_dim=cfg.box.box_dim, rng=self.rng)
        self._build_agents()
        params = [p for m in self._modules for p in m.parameters()]
        self.optimizer = torch.optim.Adam(params, lr=cfg.agent.learning_rate)
        self.baseline = 0.0

    # ------------------------------------------------------------------ #
    def _build_agents(self) -> None:
        c, a = self.cfg, self.cfg.agent
        bd = c.box.box_dim
        v, ml = self.channel.vocab_size, self.channel.message_length
        if self.game == "private_referent":
            self.sender = NeuralSender(bd, v, ml, a.embed_dim, a.hidden_dim).to(self.device)
            self.receiver = BoxReceiver(v, ml, self.num_states, bd, a.embed_dim,
                                        a.hidden_dim).to(self.device)
            self._modules = [self.sender, self.receiver]
        elif self.game == "public_referent_aux":
            self.sender = NeuralSender(self.env.feature_dim + bd, v, ml, a.embed_dim,
                                       a.hidden_dim).to(self.device)
            self.receiver = BoxReceiver(v, ml, self.num_states, bd, a.embed_dim,
                                        a.hidden_dim).to(self.device)
            self._modules = [self.sender, self.receiver]
        elif self.game == "sensation_matching":
            self.agent_a = MatchingAgent(v, ml, bd, a.embed_dim, a.hidden_dim,
                                         name="matcher_a").to(self.device)
            self.agent_b = MatchingAgent(v, ml, bd, a.embed_dim, a.hidden_dim,
                                         name="matcher_b").to(self.device)
            self._modules = [self.agent_a, self.agent_b]
        else:
            raise ValueError(f"unknown E3 game: {self.game!r}")

    def _t(self, arr: np.ndarray, dtype=torch.float32) -> torch.Tensor:
        return torch.as_tensor(arr, dtype=dtype, device=self.device)

    def _log(self, event: str, **fields: Any) -> None:
        if self.logger is not None:
            self.logger.log(event, **fields)

    # ------------------------------------------------------------------ #
    # Referential games (private_referent, public_referent_aux)
    # ------------------------------------------------------------------ #
    def _referential_batch(self, batch_size: int, *, greedy: bool = False):
        idx = self.rng.integers(0, self.num_states, size=batch_size)
        target = self._t(idx, torch.long)
        box_s = self._t(self.box.signal(SENDER, idx))
        box_r = self._t(self.box.signal(RECEIVER, idx))
        if self.game == "public_referent_aux":
            feats = self._t(self.env.features_for(idx))
            sender_in = torch.cat([feats, box_s], dim=-1)
        else:  # private_referent: the box IS the sender's only window onto r
            sender_in = box_s
        message, logprob, entropy = self.sender.act(sender_in, greedy=greedy)
        logits = self.receiver(message, box_r)
        return target, logits, logprob, entropy

    def _referential_step(self) -> dict[str, float]:
        target, logits, logprob, entropy = self._referential_batch(self.cfg.experiment.batch_size)
        pred = logits.argmax(dim=-1)
        reward = (pred == target).float()
        receiver_loss = F.cross_entropy(logits, target)
        advantage = reward - self.baseline
        sender_loss = -(advantage.detach() * logprob).mean() \
            - self.cfg.agent.entropy_coef * entropy.mean()
        loss = sender_loss + receiver_loss
        if self.cfg.experiment.feedback:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.baseline = 0.99 * self.baseline + 0.01 * float(reward.mean())
        return {"train_acc": float(reward.mean())}

    @torch.no_grad()
    def _referential_eval(self) -> float:
        accs = []
        for _ in range(self.cfg.experiment.eval_batches):
            target, logits, _, _ = self._referential_batch(self.cfg.experiment.batch_size,
                                                            greedy=True)
            accs.append(float((logits.argmax(dim=-1) == target).float().mean()))
        return float(np.mean(accs))

    # ------------------------------------------------------------------ #
    # Sensation same/different game
    # ------------------------------------------------------------------ #
    def _matching_batch(self, batch_size: int):
        t_a = self.rng.integers(0, self.num_states, size=batch_size)
        same = self.rng.random(batch_size) < 0.5
        t_b = np.where(same, t_a, (t_a + self.rng.integers(1, self.num_states, batch_size))
                       % self.num_states)
        y = self._t((t_a == t_b).astype(np.int64), torch.long)
        box_a = self._t(self.box.signal(SENDER, t_a))
        box_b = self._t(self.box.signal(RECEIVER, t_b))
        return box_a, box_b, y

    def _matching_step(self) -> dict[str, float]:
        box_a, box_b, y = self._matching_batch(self.cfg.experiment.batch_size)
        m_a, lp_a, ent_a = self.agent_a.speak(box_a)
        m_b, lp_b, ent_b = self.agent_b.speak(box_b)
        j_a = self.agent_a.judge(box_a, m_b)
        j_b = self.agent_b.judge(box_b, m_a)
        correct_a = (j_a.argmax(-1) == y).float()
        correct_b = (j_b.argmax(-1) == y).float()
        reward = correct_a * correct_b  # both judge correctly
        judge_loss = F.cross_entropy(j_a, y) + F.cross_entropy(j_b, y)
        advantage = reward - self.baseline
        speak_loss = -(advantage.detach() * (lp_a + lp_b)).mean() \
            - self.cfg.agent.entropy_coef * (ent_a + ent_b).mean()
        loss = judge_loss + speak_loss
        if self.cfg.experiment.feedback:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.baseline = 0.99 * self.baseline + 0.01 * float(reward.mean())
        return {"train_acc": float(((correct_a + correct_b) / 2).mean())}

    @torch.no_grad()
    def _matching_eval(self) -> float:
        accs = []
        for _ in range(self.cfg.experiment.eval_batches):
            box_a, box_b, y = self._matching_batch(self.cfg.experiment.batch_size)
            m_a, _, _ = self.agent_a.speak(box_a, greedy=True)
            m_b, _, _ = self.agent_b.speak(box_b, greedy=True)
            ca = (self.agent_a.judge(box_a, m_b).argmax(-1) == y).float()
            cb = (self.agent_b.judge(box_b, m_a).argmax(-1) == y).float()
            accs.append(float(((ca + cb) / 2).mean()))
        return float(np.mean(accs))

    # ------------------------------------------------------------------ #
    def _train_step(self) -> dict[str, float]:
        return (self._matching_step() if self.game == "sensation_matching"
                else self._referential_step())

    def evaluate(self) -> float:
        return (self._matching_eval() if self.game == "sensation_matching"
                else self._referential_eval())

    def run(self) -> dict[str, Any]:
        exp = self.cfg.experiment
        chance = 0.5 if self.game == "sensation_matching" else 1.0 / self.num_states
        self._log("run_start", game=self.game, condition=self.cfg.box.condition,
                  chance=chance, feedback=exp.feedback, box_informative=self.box.is_informative)
        for step in range(1, exp.num_steps + 1):
            last = self._train_step()
            if step % exp.eval_every == 0 or step == exp.num_steps:
                acc = self.evaluate()
                self._log("eval", step=step, accuracy=acc, **last)
        final = self.evaluate()
        summary = {
            "game": self.game,
            "condition": self.cfg.box.condition,
            "final_accuracy": final,
            "chance": chance,
            "box_informative": self.box.is_informative,
        }
        self._log("run_end", **summary)
        return summary
