# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""Grokking training loop for the modular-addition transformer (E4).

Trains :class:`~beetlebox.mech.transformer.ModularAdditionTransformer` on a random
train/val split of all ``p*p`` addition pairs, with weight decay — the ingredient
that produces the memorize→generalize (grokking) transition. It records the
train/val accuracy curve and a description-length **proxy** (the parameter L2 norm)
over training, so the analysis can look for the classic pattern: train accuracy
saturates early (memorization) while val accuracy jumps much later (generalization),
around when the weight norm falls (the network compressing onto a simpler circuit,
per DeMoss et al.).

This is heavy compute relative to the rest of Beetle-Box: a real grokking run is
thousands–tens-of-thousands of steps. Tests use tiny/fast configs; a full run is a
"launch it deliberately" job (see ``docs/e4_quus.md``), analogous to rich mode.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from beetlebox.config import GrokkingConfig
from beetlebox.mech.transformer import ModularAdditionTransformer
from beetlebox.runlog import RunLogger
from beetlebox.seeding import seed_everything


def _all_pairs(modulus: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Every ``(a, b)`` pair and its ``(a + b) mod p`` label."""
    a = torch.arange(modulus).repeat_interleave(modulus)
    b = torch.arange(modulus).repeat(modulus)
    labels = (a + b) % modulus
    return a, b, labels


class GrokkingRun:
    """Owns one grokking training run and its recorded history."""

    def __init__(self, cfg: GrokkingConfig, logger: RunLogger | None = None) -> None:
        self.cfg = cfg
        self.logger = logger
        seed_everything(cfg.seed)
        self.device = torch.device(cfg.device)
        self.model = ModularAdditionTransformer(cfg.modulus, cfg.d_model, cfg.n_heads).to(
            self.device)
        a, b, labels = _all_pairs(cfg.modulus)
        tokens = self.model.input_tokens(a, b).to(self.device)
        labels = labels.to(self.device)
        # Fixed random train/val split of the full dataset.
        rng = np.random.default_rng(cfg.seed)
        perm = rng.permutation(len(labels))
        n_train = int(len(labels) * cfg.train_frac)
        self._tr = perm[:n_train]
        self._va = perm[n_train:]
        self._tokens, self._labels = tokens, labels

    @torch.no_grad()
    def _accuracy(self, idx: np.ndarray) -> float:
        self.model.eval()
        logits = self.model(self._tokens[idx])
        return float((logits.argmax(-1) == self._labels[idx]).float().mean())

    def _weight_norm(self) -> float:
        """Description-length proxy: total parameter L2 norm."""
        return float(torch.sqrt(sum((p.detach() ** 2).sum() for p in self.model.parameters())))

    def run(self) -> dict[str, Any]:
        """Train for ``num_steps``, recording the grokking curve; return the history."""
        cfg = self.cfg
        opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.learning_rate,
                                weight_decay=cfg.weight_decay, betas=(0.9, 0.98))
        loss_fn = nn.CrossEntropyLoss()
        rng = np.random.default_rng(cfg.seed + 1)
        history: list[dict[str, float]] = []
        if self.logger is not None:
            self.logger.log("run_start", modulus=cfg.modulus, train_frac=cfg.train_frac,
                            d_model=cfg.d_model, weight_decay=cfg.weight_decay,
                            num_steps=cfg.num_steps, n_train=len(self._tr))
        for step in range(1, cfg.num_steps + 1):
            self.model.train()
            if cfg.batch_size and cfg.batch_size > 0 and cfg.batch_size < len(self._tr):
                batch = rng.choice(self._tr, size=cfg.batch_size, replace=False)
            else:
                batch = self._tr  # full batch
            opt.zero_grad()
            loss = loss_fn(self.model(self._tokens[batch]), self._labels[batch])
            loss.backward()
            opt.step()
            if step % cfg.eval_every == 0 or step == cfg.num_steps:
                rec = {"step": step, "train_acc": self._accuracy(self._tr),
                       "val_acc": self._accuracy(self._va), "weight_norm": self._weight_norm(),
                       "train_loss": float(loss.detach())}
                history.append(rec)
                if self.logger is not None:
                    self.logger.log("eval", **rec)

        summary = {
            "modulus": cfg.modulus,
            "final_train_acc": history[-1]["train_acc"],
            "final_val_acc": history[-1]["val_acc"],
            "grokked": history[-1]["val_acc"] >= 0.9,
            "history": history,
        }
        if self.logger is not None:
            self.logger.log("run_end", **{k: v for k, v in summary.items() if k != "history"})
        return summary
