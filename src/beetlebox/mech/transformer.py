# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""A small from-scratch transformer for modular addition (E4 mechanistic layer).

This is the grokking model of Power et al. (2022) / Nanda et al. (2023): a
one-layer attention-plus-MLP transformer trained on ``(a, b) -> (a + b) mod p``.
Trained with weight decay it first *memorizes* the training pairs and only much
later *generalizes* — the "grokking" transition — at which point it has learned a
crisp, readable circuit (a Fourier / "Clock" construction on the residual stream).

The sequence is ``[a, b, =]`` (length 3); the model predicts the answer class at
the final position over ``p`` classes. LayerNorm is deliberately omitted so the
learned representations stay linear and legible for the Fourier read-out in
:mod:`beetlebox.mech.circuits`.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ModularAdditionTransformer(nn.Module):
    """One-layer transformer over ``[a, b, =]`` predicting ``(a + b) mod p``."""

    def __init__(self, modulus: int, d_model: int = 128, n_heads: int = 4) -> None:
        super().__init__()
        self.modulus = modulus
        self.d_model = d_model
        self.n_heads = n_heads
        self.seq_len = 3  # a, b, =
        self.vocab = modulus + 1  # tokens 0..p-1 plus the '=' token at index p
        self.embed = nn.Embedding(self.vocab, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(self.seq_len, d_model))
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.unembed = nn.Linear(d_model, modulus, bias=False)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """``tokens`` is ``[B, 3]`` of ids; returns answer logits ``[B, modulus]``."""
        x = self.embed(tokens) + self.pos_embed[None, : tokens.shape[1], :]
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(x)
        return self.unembed(x[:, -1, :])  # read the answer at the '=' position

    def input_tokens(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Build the ``[B, 3]`` token batch ``[a, b, =]`` for operand tensors."""
        eq = torch.full_like(a, self.modulus)  # the '=' token id
        return torch.stack([a, b, eq], dim=1)
