# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""Circuit read-out for the grokked modular-addition transformer (E4).

A network that has grokked modular addition implements it with a **Fourier
construction**: the token embeddings become sinusoidal in the operand value at a
small set of key frequencies, and addition is done by rotating those phases
(Nanda et al.'s "Clock"). We read the circuit out of the weights directly:

- :func:`fourier_power_spectrum` — the power at each frequency in the numeric-token
  embedding matrix. A grokked model shows a few sharp spikes; an un-grokked one is
  diffuse.
- :func:`key_frequencies` — the dominant frequencies (the model's "gears").
- :func:`algorithmic_agreement` — the overlap between two models' key-frequency
  sets. This is the "Clock and the Pizza" question made quantitative: two models
  trained on *identical* data need not pick the same frequencies, so "which
  algorithm" can lack a determinate cross-seed answer even at the mechanism level.

NumPy/torch only; no interpretability framework required.
"""

from __future__ import annotations

import numpy as np
import torch

from beetlebox.mech.transformer import ModularAdditionTransformer


def _numeric_embedding(model: ModularAdditionTransformer) -> np.ndarray:
    """The embedding rows for the numeric tokens ``0..p-1`` (drop the '=' token)."""
    return model.embed.weight.detach().cpu().numpy()[: model.modulus]


def fourier_power_spectrum(model: ModularAdditionTransformer) -> np.ndarray:
    """Power at each frequency ``1..p//2`` in the numeric-token embedding.

    For each embedding dimension we take the DFT along the token axis and sum the
    squared magnitudes across dimensions, then normalize to sum to 1.
    """
    emb = _numeric_embedding(model)  # [p, d_model]
    emb = emb - emb.mean(axis=0, keepdims=True)  # drop the DC (constant) component
    fft = np.fft.rfft(emb, axis=0)  # [p//2 + 1, d_model]
    power = (np.abs(fft) ** 2).sum(axis=1)  # [p//2 + 1]
    power = power[1:]  # drop frequency 0 (removed above)
    total = power.sum()
    return power / total if total > 0 else power


def key_frequencies(model: ModularAdditionTransformer, top_k: int = 5,
                    min_power: float = 0.05) -> list[int]:
    """The dominant frequencies (1-indexed) in the embedding, strongest first.

    Returns at most ``top_k`` frequencies whose normalized power exceeds
    ``min_power`` — the small set of "gears" the grokked circuit turns on.
    """
    power = fourier_power_spectrum(model)
    order = np.argsort(power)[::-1]
    return [int(f + 1) for f in order[:top_k] if power[f] >= min_power]


def algorithmic_agreement(model_a: ModularAdditionTransformer,
                          model_b: ModularAdditionTransformer, top_k: int = 5) -> float:
    """Jaccard overlap of two models' key-frequency sets (0 = disjoint, 1 = identical).

    The "Clock and the Pizza" metric: how much do two models trained on identical
    data agree on *which* algorithm (which frequencies) they implement?
    """
    fa = set(key_frequencies(model_a, top_k=top_k))
    fb = set(key_frequencies(model_b, top_k=top_k))
    if not fa and not fb:
        return 1.0
    return len(fa & fb) / len(fa | fb)


def concentration(model: ModularAdditionTransformer, top_k: int = 5) -> float:
    """Fraction of embedding Fourier power in the top-``k`` frequencies.

    A proxy for "how grokked / how clean" the circuit is: memorizing models spread
    power across many frequencies (low concentration); grokked models concentrate it
    in a few (high).
    """
    power = fourier_power_spectrum(model)
    if power.sum() == 0:
        return 0.0
    return float(np.sort(power)[::-1][:top_k].sum())


def load_model(state_path: str, modulus: int, d_model: int = 128,
               n_heads: int = 4) -> ModularAdditionTransformer:
    """Rebuild a transformer and load weights saved with ``torch.save(model.state_dict())``."""
    model = ModularAdditionTransformer(modulus, d_model, n_heads)
    model.load_state_dict(torch.load(state_path, map_location="cpu"))
    model.eval()
    return model
