# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""Constrained communication channels.

For E1 the channel is a :class:`SymbolChannel`: an *invented*, non-natural-
language vocabulary of ``V`` integer symbols carried in fixed-length messages of
length ``L``. Using symbols with no English meaning is the clean-room guarantee
(``plan/beetle-box.md`` §3.2) -- there is no inherited meaning to smuggle in.

This module depends only on NumPy and is an extraction candidate for a standalone
emergent-communication toolkit.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from beetlebox.config import ChannelConfig


class SymbolChannel:
    """A discrete symbol channel with a fixed invented vocabulary."""

    def __init__(self, vocab_size: int, message_length: int) -> None:
        if vocab_size < 2:
            raise ValueError("vocab_size must be >= 2")
        if message_length < 1:
            raise ValueError("message_length must be >= 1")
        self.vocab_size = vocab_size
        self.message_length = message_length

    @classmethod
    def from_config(cls, cfg: ChannelConfig) -> SymbolChannel:
        return cls(cfg.vocab_size, cfg.message_length)

    @property
    def bandwidth(self) -> int:
        """Number of distinct messages expressible: ``V ** L``."""
        return self.vocab_size**self.message_length

    def contains(self, message: Sequence[int]) -> bool:
        """True iff ``message`` is a valid message on this channel."""
        msg = list(message)
        return len(msg) == self.message_length and all(
            isinstance(s, (int, np.integer)) and 0 <= int(s) < self.vocab_size for s in msg
        )

    def message_distance(self, a: Sequence[int], b: Sequence[int]) -> int:
        """Hamming distance between two messages (used for topographic similarity)."""
        a, b = list(a), list(b)
        if len(a) != self.message_length or len(b) != self.message_length:
            raise ValueError("messages must have length == message_length")
        return int(sum(1 for x, y in zip(a, b, strict=True) if int(x) != int(y)))

    def render(self, message: Sequence[int]) -> str:
        """Human-readable form for logs, e.g. ``s3-s1`` -- invented, not English."""
        return "-".join(f"s{int(s)}" for s in message)
