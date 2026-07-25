# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""Toggleable per-agent memory.

Memory access is an *independent variable*, not an implementation detail
(``plan/beetle-box.md`` §5, and centrally E2's "is context-memory an independent
check, or a longer private impression?"). The interface is therefore explicit
and switchable: :class:`NoMemory` models the memory-disabled condition and
:class:`EpisodicMemory` the memory-enabled one. E1 does not require memory; the
machinery is defined and unit-tested here so E2 can exercise it as a first-class
condition without rework.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import Any


class Memory(ABC):
    """Abstract per-agent memory."""

    #: Whether this memory can retain anything (the experimental toggle).
    enabled: bool

    @abstractmethod
    def write(self, record: dict[str, Any]) -> None:
        """Store one record."""

    @abstractmethod
    def read(self, last: int | None = None) -> list[dict[str, Any]]:
        """Return stored records, most-recent last; optionally only the last ``n``."""

    @abstractmethod
    def clear(self) -> None:
        """Forget everything."""

    @abstractmethod
    def __len__(self) -> int: ...


class NoMemory(Memory):
    """The memory-disabled condition: writes vanish, reads are always empty."""

    enabled = False

    def write(self, record: dict[str, Any]) -> None:  # noqa: D401 - intentional no-op
        return None

    def read(self, last: int | None = None) -> list[dict[str, Any]]:
        return []

    def clear(self) -> None:
        return None

    def __len__(self) -> int:
        return 0


class EpisodicMemory(Memory):
    """The memory-enabled condition: an ordered, optionally bounded episode log."""

    enabled = True

    def __init__(self, capacity: int | None = None) -> None:
        if capacity is not None and capacity < 1:
            raise ValueError("capacity must be >= 1 or None (unbounded)")
        self.capacity = capacity
        self._records: deque[dict[str, Any]] = deque(maxlen=capacity)

    def write(self, record: dict[str, Any]) -> None:
        self._records.append(dict(record))

    def read(self, last: int | None = None) -> list[dict[str, Any]]:
        records = list(self._records)
        if last is not None:
            return records[-last:]
        return records

    def clear(self) -> None:
        self._records.clear()

    def __len__(self) -> int:
        return len(self._records)


def make_memory(enabled: bool, capacity: int | None = None) -> Memory:
    """Factory: return the memory implementation for the toggle state."""
    return EpisodicMemory(capacity=capacity) if enabled else NoMemory()
