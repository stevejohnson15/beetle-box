# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""Structured, replayable run logging keyed by config hash + seed.

Every run writes an append-only ``events.jsonl`` stream plus a ``manifest.json``
capturing the full config, seed, and environment, so any run is reconstructable
after the fact. Analysis reads these logs; execution and scoring never mix (see
``plan/beetle-box.md`` §5). Depends only on the standard library -- an
extraction candidate for a standalone reproducible-logging package.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

EVENTS_FILE = "events.jsonl"
MANIFEST_FILE = "manifest.json"


def run_dir(output_dir: str | Path, config_hash: str, seed: int) -> Path:
    """Canonical results path: ``<output_dir>/<config_hash>/seed<seed>``."""
    return Path(output_dir) / config_hash / f"seed{seed}"


class RunLogger:
    """Append-only JSONL event logger for a single run."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._events_path = self.directory / EVENTS_FILE
        # Truncate any prior stream: a run owns its directory.
        self._fh = self._events_path.open("w", encoding="utf-8")

    def log(self, event_type: str, *, step: int | None = None, **fields: Any) -> None:
        """Write one structured event. ``wall_time`` is metadata, not a metric."""
        record: dict[str, Any] = {"event": event_type, "wall_time": time.time()}
        if step is not None:
            record["step"] = step
        record.update(fields)
        self._fh.write(json.dumps(record, sort_keys=True) + "\n")
        self._fh.flush()

    def write_manifest(self, config: dict[str, Any], *, seed: int, config_hash: str,
                       extra: dict[str, Any] | None = None) -> None:
        """Persist the reproduction manifest for this run."""
        manifest = {
            "config": config,
            "seed": seed,
            "config_hash": config_hash,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        }
        if extra:
            manifest.update(extra)
        (self.directory / MANIFEST_FILE).write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )

    def close(self) -> None:
        """Close the events file handle (idempotent)."""
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> RunLogger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def iter_events(directory: str | Path) -> Iterator[dict[str, Any]]:
    """Yield events from a run's ``events.jsonl`` in order."""
    path = Path(directory) / EVENTS_FILE
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_manifest(directory: str | Path) -> dict[str, Any]:
    """Load a run's ``manifest.json``."""
    return json.loads((Path(directory) / MANIFEST_FILE).read_text(encoding="utf-8"))
