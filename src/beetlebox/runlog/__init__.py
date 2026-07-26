# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""Structured, replayable run logging -- re-exported from the `reprolog` library.

This module was extracted into the standalone, Apache-2.0 `reprolog` package
(https://github.com/stevejohnson15/reprolog). Beetle-Box now consumes it and
re-exports the same names here so existing `beetlebox.runlog` imports keep working.
"""

from __future__ import annotations

from reprolog import (
    EVENTS_FILE,
    MANIFEST_FILE,
    RunLogger,
    iter_events,
    read_manifest,
    run_dir,
)

__all__ = [
    "EVENTS_FILE",
    "MANIFEST_FILE",
    "RunLogger",
    "iter_events",
    "read_manifest",
    "run_dir",
]
