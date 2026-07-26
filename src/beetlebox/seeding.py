# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""Deterministic seeding -- re-exported from the `reprolog` library.

Extracted into the standalone, Apache-2.0 `reprolog` package
(https://github.com/stevejohnson15/reprolog); re-exported here so existing
`beetlebox.seeding` imports keep working.
"""

from __future__ import annotations

from reprolog import seed_everything

__all__ = ["seed_everything"]
