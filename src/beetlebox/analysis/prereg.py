# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""Loader for frozen pre-registrations -- re-exported from `frozenprereg`.

The frozen-prereg loader (which refuses to score against a prereg not marked
``frozen: true``) was extracted into the standalone, Apache-2.0 ``frozenprereg``
package (https://github.com/stevejohnson15/frozenprereg); re-exported here so
existing ``beetlebox.analysis.prereg`` imports keep working. ``DEFAULT_PREREG``
is retained for back-compat (E1's default prereg path).
"""

from __future__ import annotations

from frozenprereg import PreregError, load_prereg

DEFAULT_PREREG = "prereg/e1_signaling.yaml"

__all__ = ["DEFAULT_PREREG", "PreregError", "load_prereg"]
