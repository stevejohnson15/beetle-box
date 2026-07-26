# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""Frontier (Anthropic API) agent backend -- re-exported from `agentharness`.

The ``ApiAgent`` frontier backend was extracted into the standalone, Apache-2.0
``agentharness`` package (https://github.com/stevejohnson15/agentharness), where
it is registered as the ``"anthropic"`` backend. Re-exported here so existing
``beetlebox.agents.api_model`` imports keep working. The ``anthropic`` SDK is
imported lazily, so this module imports without it installed.

This is the "rich-but-confounded" arm of the dual-mode design: frontier models
arrive pretrained on human language, so results here study how inherited concepts
are *redeployed* under novel pressure -- never read them as clean-room results.
"""

from __future__ import annotations

from agentharness.backends.anthropic import DEFAULT_MODEL, ApiAgent

__all__ = ["DEFAULT_MODEL", "ApiAgent"]
