# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""Constrained communication channels -- re-exported from the `emcomkit` library.

The invented, non-natural-language `SymbolChannel` was extracted into the
standalone, Apache-2.0 `emcomkit` package
(https://github.com/stevejohnson15/emcomkit); re-exported here so existing
`beetlebox.channels` imports keep working. ``SymbolChannel.from_config`` accepts
any object exposing ``vocab_size`` / ``message_length`` (e.g. a ``ChannelConfig``).
"""

from __future__ import annotations

from emcomkit.channels import SymbolChannel

__all__ = ["SymbolChannel"]
