# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""Credential loading for rich-mode agents -- re-exported from `agentharness`.

Extracted into the standalone, Apache-2.0 `agentharness` package
(https://github.com/stevejohnson15/agentharness); re-exported here so existing
`beetlebox.secrets` imports keep working. Never logs or prints the key.
"""

from __future__ import annotations

from agentharness.secrets import (
    KEY_FILENAME,
    _parse_key_file,
    find_key_file,
    load_anthropic_key,
)

# `_parse_key_file` is re-exported (not just public API) so back-compat callers
# and tests that referenced `beetlebox.secrets._parse_key_file` keep working.
__all__ = ["KEY_FILENAME", "_parse_key_file", "find_key_file", "load_anthropic_key"]
