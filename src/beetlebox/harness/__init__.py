# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""Multi-agent orchestration harness.

Turn management and run orchestration for Beetle-Box experiments. Depends only
on the library's own dataclass configs (not Hydra), an extraction candidate for
a standalone multi-agent orchestration package.
"""

from beetlebox.harness.run_manager import RunManager

__all__ = ["RunManager"]
