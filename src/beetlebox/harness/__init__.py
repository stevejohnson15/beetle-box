# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""Multi-agent orchestration harness.

Turn management and run orchestration for Beetle-Box experiments. Depends only
on the library's own dataclass configs (not Hydra), an extraction candidate for
a standalone multi-agent orchestration package.
"""

from beetlebox.harness.e2_manager import E2RunManager
from beetlebox.harness.e3_manager import E3RunManager
from beetlebox.harness.e4_manager import E4RunManager
from beetlebox.harness.run_manager import RunManager

__all__ = ["RunManager", "E2RunManager", "E3RunManager", "E4RunManager"]
