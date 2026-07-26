# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""Emergent-communication metrics -- re-exported from the `emcomkit` library.

The compositionality / convention metrics (topographic similarity, convention
stability, transmission fidelity) were extracted into the standalone, Apache-2.0
``emcomkit`` package (https://github.com/stevejohnson15/emcomkit); re-exported
here so existing ``beetlebox.analysis.metrics`` imports keep working. Thresholds
are never applied here -- they live in the frozen pre-registration and are applied
only in :mod:`beetlebox.analysis.e1`.
"""

from __future__ import annotations

from emcomkit.metrics import (
    _mapping_agreement,
    convention_stability,
    topographic_similarity,
    transmission_fidelity,
)

__all__ = [
    # `_mapping_agreement` is re-exported so back-compat callers/tests that
    # referenced `beetlebox.analysis.metrics._mapping_agreement` keep working.
    "_mapping_agreement",
    "convention_stability",
    "topographic_similarity",
    "transmission_fidelity",
]
