# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""Scoring and analysis.

Analysis is separated from execution and imports *only* frozen pre-registered
criteria, so scoring cannot be tuned to a result after the fact
(``plan/beetle-box.md`` §5). An extraction candidate for a reusable
pre-registration/scoring framework.
"""
