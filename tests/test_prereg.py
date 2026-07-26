# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""Tests for the frozen-prereg loader (guards against scoring on a draft)."""

import pytest

from beetlebox.analysis.prereg import load_prereg


def test_load_frozen_prereg_ok():
    data = load_prereg("prereg/e1_signaling.yaml")
    assert data["frozen"] is True
    assert "thresholds" in data


def test_missing_prereg_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_prereg(tmp_path / "nope.yaml")


def test_unfrozen_prereg_rejected(tmp_path):
    p = tmp_path / "draft.yaml"
    p.write_text("frozen: false\nthresholds: {}\n")
    with pytest.raises(ValueError):
        load_prereg(p)
