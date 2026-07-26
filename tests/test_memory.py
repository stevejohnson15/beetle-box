# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
from beetlebox.memory import EpisodicMemory, NoMemory, make_memory


def test_no_memory_forgets_everything():
    m = NoMemory()
    assert m.enabled is False
    m.write({"x": 1})
    assert m.read() == []
    assert len(m) == 0


def test_episodic_memory_retains_order():
    m = EpisodicMemory()
    assert m.enabled is True
    for i in range(3):
        m.write({"i": i})
    assert [r["i"] for r in m.read()] == [0, 1, 2]
    assert [r["i"] for r in m.read(last=2)] == [1, 2]
    assert len(m) == 3
    m.clear()
    assert len(m) == 0


def test_episodic_capacity_is_a_ring_buffer():
    m = EpisodicMemory(capacity=2)
    for i in range(5):
        m.write({"i": i})
    assert [r["i"] for r in m.read()] == [3, 4]


def test_factory_toggles_implementation():
    assert isinstance(make_memory(False), NoMemory)
    assert isinstance(make_memory(True), EpisodicMemory)


def test_episodic_memory_rejects_bad_capacity():
    import pytest
    with pytest.raises(ValueError):
        EpisodicMemory(capacity=0)
