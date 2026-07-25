# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
import pytest

from beetlebox.channels import SymbolChannel
from beetlebox.config import ChannelConfig


def test_bandwidth():
    assert SymbolChannel(8, 1).bandwidth == 8
    assert SymbolChannel(6, 2).bandwidth == 36


def test_from_config():
    ch = SymbolChannel.from_config(ChannelConfig(vocab_size=4, message_length=3))
    assert ch.vocab_size == 4 and ch.message_length == 3


def test_contains():
    ch = SymbolChannel(5, 2)
    assert ch.contains([0, 4])
    assert not ch.contains([0, 5])  # out of vocab
    assert not ch.contains([1])  # wrong length


def test_message_distance():
    ch = SymbolChannel(5, 3)
    assert ch.message_distance([1, 2, 3], [1, 2, 3]) == 0
    assert ch.message_distance([1, 2, 3], [1, 0, 4]) == 2


def test_invalid_construction():
    with pytest.raises(ValueError):
        SymbolChannel(1, 1)
    with pytest.raises(ValueError):
        SymbolChannel(4, 0)


def test_render_is_not_english():
    assert SymbolChannel(8, 2).render([3, 1]) == "s3-s1"
