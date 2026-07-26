# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""Tests for the frontier (Anthropic API) agent backend.

No network is used: a fake ``anthropic.Anthropic`` client is injected so we can
verify the request shape (structured-output schema), response parsing, and token
accounting without making a live call.
"""

import sys
import types

import pytest


class _FakeUsage:
    def __init__(self, i=10, o=2, cr=0, cc=0):
        self.input_tokens = i
        self.output_tokens = o
        self.cache_read_input_tokens = cr
        self.cache_creation_input_tokens = cc


class _FakeBlock:
    type = "text"

    def __init__(self, text):
        self.text = text


class _FakeResponse:
    def __init__(self, text):
        self.content = [_FakeBlock(text)]
        self.usage = _FakeUsage()


class _FakeMessages:
    def __init__(self, recorder):
        self._rec = recorder

    def create(self, **kwargs):
        self._rec.append(kwargs)
        # Echo back a valid choice from the schema's enum (pick the first).
        enum = kwargs["output_config"]["format"]["schema"]["properties"]["choice"]["enum"]
        return _FakeResponse(f'{{"choice": {enum[0]}}}')


class _FakeClient:
    def __init__(self, recorder):
        self.messages = _FakeMessages(recorder)


@pytest.fixture
def fake_anthropic(monkeypatch):
    """Inject a fake ``anthropic`` module and a dummy key."""
    recorder = []
    fake_mod = types.ModuleType("anthropic")
    fake_mod.Anthropic = lambda api_key=None: _FakeClient(recorder)
    monkeypatch.setitem(sys.modules, "anthropic", fake_mod)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    return recorder


def test_choose_returns_valid_enum_choice(fake_anthropic):
    from beetlebox.agents.api_model import ApiAgent

    agent = ApiAgent(name="t", model="claude-haiku-4-5")
    choice = agent.choose("sys", "user", [3, 4, 5])
    assert choice == 3  # fake echoes the first enum value


def test_choose_builds_structured_output_schema(fake_anthropic):
    from beetlebox.agents.api_model import ApiAgent

    agent = ApiAgent(name="t")
    agent.choose("system prompt", "user prompt", [0, 1, 2])
    req = fake_anthropic[-1]
    schema = req["output_config"]["format"]["schema"]
    assert schema["properties"]["choice"]["enum"] == [0, 1, 2]
    assert schema["additionalProperties"] is False
    assert req["system"] == "system prompt"


def test_usage_and_call_count_accumulate(fake_anthropic):
    from beetlebox.agents.api_model import ApiAgent

    agent = ApiAgent(name="t")
    agent.choose("s", "u", [0, 1])
    agent.choose("s", "u", [0, 1])
    assert agent.num_calls == 2
    assert agent.usage["input_tokens"] == 20  # 2 calls x 10
    assert agent.usage["output_tokens"] == 4


def test_reset_parameters_is_noop(fake_anthropic):
    from beetlebox.agents.api_model import ApiAgent

    ApiAgent(name="t").reset_parameters()  # frontier agents have no trainable state


def test_default_model_is_opus(fake_anthropic):
    from beetlebox.agents.api_model import DEFAULT_MODEL

    assert DEFAULT_MODEL == "claude-opus-4-8"
