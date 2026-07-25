# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""Frontier (Anthropic API) agent backend -- rich, confounded mode.

This is the "rich-but-confounded" arm of the dual-mode design (plan §3.2):
frontier models arrive pretrained on oceans of human language, so results here
study how inherited concepts are *redeployed* under novel pressure -- they must
never be read as clean-room results.

The agent makes a single constrained decision per call, using structured outputs
so the reply is a guaranteed-valid choice from a fixed set. Model defaults to
``claude-opus-4-8`` (override for cost: ``claude-haiku-4-5`` / ``claude-sonnet-5``).
Rich-mode runs cost money -- keep them small.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from beetlebox.agents.base import Agent
from beetlebox.secrets import load_anthropic_key

DEFAULT_MODEL = "claude-opus-4-8"


class ApiAgent(Agent):
    """A frontier-model participant that returns a constrained integer choice."""

    def __init__(self, *, name: str = "api_agent", model: str = DEFAULT_MODEL,
                 max_tokens: int = 512, api_key: str | None = None) -> None:
        super().__init__(name=name)
        # Import lazily so the package imports without the anthropic SDK installed.
        import anthropic

        self.model = model
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic(api_key=api_key or load_anthropic_key())

    def choose(self, system: str, user: str, choices: Sequence[int]) -> int:
        """Return one integer from ``choices`` (structured-output constrained)."""
        allowed = [int(c) for c in choices]
        schema = {
            "type": "object",
            "properties": {"choice": {"type": "integer", "enum": allowed}},
            "required": ["choice"],
            "additionalProperties": False,
        }
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
        text = next((b.text for b in response.content if b.type == "text"), "")
        return int(json.loads(text)["choice"])

    def reset_parameters(self) -> None:  # frontier agents have no trainable state
        return None
