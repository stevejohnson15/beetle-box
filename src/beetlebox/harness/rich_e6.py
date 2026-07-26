# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""E6: the reflexive layer -- observer folded into observed (plan §1, §4.6).

The meta-example of §1 made into a condition. Two frontier agents build a shared
convention in a repeated reference game (a coordinating pair, as in E1's rich mode),
and then are **turned to examine their own coordination** -- prompted to consider
whether they and their partner genuinely share the meanings of their invented
symbols, or whether each is merely privately guessing (the §293 question, in the
first person). The question E6 asks is the plan's exact one:

    Does self-examination *change the coordination*, or merely add a plausible story
    on top of it?

**This is the most interpretively dangerous experiment** (plan §4.6): the reflection
transcripts are maximally seductive -- an agent saying "yes, we understand each
other" is not evidence of anything. So the design is built to resist over-reading.
The measured quantity is **behavioral**: the change in coordination accuracy from
the pre-intervention block to the post-intervention block (``delta``), compared
against a matched ``control`` interlude that adds an equal amount of context with no
self-examination. The reflection text is recorded for the record only and is **never
scored** (:mod:`beetlebox.analysis.e6`).

Interventions:
  reflect  - each agent examines whether it and its partner share meaning
  control  - each agent answers a matched, task-irrelevant prompt (placebo)
  none     - no interlude

The agents are injectable, so the design and metric are unit-tested without any API
call. A real run makes ~``4 * rounds_per_block + 2`` API calls; keep blocks small.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from beetlebox.config import E6Config
from beetlebox.runlog import RunLogger


class _ReflexiveAgent(Protocol):
    """The agent interface E6 needs: a constrained choice and a free-text reply."""

    usage: dict[str, int]
    num_calls: int

    def choose(self, system: str, user: str, choices: Any) -> int: ...

    def respond(self, system: str, user: str) -> str: ...


class RichReflexiveRunner:
    """Two frontier agents coordinate, undergo an intervention, then coordinate again."""

    def __init__(self, cfg: E6Config, *, logger: RunLogger | None = None,
                 sender: _ReflexiveAgent | None = None,
                 receiver: _ReflexiveAgent | None = None) -> None:
        if cfg.intervention not in ("reflect", "control", "none"):
            raise ValueError(f"unknown intervention: {cfg.intervention!r}")
        self.cfg = cfg
        self.logger = logger
        self.rng = np.random.default_rng(cfg.seed)
        if sender is None or receiver is None:
            from beetlebox.agents.api_model import DEFAULT_MODEL, ApiAgent
            model = cfg.model or DEFAULT_MODEL
            sender = sender or ApiAgent(name="sender", model=model)
            receiver = receiver or ApiAgent(name="receiver", model=model)
        self.sender, self.receiver = sender, receiver
        self._sender_log: list[str] = []
        self._receiver_log: list[str] = []

    # -- coordination game ---------------------------------------------- #
    def _sender_system(self) -> str:
        return ("You and a partner are inventing a shared code. Each round you see a "
                f"hidden object (0..{self.cfg.num_referents - 1}) and send one integer "
                f"symbol (0..{self.cfg.vocab_size - 1}). Your partner tries to identify the "
                "object from your symbol. You get feedback; be consistent so they succeed.")

    def _receiver_system(self) -> str:
        return ("You and a partner share an invented code. Your partner saw a hidden object "
                f"(0..{self.cfg.num_referents - 1}) and sent one integer symbol "
                f"(0..{self.cfg.vocab_size - 1}). Identify the object. You get feedback each "
                "round; use it to learn your partner's code.")

    def _play_block(self, n_rounds: int) -> float:
        """Play ``n_rounds`` of the reference game; return coordination accuracy."""
        correct = 0
        for _ in range(n_rounds):
            r = int(self.rng.integers(0, self.cfg.num_referents))
            s_user = ("History:\n" + ("\n".join(self._sender_log) or "(none)") +
                      f"\n\nThis round the hidden object is {r}. "
                      f"Choose a symbol (0..{self.cfg.vocab_size - 1}).")
            symbol = self.sender.choose(self._sender_system(), s_user, range(self.cfg.vocab_size))
            r_user = ("History:\n" + ("\n".join(self._receiver_log) or "(none)") +
                      f"\n\nYour partner sent symbol {symbol}. "
                      f"Which object (0..{self.cfg.num_referents - 1})?")
            guess = self.receiver.choose(self._receiver_system(),
                                         r_user, range(self.cfg.num_referents))
            hit = guess == r
            correct += int(hit)
            self._sender_log.append(
                f"object={r} -> symbol={symbol}; partner guessed {guess}; correct={hit}")
            self._receiver_log.append(
                f"symbol={symbol}; guessed {guess}; true object={r}; correct={hit}")
        return correct / n_rounds

    # -- intervention --------------------------------------------------- #
    _REFLECT = ("Step back from the game. Do you and your partner genuinely share the "
                "meanings of these symbols -- is there a common understanding between you -- "
                "or is each of you just privately guessing at patterns with no shared inner "
                "grasp? Answer honestly in a few sentences.")
    _CONTROL = ("Take a brief pause from the game. In a few sentences, describe a calm "
                "natural landscape you find pleasant. This is unrelated to the task.")

    def _intervene(self) -> dict[str, str]:
        """Run the intervention turn; append each agent's reply to its own history."""
        if self.cfg.intervention == "none":
            return {}
        prompt = self._REFLECT if self.cfg.intervention == "reflect" else self._CONTROL
        s_text = self.sender.respond(self._sender_system(), prompt)
        r_text = self.receiver.respond(self._receiver_system(), prompt)
        # Fold the reply into each agent's context so it COULD influence later rounds.
        self._sender_log.append(f"[reflection] {s_text}")
        self._receiver_log.append(f"[reflection] {r_text}")
        return {"sender": s_text, "receiver": r_text}

    # -- run ------------------------------------------------------------ #
    def run(self) -> dict[str, Any]:
        """Pre block -> intervention -> post block; return the behavioral metrics."""
        cfg = self.cfg
        if self.logger is not None:
            self.logger.log("rich_run_start", experiment="e6_reflexive",
                            intervention=cfg.intervention, rounds_per_block=cfg.rounds_per_block,
                            num_referents=cfg.num_referents, model=cfg.model)
        pre = self._play_block(cfg.rounds_per_block)
        transcript = self._intervene()  # recorded, never scored
        post = self._play_block(cfg.rounds_per_block)

        usage = {k: self.sender.usage[k] + self.receiver.usage[k] for k in self.sender.usage}
        summary = {
            "experiment": "e6_reflexive", "mode": "rich", "intervention": cfg.intervention,
            "pre_accuracy": pre, "post_accuracy": post, "delta": post - pre,
            "chance": 1.0 / cfg.num_referents, "rounds_per_block": cfg.rounds_per_block,
            "api_calls": self.sender.num_calls + self.receiver.num_calls, "usage": usage,
            "reflection_transcript": transcript,  # for the record only; NOT an input to scoring
        }
        if self.logger is not None:
            self.logger.log("run_end", **{k: v for k, v in summary.items()
                                           if k != "reflection_transcript"})
            # Persist transcripts separately so it is structurally clear they are not scored.
            if transcript:
                self.logger.log("reflection_transcript", **transcript)
        return summary
