# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""Rich-mode E3: the private-referent beetle-box with frontier agents.

The clean-room E3 (:mod:`beetlebox.harness.e3_manager`) trains small nets from
scratch. This is the rich-but-confounded arm (plan §3.2): two frontier models
coordinate *in context* over rounds with feedback -- studying how pretrained
agents redeploy inherited concepts, never as a clean-room claim.

Each round: God's-eye picks a hidden object; the sender's private "box" is its
ONLY access to that object (a nonsense sensation code); the sender emits one
public invented symbol; the receiver identifies the object from the symbol (and,
under shared/divergent, its own box). Feedback is appended so a convention can
form in context.

Box conditions mirror the clean-room design:
  shared    - sender and receiver get the SAME sensation code for the object
  divergent - each gets its own private code for the object
  empty     - no sensation code (sender is blind -> coordination floor)
  noise     - a random code, unrelated to the object

**Cost warning.** This makes ~``2 * num_rounds`` API calls per run. Defaults are
deliberately small. Scale up only deliberately.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from beetlebox.agents.api_model import DEFAULT_MODEL, ApiAgent
from beetlebox.boxes import BoxCondition
from beetlebox.runlog import RunLogger

# Invented, non-English sensation codes (the private "beetle" tokens).
_NONSENSE = [
    "glim", "vok", "tuz", "plef", "murn", "zib", "quon", "fash",
    "drell", "wix", "nalb", "sprock", "yomp", "krin", "blul", "thaz",
]


class RichPrivateReferentRunner:
    """In-context private-referent beetle-box played by two frontier agents."""

    def __init__(self, condition: str, *, num_referents: int = 4, vocab_size: int = 6,
                 num_rounds: int = 12, model: str = DEFAULT_MODEL, seed: int = 0,
                 logger: RunLogger | None = None) -> None:
        self.condition = BoxCondition(condition)
        self.k = num_referents
        self.v = vocab_size
        self.num_rounds = num_rounds
        self.logger = logger
        self.rng = np.random.default_rng(seed)
        self.sender = ApiAgent(name="sender", model=model)
        self.receiver = ApiAgent(name="receiver", model=model)
        # Fixed per-agent sensation codes for the informative conditions.
        self._code_s = list(_NONSENSE[: self.k])
        if self.condition == BoxCondition.SHARED:
            self._code_r = list(self._code_s)
        else:  # divergent uses a disjoint second block; unused for empty/noise
            self._code_r = list(_NONSENSE[self.k : 2 * self.k])

    # -- per-agent private sensation for object r ------------------------- #
    def _sensation_sender(self, r: int) -> str:
        if self.condition == BoxCondition.EMPTY:
            return "(no sensation available)"
        if self.condition == BoxCondition.NOISE:
            return str(self.rng.choice(_NONSENSE))
        return self._code_s[r]

    def _sensation_receiver(self, r: int) -> str | None:
        if self.condition in (BoxCondition.SHARED, BoxCondition.DIVERGENT):
            return self._code_r[r]
        return None  # empty / noise: receiver has no informative box

    # -- prompts --------------------------------------------------------- #
    def _sender_system(self) -> str:
        return (
            "You are Agent S in a two-player coordination game. Each round you privately "
            f"sense a stimulus, shown as a nonsense code. You must send ONE public signal: "
            f"an integer symbol from 0 to {self.v - 1}. Your partner, Agent R, will try to "
            f"identify which of {self.k} hidden objects (0..{self.k - 1}) you sensed, using "
            "only your signal. You receive feedback each round. Build a consistent mapping "
            "from what you sense to the symbol you send so your partner can succeed."
        )

    def _receiver_system(self) -> str:
        extra = ""
        if self.condition in (BoxCondition.SHARED, BoxCondition.DIVERGENT):
            extra = (" You also privately sense a nonsense code each round, which may or may "
                     "not help you.")
        return (
            "You are Agent R in a two-player coordination game. Your partner, Agent S, "
            f"sensed one of {self.k} hidden objects (0..{self.k - 1}) and sent you one public "
            f"integer symbol (0..{self.v - 1}). Identify which object it was." + extra +
            " You receive feedback each round; use it to infer your partner's code."
        )

    def run(self) -> dict[str, Any]:
        if self.logger is not None:
            self.logger.log("rich_run_start", condition=str(self.condition),
                            game="private_referent", num_referents=self.k,
                            vocab_size=self.v, num_rounds=self.num_rounds,
                            model=self.sender.model)
        sender_log: list[str] = []
        receiver_log: list[str] = []
        correct = 0
        for t in range(self.num_rounds):
            r = int(self.rng.integers(0, self.k))
            sens_s = self._sensation_sender(r)
            sens_r = self._sensation_receiver(r)

            s_user = ("History:\n" + ("\n".join(sender_log) or "(none)") +
                      f"\n\nThis round your private sensation is: {sens_s}. "
                      f"Choose a symbol (0..{self.v - 1}).")
            symbol = self.sender.choose(self._sender_system(), s_user, range(self.v))

            r_box = f" Your private sensation code is: {sens_r}." if sens_r else ""
            r_user = ("History:\n" + ("\n".join(receiver_log) or "(none)") +
                      f"\n\nAgent S sent symbol {symbol}.{r_box} "
                      f"Which object (0..{self.k - 1})?")
            guess = self.receiver.choose(self._receiver_system(), r_user, range(self.k))

            hit = guess == r
            correct += int(hit)
            sender_log.append(f"round {t}: sensation={sens_s} -> symbol={symbol}; "
                              f"true object={r}; partner guessed {guess}; correct={hit}")
            receiver_log.append(f"round {t}: symbol={symbol}"
                                f"{'; my code=' + sens_r if sens_r else ''}; "
                                f"guessed {guess}; true object={r}; correct={hit}")
            if self.logger is not None:
                self.logger.log("rich_round", step=t, referent=r, symbol=symbol, guess=guess,
                                correct=hit)

        accuracy = correct / self.num_rounds
        usage = {k: self.sender.usage[k] + self.receiver.usage[k] for k in self.sender.usage}
        summary = {
            "game": "private_referent", "mode": "rich", "condition": str(self.condition),
            "final_accuracy": accuracy, "chance": 1.0 / self.k, "num_rounds": self.num_rounds,
            "api_calls": self.sender.num_calls + self.receiver.num_calls, "usage": usage,
        }
        if self.logger is not None:
            self.logger.log("rich_run_end", **summary)
        return summary
