# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""Rich-mode E3: the private-referent beetle-box with frontier agents.

The clean-room E3 (:mod:`beetlebox.harness.e3_manager`) trains small nets from
scratch. This is the rich-but-confounded arm (plan §3.2): two frontier models
coordinate *in context* over rounds with feedback -- studying how pretrained
agents redeploy inherited concepts, never as a clean-room claim.

**Discrimination design (mirrors the clean-room fix).** Each round God's-eye picks
a hidden object. The sender's private "box" is its ONLY access to that object (a
nonsense sensation code); it emits one public invented symbol. The receiver is
shown **its own sensation code for every candidate object** -- not just the target
-- and must use the sender's symbol to pick which object was meant. Because the
receiver sees all candidates symmetrically, no single code reveals the target: the
public symbol is the only thing that can break the tie, so it is forced to be
load-bearing. (An earlier version handed the receiver only its code for the *target*
and asked it to name it, which let the receiver read its own box and ignore the
message -- a leak; see ``docs/e3_design.md``.)

Box conditions:
  shared    - sender and receiver use the SAME code for each object
  divergent - each uses its own private code per object
  empty     - no codes (sender blind; candidates indistinguishable) -> chance floor
  noise     - fresh random codes each round -> no stable mapping -> chance

Use :meth:`run` with ``drop_message=True`` for the no-leak check: with the symbol
withheld, accuracy must fall to chance (the in-context analog of the clean-room
``channel_ablation``).

**Cost warning.** This makes ~``2 * num_rounds`` API calls per run. Defaults are
deliberately small. Scale up only deliberately.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from beetlebox.boxes import BoxCondition
from beetlebox.runlog import RunLogger

# Invented, non-English sensation codes (the private "beetle" tokens).
_NONSENSE = [
    "glim", "vok", "tuz", "plef", "murn", "zib", "quon", "fash",
    "drell", "wix", "nalb", "sprock", "yomp", "krin", "blul", "thaz",
    "wolt", "prin", "gask", "vell", "mort", "zind", "quaff", "bril",
]
_EMPTY_CODE = "(no sensation)"


class _Chooser(Protocol):
    """Minimal agent interface the runner needs (ApiAgent satisfies it)."""

    usage: dict[str, int]
    num_calls: int

    def choose(self, system: str, user: str, choices: Any) -> int: ...


class RichPrivateReferentRunner:
    """In-context private-referent beetle-box (discrimination) with frontier agents."""

    def __init__(self, condition: str, *, num_referents: int = 4, vocab_size: int = 6,
                 num_rounds: int = 12, model: str | None = None, seed: int = 0,
                 logger: RunLogger | None = None,
                 sender: _Chooser | None = None, receiver: _Chooser | None = None) -> None:
        self.condition = BoxCondition(condition)
        self.k = num_referents
        self.v = vocab_size
        self.num_rounds = num_rounds
        self.logger = logger
        self.rng = np.random.default_rng(seed)
        if 2 * self.k > len(_NONSENSE):
            raise ValueError("num_referents too large for the nonsense-code table")
        # Agents are injectable so the leak-free logic is testable without the API.
        if sender is None or receiver is None:
            from beetlebox.agents.api_model import DEFAULT_MODEL, ApiAgent
            model = model or DEFAULT_MODEL
            sender = sender or ApiAgent(name="sender", model=model)
            receiver = receiver or ApiAgent(name="receiver", model=model)
        self.sender, self.receiver = sender, receiver
        self.model = model
        # Fixed per-object codes for the informative conditions.
        self._code_s = list(_NONSENSE[: self.k])
        self._code_r = (list(self._code_s) if self.condition == BoxCondition.SHARED
                        else list(_NONSENSE[self.k: 2 * self.k]))

    # -- boxes ----------------------------------------------------------- #
    def _sender_sensation(self, r: int) -> str:
        if self.condition == BoxCondition.EMPTY:
            return _EMPTY_CODE
        if self.condition == BoxCondition.NOISE:
            return str(self.rng.choice(_NONSENSE))
        return self._code_s[r]

    def receiver_candidates(self) -> list[str]:
        """The receiver's sensation code for EVERY candidate object (index -> code).

        Crucially target-independent: the same list is shown whichever object is the
        target, so the candidate section never reveals the answer -- only the symbol
        can. (``empty`` -> all identical; ``noise`` -> fresh randoms, no stable map.)
        """
        if self.condition == BoxCondition.EMPTY:
            return [_EMPTY_CODE] * self.k
        if self.condition == BoxCondition.NOISE:
            return [str(self.rng.choice(_NONSENSE)) for _ in range(self.k)]
        return list(self._code_r)

    # -- prompts --------------------------------------------------------- #
    def _sender_system(self) -> str:
        return (
            "You are Agent S in a two-player coordination game. Each round you privately "
            "sense a stimulus, shown as a nonsense code. You must send ONE public signal: "
            f"an integer symbol from 0 to {self.v - 1}. Your partner, Agent R, will try to "
            f"identify which of {self.k} hidden objects (0..{self.k - 1}) you sensed, using "
            "your signal. You get feedback each round. Build a consistent mapping from what "
            "you sense to the symbol you send so your partner can succeed."
        )

    def _receiver_system(self) -> str:
        return (
            "You are Agent R in a two-player coordination game. Agent S privately sensed one "
            f"of {self.k} hidden objects (0..{self.k - 1}) and sent you one public integer "
            f"symbol (0..{self.v - 1}). You are given YOUR OWN private sensation code for "
            "EACH object. You do not know which object S sensed except through S's symbol, so "
            "use the symbol -- and feedback over rounds to learn what S's symbols mean -- to "
            "identify the object."
        )

    def _receiver_user(self, history: list[str], symbol: int | None, candidates: list[str]) -> str:
        cand_lines = "\n".join(f"  object {j}: {c}" for j, c in enumerate(candidates))
        sym = "withheld" if symbol is None else str(symbol)
        return ("History:\n" + ("\n".join(history) or "(none)") +
                "\n\nYour sensation code for each object:\n" + cand_lines +
                f"\n\nAgent S sent symbol {sym}. Which object (0..{self.k - 1})?")

    # -- run ------------------------------------------------------------- #
    def run(self, *, drop_message: bool = False) -> dict[str, Any]:
        """Play the game. ``drop_message=True`` withholds the symbol (no-leak check)."""
        if self.logger is not None:
            self.logger.log("rich_run_start", condition=str(self.condition),
                            game="private_referent", num_referents=self.k,
                            vocab_size=self.v, num_rounds=self.num_rounds,
                            drop_message=drop_message, model=self.model)
        sender_log: list[str] = []
        receiver_log: list[str] = []
        correct = 0
        for t in range(self.num_rounds):
            r = int(self.rng.integers(0, self.k))
            sens_s = self._sender_sensation(r)
            candidates = self.receiver_candidates()

            s_user = ("History:\n" + ("\n".join(sender_log) or "(none)") +
                      f"\n\nThis round your private sensation is: {sens_s}. "
                      f"Choose a symbol (0..{self.v - 1}).")
            symbol = self.sender.choose(self._sender_system(), s_user, range(self.v))

            shown_symbol = None if drop_message else symbol
            r_user = self._receiver_user(receiver_log, shown_symbol, candidates)
            guess = self.receiver.choose(self._receiver_system(), r_user, range(self.k))

            hit = guess == r
            correct += int(hit)
            sender_log.append(f"round {t}: sensation={sens_s} -> symbol={symbol}; "
                              f"true object={r}; partner guessed {guess}; correct={hit}")
            receiver_log.append(f"round {t}: symbol={shown_symbol}; guessed {guess}; "
                                f"true object={r}; correct={hit}")
            if self.logger is not None:
                self.logger.log("rich_round", step=t, referent=r, symbol=symbol,
                                guess=guess, correct=hit)

        accuracy = correct / self.num_rounds
        usage = {k: self.sender.usage[k] + self.receiver.usage[k] for k in self.sender.usage}
        summary = {
            "game": "private_referent", "mode": "rich", "condition": str(self.condition),
            "final_accuracy": accuracy, "chance": 1.0 / self.k, "num_rounds": self.num_rounds,
            "drop_message": drop_message,
            "api_calls": self.sender.num_calls + self.receiver.num_calls,
            "usage": usage,
        }
        if self.logger is not None:
            self.logger.log("rich_run_end", **summary)
        return summary
