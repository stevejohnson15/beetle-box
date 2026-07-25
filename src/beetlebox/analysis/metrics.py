# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""Pure metric functions for E1.

These compute the pre-registered quantities (emergence, convention stability,
transmission fidelity across turnover, topographic similarity) from a run's
logged events. They are deliberately free of any pass/fail thresholds -- the
thresholds live in the frozen pre-registration and are applied only in
:mod:`beetlebox.analysis.e1`. This keeps scoring honest (``plan/beetle-box.md``
§3.2): the criteria are fixed before the run, not tuned to it.
"""

from __future__ import annotations

from typing import Any

from scipy.stats import spearmanr

from beetlebox.channels import SymbolChannel
from beetlebox.envs import SignalingEnv


def _mapping_agreement(a: list[list[int]], b: list[list[int]]) -> float:
    """Fraction of referents assigned an identical message in both mappings."""
    if not a or len(a) != len(b):
        return 0.0
    same = sum(1 for ma, mb in zip(a, b, strict=True) if list(ma) == list(mb))
    return same / len(a)


def convention_stability(eval_events: list[dict[str, Any]], window: int) -> float:
    """Mean agreement of the greedy referent->message map over the last ``window`` evals.

    1.0 means the convention was frozen across those snapshots; low values mean
    the referent->message assignment was still wandering.
    """
    maps = [e["mapping"] for e in eval_events if "mapping" in e]
    tail = maps[-(window + 1):] if window > 0 else maps
    if len(tail) < 2:
        return float("nan")
    agreements = [_mapping_agreement(tail[i], tail[i + 1]) for i in range(len(tail) - 1)]
    return sum(agreements) / len(agreements)


def transmission_fidelity(eval_events: list[dict[str, Any]],
                          turnover_step: int | None) -> dict[str, Any]:
    """Accuracy just before turnover vs. after a fresh receiver is introduced.

    A convention that "survives its founders" is re-adopted by the new agent, so
    post-turnover accuracy recovers toward the pre-turnover level.
    """
    if turnover_step is None:
        return {"applicable": False}
    pre = [e for e in eval_events if e.get("step", 0) <= turnover_step]
    post = [e for e in eval_events if e.get("step", 0) > turnover_step]
    if not pre or not post:
        return {"applicable": False}
    pre_acc = pre[-1]["accuracy"]
    post_acc = post[-1]["accuracy"]
    return {
        "applicable": True,
        "pre_turnover_accuracy": pre_acc,
        "post_turnover_accuracy": post_acc,
        "recovery_ratio": (post_acc / pre_acc) if pre_acc > 0 else float("nan"),
    }


def topographic_similarity(mapping: list[list[int]], env: SignalingEnv,
                           channel: SymbolChannel) -> dict[str, Any]:
    """Spearman correlation of pairwise referent distance vs. message distance.

    Meaningful only when referents carry structure (grid mode) and messages carry
    more than one symbol; otherwise it is reported as not-applicable rather than
    as a spurious number.
    """
    applicable = env.mode == "grid" and channel.message_length > 1
    if not applicable:
        return {"applicable": False}
    k = len(mapping)
    ref_d: list[int] = []
    msg_d: list[int] = []
    for i in range(k):
        for j in range(i + 1, k):
            ref_d.append(env.referent_distance(i, j))
            msg_d.append(channel.message_distance(mapping[i], mapping[j]))
    rho, p = spearmanr(ref_d, msg_d)
    return {"applicable": True, "rho": float(rho), "p_value": float(p), "num_pairs": len(ref_d)}
