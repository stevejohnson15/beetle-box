# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""Score an E3 condition-sweep against the frozen pre-registration.

E3's result is a *comparison across conditions*, so this scorer takes several run
directories (one per box condition, same game) and computes:

  * per-condition final accuracy vs. chance,
  * the earnable-cancellation check (informative boxes must beat empty ones where
    the box is the only route to the answer), and
  * the beetle-drops-out gap (shared - divergent), classified per the prereg.

Usage::

    python -m beetlebox.analysis.e3 results/<hashA>/seed0 results/<hashB>/seed0 ...

Pass the run dirs for all four conditions of one game.
"""

from __future__ import annotations

import argparse
from typing import Any

from beetlebox.analysis.prereg import load_prereg
from beetlebox.runlog import iter_events, read_manifest

DEFAULT_PREREG = "prereg/e3_beetle_box.yaml"
_BOX_ONLY_GAMES = {"private_referent", "sensation_matching"}


def _run_summary(run_dir: str) -> dict[str, Any]:
    manifest = read_manifest(run_dir)
    end = next((e for e in iter_events(run_dir)
                if e.get("event") in ("run_end", "rich_run_end")), {})
    return {
        "condition": end.get("condition") or manifest["config"]["box"]["condition"],
        "game": end.get("game") or manifest["config"]["experiment"]["game"],
        "final_accuracy": end.get("final_accuracy"),
        "chance": end.get("chance"),
    }


def score_game(run_dirs: list[str], prereg_path: str = DEFAULT_PREREG) -> dict[str, Any]:
    """Score one game's condition sweep against the frozen pre-registration.

    ``run_dirs`` must be runs of the *same* game (one per box condition). Returns
    per-condition accuracies plus the earnable-cancellation check and the
    beetle-drops-out gap. Raises ``ValueError`` if the runs mix games.
    """
    prereg = load_prereg(prereg_path)
    thr = prereg.get("thresholds", {})
    runs = [_run_summary(d) for d in run_dirs]
    games = {r["game"] for r in runs}
    if len(games) != 1:
        raise ValueError(f"all runs must be the same game; got {games}")
    game = games.pop()
    by_cond = {r["condition"]: r["final_accuracy"] for r in runs}
    chance = next((r["chance"] for r in runs if r["chance"] is not None), None)

    checks: dict[str, Any] = {}

    # Earnable cancellation: informative boxes must beat empty (box-only games).
    if game in _BOX_ONLY_GAMES and "empty" in by_cond:
        informative = [by_cond[c] for c in ("shared", "divergent") if c in by_cond]
        if informative:
            inf_mean = sum(informative) / len(informative)
            gap = inf_mean - by_cond["empty"]
            checks["earnable"] = {
                "pass": gap >= thr.get("earnable_margin", 0.25),
                "informative_mean": inf_mean, "empty": by_cond["empty"], "gap": gap,
            }

    # Beetle-drops-out gap: shared vs. divergent.
    if "shared" in by_cond and "divergent" in by_cond:
        drop_gap = by_cond["shared"] - by_cond["divergent"]
        tol = thr.get("drop_gap_tolerance", 0.10)
        checks["beetle_gap"] = {
            "shared": by_cond["shared"], "divergent": by_cond["divergent"],
            "gap": drop_gap,
            "verdict": ("beetle_drops_out" if abs(drop_gap) <= tol
                        else "shared_inner_state_helps"),
        }

    return {
        "game": game,
        "chance": chance,
        "by_condition": by_cond,
        "checks": checks,
        "licenses": prereg.get("licenses", ""),
    }


def _format_report(result: dict[str, Any]) -> str:
    lines = [f"E3 report -- game={result['game']}  chance={result['chance']}"]
    lines.append("  final accuracy by condition:")
    for cond in ("shared", "divergent", "empty", "noise"):
        if cond in result["by_condition"]:
            lines.append(f"    {cond:9s} {result['by_condition'][cond]:.3f}")
    checks = result["checks"]
    if "earnable" in checks:
        e = checks["earnable"]
        status = "PASS" if e["pass"] else "FAIL"
        lines.append(f"  [{status}] earnable-cancellation: "
                     f"informative_mean={e['informative_mean']:.3f} "
                     f"empty={e['empty']:.3f} gap={e['gap']:.3f}")
    if "beetle_gap" in checks:
        b = checks["beetle_gap"]
        lines.append(f"  beetle gap (shared-divergent)={b['gap']:.3f} -> {b['verdict']}")
    if result["licenses"]:
        lines.append("  what this does / does not license:")
        for ln in str(result["licenses"]).strip().splitlines():
            lines.append(f"    {ln}")
    return "\n".join(lines)


def main() -> None:
    """CLI: score an E3 condition sweep and print the report."""
    ap = argparse.ArgumentParser(description="Score an E3 condition-sweep.")
    ap.add_argument("run_dirs", nargs="+", help="run dirs for the conditions of one game")
    ap.add_argument("--prereg", default=DEFAULT_PREREG)
    args = ap.parse_args()
    print(_format_report(score_game(args.run_dirs, args.prereg)))


if __name__ == "__main__":
    main()
