# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""Score an E4 behavioral run (quus) against its frozen pre-registration.

The metrics read out what the seeded students extrapolated above the bend, where
``plus`` and ``quus`` finally disagree:

- **plus_rate** — mean over students of the fraction of above-bend pairs predicted
  as ``plus``. High = the students extend ordinary addition.
- **quus_rate** — likewise for the ``quus`` value.
- **cross_seed_agreement** — fraction of above-bend pairs where *all* students
  predict the same thing. High = convergence (a shared inductive prior fixing the
  rule); low = divergence (the underdetermination is unresolved).

The frozen prereg then interprets these per encoding: a representable (``scalar``)
encoding should let a simplicity prior converge toward plus; an ``onehot`` encoding
leaves the above-bend outputs unconstrained, so students diverge.

Usage::

    python -m beetlebox.analysis.e4 results/<config_hash>/seed<seed>
"""

from __future__ import annotations

import argparse
from typing import Any

from beetlebox.analysis.prereg import load_prereg
from beetlebox.runlog import iter_events, read_manifest

DEFAULT_PREREG = "prereg/e4_quus.yaml"


def _rate(predictions: list[list[int]], target: list[int]) -> float:
    """Mean over students of the fraction of items matching ``target``."""
    per_student = [sum(int(p == t) for p, t in zip(preds, target, strict=True)) / len(target)
                   for preds in predictions]
    return sum(per_student) / len(per_student)


def cross_seed_agreement(predictions: list[list[int]]) -> float:
    """Fraction of items on which every student predicts the same value."""
    n_items = len(predictions[0])
    agree = sum(len({preds[j] for preds in predictions}) == 1 for j in range(n_items))
    return agree / n_items


def _load_predictions(run_dir: str) -> dict[str, Any]:
    ev = next((e for e in iter_events(run_dir) if e.get("event") == "predictions"), None)
    if ev is None:
        raise ValueError(f"no 'predictions' event in {run_dir}")
    return ev


def score_run(run_dir: str, prereg_path: str = DEFAULT_PREREG) -> dict[str, Any]:
    """Compute the E4 behavioral metrics and apply the frozen checks."""
    prereg = load_prereg(prereg_path)
    thr = prereg.get("thresholds", {})
    manifest = read_manifest(run_dir)
    end = next((e for e in iter_events(run_dir) if e.get("event") == "run_end"), {})
    ev = _load_predictions(run_dir)
    predictions, plus, quus = ev["predictions"], ev["plus"], ev["quus"]
    encoding = end.get("encoding") or manifest["config"]["quus"]["encoding"]

    plus_rate = _rate(predictions, plus)
    quus_rate = _rate(predictions, quus)
    agreement = cross_seed_agreement(predictions)

    checks: dict[str, Any] = {}
    if encoding == "scalar":
        checks["shared_prior_resolves"] = {
            "pass": plus_rate >= thr.get("scalar_min_plus_rate", 0.6)
            and agreement >= thr.get("scalar_min_agreement", 0.5),
            "plus_rate": plus_rate, "cross_seed_agreement": agreement,
        }
    elif encoding == "onehot":
        checks["underdetermination_visible"] = {
            "pass": plus_rate <= thr.get("onehot_max_plus_rate", 0.2)
            and agreement <= thr.get("onehot_max_agreement", 0.6),
            "plus_rate": plus_rate, "cross_seed_agreement": agreement,
        }

    return {
        "run_dir": run_dir,
        "encoding": encoding,
        "plus_rate": plus_rate,
        "quus_rate": quus_rate,
        "cross_seed_agreement": agreement,
        "num_seeds": len(predictions),
        "checks": checks,
        "licenses": prereg.get("licenses", ""),
    }


def _format_report(r: dict[str, Any]) -> str:
    lines = [f"E4 report -- {r['run_dir']}",
             f"  encoding={r['encoding']}  num_seeds={r['num_seeds']}",
             f"  plus_rate={r['plus_rate']:.3f}  quus_rate={r['quus_rate']:.3f}  "
             f"cross_seed_agreement={r['cross_seed_agreement']:.3f}"]
    for name, c in r["checks"].items():
        lines.append(f"  [{'PASS' if c.get('pass') else 'FAIL'}] {name}")
    if r["licenses"]:
        lines.append("  what this does / does not license:")
        for ln in str(r["licenses"]).strip().splitlines():
            lines.append(f"    {ln}")
    return "\n".join(lines)


def main() -> None:
    """CLI: score one E4 behavioral run against its frozen prereg."""
    ap = argparse.ArgumentParser(description="Score an E4 quus run.")
    ap.add_argument("run_dir", help="path to results/<config_hash>/seed<seed>")
    ap.add_argument("--prereg", default=DEFAULT_PREREG)
    args = ap.parse_args()
    print(_format_report(score_run(args.run_dir, args.prereg)))


if __name__ == "__main__":
    main()
