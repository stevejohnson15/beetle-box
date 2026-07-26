# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""Score an E1 run against its frozen pre-registration.

Usage::

    python -m beetlebox.analysis.e1 results/<config_hash>/seed0 [--prereg PATH]

Reads the run's logs, reconstructs the environment/channel from the manifest,
computes the pre-registered metrics, applies the frozen thresholds, and prints a
report. The report ends with the *licenses* text from the prereg -- a standing
reminder of what the result does and does not entitle us to say.
"""

from __future__ import annotations

import argparse
from typing import Any

from beetlebox.analysis import metrics
from beetlebox.analysis.prereg import DEFAULT_PREREG, load_prereg
from beetlebox.channels import SymbolChannel
from beetlebox.config import from_dict
from beetlebox.envs import SignalingEnv
from beetlebox.runlog import iter_events, read_manifest


def score_run(run_dir: str, prereg_path: str = DEFAULT_PREREG) -> dict[str, Any]:
    """Compute metrics and pre-registered checks for a run directory."""
    prereg = load_prereg(prereg_path)
    thr = prereg.get("thresholds", {})
    manifest = read_manifest(run_dir)
    cfg = from_dict(manifest["config"])
    env = SignalingEnv(cfg.env)
    channel = SymbolChannel.from_config(cfg.channel)

    events = list(iter_events(run_dir))
    eval_events = [e for e in events if e.get("event") == "eval"]
    run_end = next((e for e in events if e.get("event") == "run_end"), {})

    final_accuracy = run_end.get("final_accuracy")
    chance = run_end.get("chance", 1.0 / env.num_classes)
    turnover_step = run_end.get("turnover_step")

    stability = metrics.convention_stability(eval_events, int(thr.get("stability_window", 5)))
    fidelity = metrics.transmission_fidelity(eval_events, turnover_step)
    topsim = metrics.topographic_similarity(
        run_end.get("final_mapping", []), env, channel
    )

    checks: dict[str, Any] = {}
    if cfg.experiment.feedback:
        checks["emergence"] = {
            "pass": final_accuracy is not None
            and final_accuracy >= thr.get("emergence_accuracy", 0.75)
            and final_accuracy >= chance * thr.get("emergence_ratio_over_chance", 1.5),
            "final_accuracy": final_accuracy,
            "chance": chance,
        }
    else:
        checks["ablation_at_chance"] = {
            "pass": final_accuracy is not None
            and final_accuracy <= thr.get("ablation_max_accuracy", 2.0 * chance),
            "final_accuracy": final_accuracy,
            "chance": chance,
        }

    if stability == stability:  # not NaN
        checks["stability"] = {
            "pass": stability >= thr.get("stability_min", 0.9),
            "stability": stability,
        }
    if fidelity.get("applicable"):
        checks["turnover_fidelity"] = {
            "pass": fidelity["post_turnover_accuracy"] >= thr.get("turnover_fidelity_min", 0.75),
            **fidelity,
        }
    if topsim.get("applicable"):
        checks["topographic_similarity"] = {
            "pass": topsim["rho"] >= thr.get("topsim_min_rho", 0.3),
            **topsim,
        }

    return {
        "run_dir": run_dir,
        "config_hash": manifest.get("config_hash"),
        "seed": manifest.get("seed"),
        "feedback": cfg.experiment.feedback,
        "final_accuracy": final_accuracy,
        "chance": chance,
        "stability": stability,
        "fidelity": fidelity,
        "topographic_similarity": topsim,
        "checks": checks,
        "licenses": prereg.get("licenses", ""),
    }


def _format_report(result: dict[str, Any]) -> str:
    lines = []
    lines.append(f"E1 report -- {result['run_dir']}")
    lines.append(f"  config_hash={result['config_hash']}  seed={result['seed']}  "
                 f"feedback={result['feedback']}")
    lines.append(f"  final_accuracy={result['final_accuracy']:.3f}  "
                 f"chance={result['chance']:.3f}")
    lines.append(f"  convention_stability={result['stability']}")
    fid = result["fidelity"]
    if fid.get("applicable"):
        lines.append(f"  turnover: pre={fid['pre_turnover_accuracy']:.3f} "
                     f"post={fid['post_turnover_accuracy']:.3f} "
                     f"recovery_ratio={fid['recovery_ratio']:.3f}")
    ts = result["topographic_similarity"]
    if ts.get("applicable"):
        lines.append(f"  topographic_similarity rho={ts['rho']:.3f} p={ts['p_value']:.3g}")
    lines.append("  pre-registered checks:")
    for name, c in result["checks"].items():
        status = "PASS" if c.get("pass") else "FAIL"
        lines.append(f"    [{status}] {name}")
    if result["licenses"]:
        lines.append("  what this does / does not license:")
        for ln in str(result["licenses"]).strip().splitlines():
            lines.append(f"    {ln}")
    return "\n".join(lines)


def main() -> None:
    """CLI: score one E1 run against its frozen prereg and print the report."""
    ap = argparse.ArgumentParser(description="Score an E1 run against its frozen prereg.")
    ap.add_argument("run_dir", help="path to results/<config_hash>/seed<seed>")
    ap.add_argument("--prereg", default=DEFAULT_PREREG, help="path to frozen prereg YAML")
    args = ap.parse_args()
    result = score_run(args.run_dir, args.prereg)
    print(_format_report(result))


if __name__ == "__main__":
    main()
