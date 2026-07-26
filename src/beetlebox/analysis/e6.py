# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""Score E6 (the reflexive layer) against its frozen pre-registration.

E6 asks whether agents examining their *own* coordination changes it, or merely adds
a plausible story on top. The only admissible evidence is **behavioral**: the change
in coordination accuracy across the intervention (``delta = post - pre``), and — the
decisive quantity — the **contrast against a matched control** interlude:

    reflection_effect = delta(reflect) - delta(control)

If this is within a pre-registered null band, self-examination did not change the
coordination beyond simply adding context: it "merely added a plausible story on
top." If it exceeds the band, reflection was behaviorally load-bearing.

**Strict guardrail (plan §4.6):** the reflection transcripts are *not* scored and
are not an input here. No output of this analysis licenses a claim about whether the
agents "really" share meaning — only whether reflection changed their behavior
relative to control.

Usage::

    # score a single run's within-run delta:
    python -m beetlebox.analysis.e6 results/<reflect_run>
    # or contrast reflect vs control (the decisive comparison):
    python -m beetlebox.analysis.e6 --reflect results/<reflect_run> --control results/<control_run>
"""

from __future__ import annotations

import argparse
from typing import Any

from beetlebox.analysis.prereg import load_prereg
from beetlebox.runlog import iter_events, read_manifest

DEFAULT_PREREG = "prereg/e6_reflexive.yaml"


def _run_end(run_dir: str) -> dict[str, Any]:
    end = next((e for e in iter_events(run_dir) if e.get("event") == "run_end"), None)
    if end is None:
        raise ValueError(f"no 'run_end' event in {run_dir}")
    return end


def score_run(run_dir: str) -> dict[str, Any]:
    """Return one run's behavioral metrics (pre/post/delta + intervention)."""
    end = _run_end(run_dir)
    manifest = read_manifest(run_dir)
    return {
        "run_dir": run_dir,
        "intervention": end.get("intervention") or manifest["config"]["intervention"],
        "pre_accuracy": end["pre_accuracy"],
        "post_accuracy": end["post_accuracy"],
        "delta": end["delta"],
        "chance": end.get("chance"),
    }


def _as_list(dirs: str | list[str]) -> list[str]:
    return [dirs] if isinstance(dirs, str) else list(dirs)


def _mean_delta(dirs: list[str]) -> tuple[float, list[float]]:
    deltas = [score_run(d)["delta"] for d in dirs]
    return sum(deltas) / len(deltas), deltas


def compare(reflect_dirs: str | list[str], control_dirs: str | list[str],
            prereg_path: str = DEFAULT_PREREG) -> dict[str, Any]:
    """Contrast reflect vs control (the decisive comparison).

    Each argument may be a single run dir or a list of same-condition dirs (e.g.
    multiple seeds); deltas are averaged so a single noisy run does not drive the
    verdict -- E6 is easy to over-read, so the estimate is aggregated by design.
    """
    prereg = load_prereg(prereg_path)
    band = prereg.get("thresholds", {}).get("null_band", 0.15)
    reflect_dirs, control_dirs = _as_list(reflect_dirs), _as_list(control_dirs)
    mean_r, deltas_r = _mean_delta(reflect_dirs)
    mean_c, deltas_c = _mean_delta(control_dirs)
    effect = mean_r - mean_c
    verdict = ("reflection_changes_coordination" if abs(effect) > band
               else "reflection_adds_a_story_only")
    return {
        "reflect": {"mean_delta": mean_r, "deltas": deltas_r, "n": len(deltas_r)},
        "control": {"mean_delta": mean_c, "deltas": deltas_c, "n": len(deltas_c)},
        "reflection_effect": effect,
        "null_band": band,
        "verdict": verdict,
        "licenses": prereg.get("licenses", ""),
    }


def _format_single(r: dict[str, Any]) -> str:
    return (f"E6 run -- {r['run_dir']}\n"
            f"  intervention={r['intervention']}  pre={r['pre_accuracy']:.3f}  "
            f"post={r['post_accuracy']:.3f}  delta={r['delta']:+.3f}  chance={r['chance']}")


def _format_comparison(r: dict[str, Any]) -> str:
    lines = ["E6 report -- reflect vs control",
             f"  reflect: mean_delta={r['reflect']['mean_delta']:+.3f} "
             f"(n={r['reflect']['n']}, deltas={[round(d, 3) for d in r['reflect']['deltas']]})",
             f"  control: mean_delta={r['control']['mean_delta']:+.3f} "
             f"(n={r['control']['n']}, deltas={[round(d, 3) for d in r['control']['deltas']]})",
             f"  reflection_effect = mean_delta(reflect) - mean_delta(control) = "
             f"{r['reflection_effect']:+.3f}  (null band ±{r['null_band']})",
             f"  verdict: {r['verdict']}"]
    if r["licenses"]:
        lines.append("  what this does / does not license:")
        for ln in str(r["licenses"]).strip().splitlines():
            lines.append(f"    {ln}")
    return "\n".join(lines)


def main() -> None:
    """CLI: score a single E6 run, or contrast reflect vs control."""
    ap = argparse.ArgumentParser(description="Score an E6 reflexive run / contrast.")
    ap.add_argument("run_dir", nargs="?", help="a single run dir to summarize")
    ap.add_argument("--reflect", nargs="+", help="reflect run dir(s) (multiple seeds ok)")
    ap.add_argument("--control", nargs="+", help="control run dir(s) (multiple seeds ok)")
    args = ap.parse_args()
    if args.reflect and args.control:
        print(_format_comparison(compare(args.reflect, args.control)))
    elif args.run_dir:
        print(_format_single(score_run(args.run_dir)))
    else:
        ap.error("provide a run_dir, or both --reflect and --control")


if __name__ == "__main__":
    main()
