# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""Score an E5 (forms of life / grounding) run against its frozen pre-registration.

The headline question is whether grounding changes the **character** of the
emergent language. The sharpest handle is **selectivity**: does the language mark
only the payoff-relevant distinctions, or full referent identity?

- **relevant / irrelevant recoverability** — for each grid attribute, the fraction
  of referents whose attribute value is pinned down by their message (all referents
  sharing a message share that attribute value). A grounded language should make the
  *payoff-relevant* attribute (attribute 0) highly recoverable and the *irrelevant*
  one (attribute 1) not; an ungrounded language, needing full identity, makes both
  recoverable.
- **selectivity_gap** = relevant − irrelevant recoverability. Large under grounding,
  ~0 under bare identification.
- **lexicon_size** — distinct messages used; grounded uses far fewer.

Secondary (read from the run summary): native performance, robustness to channel
noise, and transfer to a fresh receiver (turnover).

Usage::

    python -m beetlebox.analysis.e5 results/<config_hash>/seed<seed>
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Any

import numpy as np

from beetlebox.analysis.prereg import load_prereg
from beetlebox.config import EnvConfig
from beetlebox.envs import SignalingEnv
from beetlebox.runlog import iter_events, read_manifest

DEFAULT_PREREG = "prereg/e5_forms_of_life.yaml"


def recoverability(mapping: list[list[int]], env: SignalingEnv) -> tuple[list[float], int]:
    """Per-attribute recoverability from the referent->message map, and lexicon size.

    Returns ``([recover_attr0, recover_attr1, ...], distinct_messages)``. Attribute
    ``j`` is *recoverable* for a referent when every referent sharing its message
    also shares that attribute value.
    """
    msgs = [tuple(m) for m in mapping]
    attrs = np.stack([env.attributes_for(r) for r in range(env.num_classes)])
    groups: dict[tuple, list[int]] = defaultdict(list)
    for r, m in enumerate(msgs):
        groups[m].append(r)
    out = []
    for j in range(attrs.shape[1]):
        hits = sum(1 for r in range(len(msgs))
                   if len({attrs[x, j] for x in groups[msgs[r]]}) == 1)
        out.append(hits / len(msgs))
    return out, len(set(msgs))


def score_run(run_dir: str, prereg_path: str = DEFAULT_PREREG) -> dict[str, Any]:
    """Compute the E5 selectivity metrics and apply the frozen pre-registered checks."""
    prereg = load_prereg(prereg_path)
    thr = prereg.get("thresholds", {})
    manifest = read_manifest(run_dir)
    end = next((e for e in iter_events(run_dir) if e.get("event") == "run_end"), {})
    grounded = end.get("grounded", manifest["config"]["experiment"]["grounded"])
    env = SignalingEnv(EnvConfig(**manifest["config"]["env"]))

    recover, lexicon = recoverability(end["final_mapping"], env)
    relevant = recover[0]
    irrelevant = recover[1] if len(recover) > 1 else float("nan")
    gap = relevant - irrelevant

    checks: dict[str, Any] = {}
    if grounded:
        checks["grounding_is_selective"] = {
            "pass": gap >= thr.get("grounded_min_selectivity_gap", 0.5),
            "selectivity_gap": gap,
        }
    else:
        checks["encodes_full_identity"] = {
            "pass": gap <= thr.get("ungrounded_max_selectivity_gap", 0.2)
            and relevant >= thr.get("ungrounded_min_recoverability", 0.6),
            "selectivity_gap": gap, "relevant_recoverability": relevant,
        }

    return {
        "run_dir": run_dir,
        "grounded": grounded,
        "performance": end.get("performance"),
        "relevant_recoverability": relevant,
        "irrelevant_recoverability": irrelevant,
        "selectivity_gap": gap,
        "lexicon_size": lexicon,
        "num_classes": end.get("num_classes", env.num_classes),
        "robustness_ratio": end.get("robustness_ratio"),
        "transfer": {"pre": end.get("pre_turnover_performance"),
                     "post": end.get("post_turnover_performance")},
        "checks": checks,
        "licenses": prereg.get("licenses", ""),
    }


def _format_report(r: dict[str, Any]) -> str:
    lines = [f"E5 report -- {r['run_dir']}",
             f"  grounded={r['grounded']}  performance={r['performance']:.3f}",
             f"  lexicon_size={r['lexicon_size']}/{r['num_classes']}  "
             f"relevant_recoverability={r['relevant_recoverability']:.3f}  "
             f"irrelevant_recoverability={r['irrelevant_recoverability']:.3f}  "
             f"selectivity_gap={r['selectivity_gap']:.3f}",
             f"  robustness_ratio={r['robustness_ratio']:.3f}"]
    t = r["transfer"]
    if t["pre"] is not None:
        lines.append(f"  transfer (post/pre turnover)={t['post']:.3f}/{t['pre']:.3f}")
    for name, c in r["checks"].items():
        lines.append(f"  [{'PASS' if c.get('pass') else 'FAIL'}] {name}")
    if r["licenses"]:
        lines.append("  what this does / does not license:")
        for ln in str(r["licenses"]).strip().splitlines():
            lines.append(f"    {ln}")
    return "\n".join(lines)


def main() -> None:
    """CLI: score one E5 run against its frozen prereg and print the report."""
    ap = argparse.ArgumentParser(description="Score an E5 forms-of-life run.")
    ap.add_argument("run_dir", help="path to results/<config_hash>/seed<seed>")
    ap.add_argument("--prereg", default=DEFAULT_PREREG)
    args = ap.parse_args()
    print(_format_report(score_run(args.run_dir, args.prereg)))


if __name__ == "__main__":
    main()
