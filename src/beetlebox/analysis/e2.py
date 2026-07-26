# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""Score an E2 (diarist, §258) run against its frozen pre-registration.

The metrics ask Wittgenstein's question with a God's-eye ruler the diarist never
has: does the diarist name the *same* percept-type with the *same* term over time?

- **consistency** — for each type, the fraction of its occurrences that received
  the type's modal term; averaged over types. 1.0 = a type is always named the
  same; ~0 = it is renamed every time. This is "does 'same again' hold?"
- **purity** — the mirror: for each emitted term, the fraction of its uses that
  fell on the term's modal type; averaged over terms. Low purity means a term is
  overloaded across types (the naming conflates sensations).
- **drift** — consistency computed on the first vs. second half of the run. A
  drop means the naming wandered as the diary grew.
- **gap-conditioned consistency** — consistency split by the time gap since a type
  last appeared (short vs. long gaps). A drop for long gaps is "same again" fading
  with time — the signature of bounded/absent memory.

Usage::

    python -m beetlebox.analysis.e2 results/<config_hash>/seed<seed>
"""

from __future__ import annotations

import argparse
import collections
from typing import Any

from beetlebox.analysis.prereg import load_prereg
from beetlebox.runlog import iter_events, read_manifest

DEFAULT_PREREG = "prereg/e2_diarist.yaml"


def _modal_fraction(groups: dict[Any, list]) -> float:
    """Mean over groups of (most-common-value count / group size)."""
    fracs = [collections.Counter(v).most_common(1)[0][1] / len(v)
             for v in groups.values() if v]
    return sum(fracs) / len(fracs) if fracs else float("nan")


def consistency(types: list[int], terms: list[int]) -> float:
    """Mean over types of the fraction named with that type's modal term."""
    by_type: dict[int, list[int]] = collections.defaultdict(list)
    for t, w in zip(types, terms, strict=True):
        by_type[t].append(w)
    return _modal_fraction(by_type)


def purity(types: list[int], terms: list[int]) -> float:
    """Mean over terms of the fraction of uses falling on that term's modal type."""
    by_term: dict[int, list[int]] = collections.defaultdict(list)
    for t, w in zip(types, terms, strict=True):
        by_term[w].append(t)
    return _modal_fraction(by_term)


def drift(types: list[int], terms: list[int], split: float = 0.5) -> dict[str, float]:
    """Consistency on the early vs. late halves of the run (and their difference)."""
    n = int(len(types) * split)
    early = consistency(types[:n], terms[:n])
    late = consistency(types[n:], terms[n:])
    return {"early": early, "late": late, "drop": early - late}


def gap_conditioned_consistency(types: list[int], terms: list[int], gaps: list[int],
                                gap_threshold: int = 20) -> dict[str, Any]:
    """Consistency for short-gap vs. long-gap recurrences (first sights excluded).

    Each recurrence is scored against the type's *overall* modal term, then split
    by whether the gap since the type last appeared is short or long.
    """
    by_type: dict[int, list[int]] = collections.defaultdict(list)
    for t, w in zip(types, terms, strict=True):
        by_type[t].append(w)
    modal = {t: collections.Counter(ws).most_common(1)[0][0] for t, ws in by_type.items()}
    short_hits = short_n = long_hits = long_n = 0
    for t, w, g in zip(types, terms, gaps, strict=True):
        if g < 0:  # first sight of this type: no "again" to judge
            continue
        hit = int(w == modal[t])
        if g <= gap_threshold:
            short_hits += hit
            short_n += 1
        else:
            long_hits += hit
            long_n += 1
    return {
        "short_gap": short_hits / short_n if short_n else float("nan"),
        "long_gap": long_hits / long_n if long_n else float("nan"),
        "gap_threshold": gap_threshold,
    }


def _load_sequence(run_dir: str) -> tuple[list[int], list[int], list[int]]:
    seq = next((e for e in iter_events(run_dir) if e.get("event") == "sequence"), None)
    if seq is None:
        raise ValueError(f"no 'sequence' event in {run_dir}")
    return seq["types"], seq["terms"], seq["gaps"]


def score_run(run_dir: str, prereg_path: str = DEFAULT_PREREG) -> dict[str, Any]:
    """Compute the E2 metrics and apply the frozen pre-registered checks."""
    prereg = load_prereg(prereg_path)
    thr = prereg.get("thresholds", {})
    manifest = read_manifest(run_dir)
    end = next((e for e in iter_events(run_dir) if e.get("event") == "run_end"), {})
    types, terms, gaps = _load_sequence(run_dir)

    cons = consistency(types, terms)
    pur = purity(types, terms)
    dr = drift(types, terms)
    gap = gap_conditioned_consistency(types, terms, gaps,
                                      int(thr.get("gap_threshold", 20)))
    policy = end.get("policy") or manifest["config"]["diarist"]["policy"]
    memory = end.get("memory") or manifest["config"]["diarist"]["memory"]

    checks: dict[str, Any] = {}
    # For the diary model, full memory should manufacture stability that absent
    # memory cannot -- but this is a per-condition observation, so we only flag the
    # two poles when they are the one under test.
    if policy == "prototype" and memory == "full":
        checks["stability_with_memory"] = {
            "pass": cons >= thr.get("full_memory_min_consistency", 0.9),
            "consistency": cons,
        }
    if policy == "prototype" and memory == "none":
        checks["chance_without_memory"] = {
            "pass": cons <= thr.get("no_memory_max_consistency", 0.1),
            "consistency": cons,
        }

    return {
        "run_dir": run_dir,
        "policy": policy,
        "memory": memory,
        "noise": end.get("noise"),
        "consistency": cons,
        "purity": pur,
        "distinct_terms": end.get("distinct_terms", len(set(terms))),
        "drift": dr,
        "gap": gap,
        "checks": checks,
        "licenses": prereg.get("licenses", ""),
    }


def _format_report(r: dict[str, Any]) -> str:
    lines = [f"E2 report -- {r['run_dir']}",
             f"  policy={r['policy']}  memory={r['memory']}  noise={r['noise']}",
             f"  consistency={r['consistency']:.3f}  purity={r['purity']:.3f}  "
             f"distinct_terms={r['distinct_terms']}",
             f"  drift: early={r['drift']['early']:.3f} late={r['drift']['late']:.3f} "
             f"drop={r['drift']['drop']:.3f}",
             f"  gap-conditioned: short={r['gap']['short_gap']:.3f} "
             f"long={r['gap']['long_gap']:.3f} (threshold {r['gap']['gap_threshold']})"]
    for name, c in r["checks"].items():
        lines.append(f"  [{'PASS' if c.get('pass') else 'FAIL'}] {name}")
    if r["licenses"]:
        lines.append("  what this does / does not license:")
        for ln in str(r["licenses"]).strip().splitlines():
            lines.append(f"    {ln}")
    return "\n".join(lines)


def main() -> None:
    """CLI: score one E2 run against its frozen prereg and print the report."""
    ap = argparse.ArgumentParser(description="Score an E2 diarist run.")
    ap.add_argument("run_dir", help="path to results/<config_hash>/seed<seed>")
    ap.add_argument("--prereg", default=DEFAULT_PREREG)
    args = ap.parse_args()
    print(_format_report(score_run(args.run_dir, args.prereg)))


if __name__ == "__main__":
    main()
