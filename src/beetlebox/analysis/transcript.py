# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""Human-readable transcripts of the E1 signaling exchange.

Two views of the same game:

* **Convention snapshot** -- for every referent, what the sender says and what the
  receiver guesses back, at one moment in training. Watching these snapshots
  across steps shows the shared convention forming: early on, referents *collide*
  (two referents get the same message) and the receiver misses; later the code
  settles into a clean, agreed mapping.

* **Exchange log** -- individual trials rendered as a back-and-forth line
  ("referent 3 -> sender says s6 -> receiver hears s6 -> guesses 3 [OK]"). This is
  the per-trial "chatter," including the stochastic exploration early in training.

Snapshots are reconstructable from a run's ``events.jsonl`` alone (the harness
logs the sender mapping and the receiver guesses at every eval), so::

    python -m beetlebox.analysis.transcript results/<config_hash>/seed<seed>

prints the convention as it evolved, no model weights required.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from typing import Any

from beetlebox.channels import SymbolChannel
from beetlebox.config import from_dict
from beetlebox.runlog import iter_events, read_manifest

_OK, _NO = "OK", "x"


def _render_message(message: list[int]) -> str:
    return "-".join(f"s{int(s)}" for s in message)


def convention_rows(mapping: list[list[int]], guesses: list[int]) -> list[dict[str, Any]]:
    """Per-referent exchange rows, flagging message collisions and misses."""
    seen: dict[str, int] = {}
    rows = []
    for r, (msg, guess) in enumerate(zip(mapping, guesses, strict=True)):
        key = _render_message(msg)
        collides_with = seen.get(key)
        seen.setdefault(key, r)
        rows.append({
            "referent": r,
            "message": msg,
            "message_str": key,
            "guess": guess,
            "correct": r == guess,
            "collision_with": collides_with,  # earlier referent sharing this message
        })
    return rows


def format_convention(mapping: list[list[int]], guesses: list[int], *,
                      step: int | None = None, accuracy: float | None = None) -> str:
    """Render one convention snapshot as an aligned table."""
    rows = convention_rows(mapping, guesses)
    header = "convention snapshot"
    if step is not None:
        header += f" @ step {step}"
    if accuracy is not None:
        header += f"  (accuracy {accuracy:.3f})"
    lines = [header, "  referent  sender says   receiver guesses   result"]
    for row in rows:
        flag = _OK if row["correct"] else _NO
        note = ""
        if row["collision_with"] is not None:
            note = f"   <- collides with referent {row['collision_with']}"
        lines.append(f"  {row['referent']:>8}  {row['message_str']:>11}   "
                     f"{row['guess']:>16}   [{flag}]{note}")
    n_correct = sum(r["correct"] for r in rows)
    lines.append(f"  {n_correct}/{len(rows)} referents correctly communicated")
    return "\n".join(lines)


def format_exchanges(exchanges: list[dict[str, Any]]) -> str:
    """Render a list of individual trials as back-and-forth lines."""
    lines = ["exchange log (individual trials):"]
    for i, ex in enumerate(exchanges):
        flag = _OK if ex["correct"] else _NO
        lines.append(f"  trial {i:>3}: referent {ex['referent']} "
                     f"-> sender says {_render_message(ex['message'])} "
                     f"-> receiver guesses {ex['guess']}  [{flag}]")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Reconstructing convention snapshots from a run's event log
# --------------------------------------------------------------------------- #
def iter_snapshots(run_dir: str) -> Iterator[dict[str, Any]]:
    """Yield ``{step, accuracy, mapping, guesses}`` for each logged eval."""
    for ev in iter_events(run_dir):
        if ev.get("event") == "eval" and "guesses" in ev:
            yield {"step": ev["step"], "accuracy": ev["accuracy"],
                   "mapping": ev["mapping"], "guesses": ev["guesses"]}


def render_run(run_dir: str, steps: list[int] | None = None) -> str:
    """Render the convention at each checkpoint (or a chosen subset of steps)."""
    manifest = read_manifest(run_dir)
    cfg = from_dict(manifest["config"])
    channel = SymbolChannel.from_config(cfg.channel)
    snaps = list(iter_snapshots(run_dir))
    if not snaps:
        return (f"No transcript data in {run_dir}. Re-run the experiment: the "
                "harness must log per-referent guesses (eval events).")
    if steps is not None:
        wanted = set(steps)
        snaps = [s for s in snaps if s["step"] in wanted] or snaps
    blocks = [f"E1 signaling transcript -- {run_dir}",
              f"vocabulary V={channel.vocab_size}, message length L={channel.message_length}",
              ""]
    for s in snaps:
        blocks.append(format_convention(s["mapping"], s["guesses"],
                                        step=s["step"], accuracy=s["accuracy"]))
        blocks.append("")
    return "\n".join(blocks)


def main() -> None:
    """CLI: print a run's exchange transcript reconstructed from its event log."""
    ap = argparse.ArgumentParser(description="Print the E1 signaling exchange transcript.")
    ap.add_argument("run_dir", help="path to results/<config_hash>/seed<seed>")
    ap.add_argument("--steps", type=int, nargs="*", default=None,
                    help="only show these training steps (default: all checkpoints)")
    args = ap.parse_args()
    print(render_run(args.run_dir, args.steps))


if __name__ == "__main__":
    main()
