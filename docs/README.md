<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Beetle-Box contributors
-->
# Beetle-Box documentation

This directory holds the per-experiment documentation. Beetle-Box follows a
**three-part documentation pattern for every experiment** — apply it to E1–E6
and anything added later.

## The per-experiment pattern

For each experiment `eN_<name>`, provide all three of:

1. **`docs/eN_<name>.md` — the approach.** A full, self-contained description:
   the Wittgensteinian target and why it is being operationalized; the setup;
   the manipulations (independent variables); the metrics and their frozen
   thresholds; **what the results do and do not license** (kept honest per
   `plan/beetle-box.md` §3); how to run it; and references. A reader should be
   able to understand the experiment from this file alone.

2. **`notebooks/eN_<name>.ipynb` — a working demonstration.** A narrated,
   executable notebook that runs the experiment live (small/fast settings),
   shows the emergent behavior, plots the key metrics, and walks the reader
   through the interpretation. It must execute top-to-bottom without error
   (`make notebooks` / `jupyter nbconvert --execute` — see below).

3. **`results/exemplary/eN_<name>/` — an exemplary run (where appropriate).** A
   committed, curated write-up of one representative run: `README.md` describing
   the run and its outcome, plus the raw artifacts (`manifest.json`,
   `events.jsonl`), the scored report, and any figures. This is the
   "here is what a good run looks like" reference. Transient runs under
   `results/<hash>/seed<seed>/` stay gitignored; only `results/exemplary/` is
   committed.

Cross-link the three: the approach doc points at the notebook and exemplary run;
the notebook cites the approach doc; the exemplary README names the exact command
and config that produced it.

**Background reading.** [`supplemental_reading.md`](supplemental_reading.md) is the
consolidated, per-experiment academic bibliography — the philosophy and technical
literature needed to understand and extend the framework.

## Index

| Experiment | Approach | Notebook | Exemplary run |
|---|---|---|---|
| E1 — convention from use (signaling) | [`e1_signaling.md`](e1_signaling.md) | [`../notebooks/e1_signaling.ipynb`](../notebooks/e1_signaling.ipynb) | [`../results/exemplary/e1_signaling/`](../results/exemplary/e1_signaling/) |
| E2 — the private diarist (§258) | [`e2_diarist.md`](e2_diarist.md) | [`../notebooks/e2_diarist.ipynb`](../notebooks/e2_diarist.ipynb) | [`../results/exemplary/e2_diarist/`](../results/exemplary/e2_diarist/) |
| E3 — the beetle-box (§293) | [`e3_design.md`](e3_design.md) | [`../notebooks/e3_beetle_box.ipynb`](../notebooks/e3_beetle_box.ipynb) | [`../results/exemplary/e3_beetle_box/`](../results/exemplary/e3_beetle_box/) |
| E4 — quus / rule-following | [`e4_quus.md`](e4_quus.md) | [`../notebooks/e4_quus.ipynb`](../notebooks/e4_quus.ipynb) | [`../results/exemplary/e4_quus/`](../results/exemplary/e4_quus/) |
| E5 — forms of life / grounding | [`e5_forms_of_life.md`](e5_forms_of_life.md) | [`../notebooks/e5_forms_of_life.ipynb`](../notebooks/e5_forms_of_life.ipynb) | [`../results/exemplary/e5_forms_of_life/`](../results/exemplary/e5_forms_of_life/) |
| E6 — the reflexive layer | [`e6_reflexive.md`](e6_reflexive.md) | [`../notebooks/e6_reflexive.ipynb`](../notebooks/e6_reflexive.ipynb) | [`../results/exemplary/e6_reflexive/`](../results/exemplary/e6_reflexive/) |

## Running the notebooks

```bash
uv sync --extra notebooks          # jupyter, nbconvert, matplotlib, ...
# Execute a notebook in place (verifies it runs end-to-end):
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/e1_signaling.ipynb
# Or open interactively:
uv run jupyter lab notebooks/e1_signaling.ipynb
```

## Conventions

- Every committed source, doc, and config file carries the Apache-2.0 SPDX header
  (`plan/beetle-box.md`, `CONTRIBUTING.md`). Markdown uses an HTML-comment header.
- Notebooks are kept small and fast (seconds, CPU-only) so they double as smoke
  tests; heavy sweeps belong in `experiments/` runs, not notebooks.
- Scoring in docs and notebooks uses the **frozen** `prereg/eN_<name>.yaml`
  criteria — never thresholds invented after the fact.
