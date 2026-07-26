<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Beetle-Box contributors
-->
# E6 — The reflexive layer (observer folded into observed)

*The finale, and the meta-example of `plan/beetle-box.md` §1 made into a condition:
agents turned to examine whether **their own** coordination constitutes shared
meaning or its shadow. This is the most interpretively dangerous experiment in the
program — its transcripts are maximally seductive — so it is built, above all, to
resist over-reading.*

- **Approach (this doc)** · **Notebook:** [`../notebooks/e6_reflexive.ipynb`](../notebooks/e6_reflexive.ipynb) · **Exemplary run:** [`../results/exemplary/e6_reflexive/`](../results/exemplary/e6_reflexive/)
- **Frozen pre-registration:** [`../prereg/e6_reflexive.yaml`](../prereg/e6_reflexive.yaml) (the strictest in the project)
- **Design document:** `plan/beetle-box.md` §1 (the meta-example), §3.2 (over-reading guardrail), §4.6

---

## 1. The target

Beetle-Box began (plan §1) in a conversation between a human and an LLM about
whether they could share meaning — a conversation that was itself an instance of
the thing under investigation. Coordination succeeded; the "boxes" never came up.
But from *inside* the exchange, two readings are indistinguishable: the
**deflationary** ("public practice sufficed, the beetle was idle") and the
**form-of-life** ("a high-fidelity shadow, the human silently supplying the
grounding"). That indistinguishability is exactly what §293 predicts — and the
motivating irony of the whole project.

E6 folds the observer into the observed: it turns coordinating agents onto their own
coordination and asks the one question that can be asked without over-reading:

> Does self-examination **change the coordination**, or merely **add a plausible
> story on top of it**?

## 2. The design (built to resist over-reading)

`beetlebox.harness.rich_e6` (rich / frontier mode only — reflection is a linguistic
act):

1. Two frontier agents build a shared reference convention over a **pre** block
   (a repeated signaling game; coordination = accuracy).
2. An **intervention** turn is inserted and folded into each agent's context:
   - **`reflect`** — each agent examines whether it and its partner genuinely share
     the meanings of their symbols, or each is merely privately guessing;
   - **`control`** — each agent answers a matched, task-irrelevant prompt (a
     placebo that adds the same amount of context);
   - **`none`** — no interlude.
3. Coordination is measured again over a **post** block.

The **only admissible signal** is behavioral and control-subtracted:

    reflection_effect = mean_delta(reflect) − mean_delta(control),   delta = post − pre

averaged over seeds so a single noisy run cannot drive the verdict
(`beetlebox.analysis.e6`). If `|effect|` is within the frozen null band, reflection
"added a story only"; if it exceeds it, reflection was behaviorally load-bearing.

**Strict guardrails (frozen prereg):**

- The reflection **transcripts are NOT evidence and are never scored** — an agent
  asserting "we understand each other" counts for nothing. They are stored for the
  record only, structurally separated from the scored data.
- Raw `delta(reflect)` alone is inadmissible — any interlude adds context; only the
  control-subtracted, seed-averaged effect counts.
- **No outcome licenses a claim about whether the agents "really" share meaning**,
  have inner states, or understand anything. E6's value is *conceptual, not
  confirmatory* (plan §4.6).

## 3. Result

Exemplary run (Haiku, 4 referents, 8-round blocks, 5 seeds per condition):

| condition | mean Δ (post − pre) |
|---|---|
| `reflect` | +0.20 |
| `control` | +0.25 |

**reflection_effect = −0.05**, within the null band (±0.15) → **`reflection_adds_a_story_only`.**
Both conditions improve over the intervention, and turning the agents onto their own
coordination confers **no advantage over a matched neutral interlude**. The
`reflect` agents produced fluent first-person statements about shared understanding
— and left the practice untouched. **The meta-example's irony, reproduced in the
third person and as a number:** the story appears; the coordination does not move.

(A single small run of this experiment drifted *outside* the band on one round of
noise; the seed-averaged estimate reverses to the robust null — precisely why the
metric is control-subtracted and seed-averaged, and why the transcripts are
excluded from scoring.)

## 4. What E6 does and does not license

From the frozen prereg (printed with every report):

> From inside the exchange, the deflationary and form-of-life readings are
> indistinguishable — exactly what §293 predicts and what E6 dramatizes by folding
> the observer into the observed. E6 does **not** adjudicate between them. It reports
> one thing only: whether an agent examining its own coordination changed that
> coordination relative to a matched control. A null effect is the irony reproduced
> — reflection adds a story, not a change. A non-null effect shows behavioral
> influence, nothing more. Reading the reflection transcripts as evidence of shared
> meaning is precisely the over-reading this pre-registration exists to forbid.

## 5. How to run (rich mode — costs API calls)

```bash
uv run python experiments/e6_reflexive/run.py -m intervention=reflect,control
uv run python -m beetlebox.analysis.e6 --reflect results/<reflect>/seed0 \
                                        --control results/<control>/seed0
# multiple seeds per condition are averaged:
uv run python -m beetlebox.analysis.e6 --reflect results/<r>/seed0 results/<r>/seed1 \
                                        --control results/<c>/seed0 results/<c>/seed1
```

The [notebook](../notebooks/e6_reflexive.ipynb) demonstrates the apparatus and the
control-subtracted metric with deterministic fake agents (no API), and reports the
committed live numbers.

## 6. References

- Wittgenstein, *Philosophical Investigations* §293 (the first-person predicament).
- "The Bewitching AI" (2025); "Mechanistic Indicators of Understanding" (2026) — the
  shadow-vs-understanding framing E6 turns on itself. See
  [`supplemental_reading.md`](supplemental_reading.md).
