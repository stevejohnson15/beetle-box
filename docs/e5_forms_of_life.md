<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Beetle-Box contributors
-->
# E5 — Forms of life / grounding (capstone)

*The crux of the whole program: does meaning live in the public moves alone, or in
the way those moves are woven into a form of life? E5 takes E1's signaling game and
runs it **ungrounded** (symbols are pure labels) versus **grounded** (symbols drive
real consequences in a world with stakes) and asks whether grounding changes the
**character** of the emergent language.*

- **Approach (this doc)** · **Notebook:** [`../notebooks/e5_forms_of_life.ipynb`](../notebooks/e5_forms_of_life.ipynb) · **Exemplary run:** [`../results/exemplary/e5_forms_of_life/`](../results/exemplary/e5_forms_of_life/)
- **Frozen pre-registration:** [`../prereg/e5_forms_of_life.yaml`](../prereg/e5_forms_of_life.yaml)
- **Design document:** `plan/beetle-box.md` §4.5 · **Background:** [`supplemental_reading.md`](supplemental_reading.md)

---

## 1. The target

For Wittgenstein, "use" is not bare statistics of word-succession; it is use woven
into the natural history of a creature — pain-behavior, care for the injured, the
whole surrounding activity (*PI* §§19, 23, 241). This is the fault line the LLM
question keeps hitting: is a language-game individuated by its **public moves
alone** (in which case a model that makes the moves is in), or by its **embedding
in a form of life** (in which case the model has a convincing shadow of the game,
not the game)?

E5 gives that fault line an operational handle. It cannot settle the philosophy,
but it can ask a sharp, measurable question: **if the same signals are made to
drive real consequences, does the language that emerges have a different
character?** If yes, the form of life is load-bearing — the surrounding activity,
not the tokens alone, is shaping what the language marks.

## 2. Setup

The same signaling game as E1 (sender sees a referent, emits an invented message;
`beetlebox.harness.e5_manager`), run in two regimes:

- **Ungrounded** (`grounded=False`): reward = **identify the referent** (E1). The
  symbols are pure labels; they must distinguish all referents equally.
- **Grounded** (`grounded=True`): the receiver chooses an **action**, and the
  reward is the world **payoff** of that action for the true referent, with
  per-referent **stakes** (`beetlebox.envs.grounded.GroundedWorld`) — a minimal
  resource/survival task. The correct action depends only on the referent's
  *relevant* attribute (attribute 0); the *irrelevant* attribute (attribute 1) is
  payoff-neutral but sets the **stakes** (a mistake on a high-stakes object costs
  more). Words now drive differential consequences.

Everything else is held fixed: the sender, the invented channel, and the referent
grid are identical across regimes. Only the receiver's output space (actions vs.
referent identity) and the reward differ — so any change in the language is
attributable to the grounding.

## 3. Metrics (frozen in the pre-registration)

The sharpest handle is **selectivity** — does the language mark only what matters?
Computed by `beetlebox.analysis.e5` from the greedy referent→message map:

- **relevant / irrelevant recoverability** — for each attribute, the fraction of
  referents whose value is pinned down by their message. Full identity ⇒ both high;
  a selective code ⇒ relevant high, irrelevant low.
- **selectivity_gap** = relevant − irrelevant recoverability. The headline number.
- **lexicon_size** — distinct messages used.

Reported alongside (secondary): native **performance** (identification accuracy or
normalized payoff), **robustness** (performance retained under channel noise), and
**transfer** (performance after a fresh receiver is swapped in — turnover).

## 4. Result

Exemplary run (grid: 2 attributes × 4 values = 16 referents; channel V=6, L=2):

| regime | performance | lexicon | relevant recover. | irrelevant recover. | selectivity gap |
|---|---|---|---|---|---|
| **ungrounded** | 0.94 (id acc) | 15/16 | 0.88 | 0.88 | **0.00** |
| **grounded** | 1.00 (payoff) | 4–5/16 | 1.00 | 0.00 | **1.00** |

The contrast is stark. The **ungrounded** language encodes **full identity** — a
near-unique message per referent, both attributes recoverable. The **grounded**
language is **selective** — a handful of messages, one per payoff-relevant value,
with the irrelevant attribute *completely discarded*. Same referents, same channel;
the consequential task reshaped what the language marks.

**Both prereg checks pass:** `grounding_is_selective` (gap ≥ 0.5) and
`encodes_full_identity` (gap ≤ 0.2 with high recoverability).

## 5. What E5 does and does not license

From the frozen prereg (printed with every scored report):

> If grounding measurably changes the character of the emergent language — here, by
> making it **selective** where bare identification encodes full identity — that is
> evidence the **form of life is load-bearing**: the surrounding activity, not the
> tokens alone, shapes what the language means. This is the sharpest operational
> handle on the LLM fault line (public moves vs. embedding in a form of life). It
> does **not** show the agents "understand" anything, nor prove an ungrounded
> language could not be pushed to behave identically under a different pressure; it
> shows that under this grounding the language's structure is different, and why.
> Report the change in character, not a verdict on understanding.

## 6. How to run

```bash
uv run python experiments/e5_forms_of_life/run.py -m experiment=e5_grounded,e5_ungrounded
uv run python -m beetlebox.analysis.e5 results/<config_hash>/seed0
```

See the [notebook](../notebooks/e5_forms_of_life.ipynb) for a live walkthrough and
the [exemplary run](../results/exemplary/e5_forms_of_life/) for the described sweep.

## 7. References

- Wittgenstein, *Philosophical Investigations* on forms of life (§§19, 23, 241);
  meaning-as-use woven into the natural history of a creature.
- "Mechanistic Indicators of Understanding in Large Language Models," *Philosophical
  Studies* (2026). See [`supplemental_reading.md`](supplemental_reading.md) for the
  full, annotated list.
