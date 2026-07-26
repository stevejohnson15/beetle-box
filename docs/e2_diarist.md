<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Beetle-Box contributors
-->
# E2 — The private diarist (*Philosophical Investigations* §258)

*The private-language argument, made into an instrument. A single agent names a
private percept stream with self-invented terms under **no external correction**,
and we watch whether "same again" stabilizes or wanders — and, if it stabilizes,
what the stability actually rests on.*

- **Approach (this doc)** · **Notebook:** [`../notebooks/e2_diarist.ipynb`](../notebooks/e2_diarist.ipynb) · **Exemplary run:** [`../results/exemplary/e2_diarist/`](../results/exemplary/e2_diarist/)
- **Frozen pre-registration:** [`../prereg/e2_diarist.yaml`](../prereg/e2_diarist.yaml)
- **Design document:** `plan/beetle-box.md` §4.2

---

## 1. The Wittgensteinian target

§258's diarist writes the sign "S" whenever a private sensation recurs. There is
no public criterion for "same again" — no one else can check, and there is nothing
outside the diarist's own impressions to check against. Wittgenstein's claim:
absent an independent check, there is no difference between *following the rule for
S* and merely *seeming to* — "whatever is going to seem right to me is right, and
that only means that here we can't talk about 'right'."

E2 turns this into a manipulable instrument. The sensations are a stream of percept
vectors, each drawn from one of `K` latent **types** (the God's-eye ground truth
the diarist never sees). The diarist emits a **term** for each percept, with no
reward and no correction. Then — using the God's-eye types as a ruler the diarist
lacks — we measure whether a type keeps getting the same term.

The point is **not** to refute Wittgenstein. It is that a **diary (memory)** can
*manufacture* the very stability §258 says is impossible without a check — which
forces his real question rather than settling it (see §5).

## 2. Setup

- **Percept stream** (`beetlebox.envs.percept.PerceptStream`): `K` types, each with
  a fixed prototype vector; a percept is `prototype[type] + noise`. Small noise →
  types are well separated and a recurrence is obvious; large noise → the clouds
  overlap and "same again" is genuinely ambiguous. Types are drawn i.i.d., so each
  recurs after random **time gaps**.
- **Diarist** (`beetlebox.agents.diarist`): because §258 has no training signal
  (there is nothing to be right or wrong against), the diarist is not a trained
  network but an explicit naming *policy*:
  - **`prototype`** (the centerpiece): match each percept against the diarist's
    stored past impressions (its diary); reuse the nearest one's term within a
    threshold, else coin a new term. Its **memory** is the manipulation.
  - **`fixed_quantizer`** (contrast): a deterministic percept→term rule with no
    diary — a stateless *public* criterion.
  - **`noisy_impression`** (contrast/floor): name by an in-the-moment stochastic
    impression with no persistence — "whatever seems right."

## 3. Manipulations

| Manipulation | Config | What it tests |
|---|---|---|
| **Memory access** (central) | `diarist=full` / `windowed` / `none` | Whether the diary manufactures "same again." Maps to `beetlebox.memory`: `none` = no diary, `windowed` = last W impressions, `full` = unbounded. |
| **Percept structure** | `percept=clean` / `noisy` | Clean vs. ambiguous "same again" (percept noise). |
| **Time gaps** | (emergent) | Does "same again" hold across long gaps since a type last appeared? |
| **Policy** | `diarist=fixed_quantizer` / `noisy_impression` | Stateless public rule vs. persistence-free impression, for contrast. |

## 4. Metrics (frozen in the pre-registration)

Scored by `beetlebox.analysis.e2` against [`prereg/e2_diarist.yaml`](../prereg/e2_diarist.yaml):

- **consistency** — for each type, the fraction of its occurrences given the type's
  modal term (averaged over types). "Does 'same again' hold?" 1.0 = always the same
  term; ~0 = renamed every time.
- **purity** — mirror image: for each term, the fraction of uses on its modal type.
  Low purity = a term is overloaded across types.
- **drift** — consistency on the first vs. second half of the run; a drop means the
  naming wandered over time.
- **gap-conditioned consistency** — consistency split by short vs. long gap since a
  type last appeared; a drop for long gaps is "same again" fading with time.

## 5. What E2 does and does not license

From the frozen prereg (printed with every scored report):

> This does **not** refute Wittgenstein. The interesting case is precisely that
> context-memory (the diary) can **manufacture** the stability §258 says is
> impossible without an independent check. That forces the real question rather
> than settling it: is the diary an **independent check**, or merely a **longer
> private impression** masquerading as one? The diarist checks its own past
> impressions against its present one — there is still no external criterion. The
> consistency number cannot tell these apart; report the dilemma, not a resolution.

Concretely: with a full diary, consistency ≈ 1.0; with no diary, ≈ 0.0. Memory
produced the entire difference. But the diary is the agent's own record of how
things *seemed* — so the stability it buys is exactly what §258 warns is
indistinguishable from a genuine check.

## 6. How to run

```bash
# Sweep the memory access (the central manipulation):
uv run python experiments/e2_diarist/run.py -m diarist=full,windowed,none
# Noisy "same again":
uv run python experiments/e2_diarist/run.py diarist=full percept=noisy
# Score a run against the FROZEN pre-registration:
uv run python -m beetlebox.analysis.e2 results/<config_hash>/seed0
```

Rich (frontier) mode — a pretrained model as the diarist, its diary shown or
withheld — lives in `beetlebox.harness.rich_e2` (costs API calls; keep small):

```python
from beetlebox.harness.rich_e2 import RichDiaristRunner
RichDiaristRunner(memory="full", num_steps=16, model="claude-haiku-4-5").run()
```

See the [notebook](../notebooks/e2_diarist.ipynb) for a live walkthrough and the
[exemplary run](../results/exemplary/e2_diarist/) for a fully-described sweep.

## 7. References

- Wittgenstein, *Philosophical Investigations* §§243–315, esp. the diarist at §258.
- Kripke, *Wittgenstein on Rules and Private Language* (1982) — the rule-following
  reading E2 sets up for E4.
