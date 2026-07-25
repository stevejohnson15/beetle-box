<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Beetle-Box contributors
-->
# E3 — The Beetle-Box: design options catalog

*Philosophical Investigations §293, operationalized. This document catalogs the
**three game designs** and **four box conditions** Beetle-Box implements for E3.
Per the project's exploration-first stance, all options are buildable and
selectable — none is privileged as "the" design. Choose per the question you are
asking, and read the trade-offs before reporting a result.*

See also: `plan/beetle-box.md` §4.3 (E3 target), §3.3 (earnable cancellation),
`prereg/e3_beetle_box.yaml` (frozen hypotheses & thresholds).

---

## The shared idea

Every agent holds a private **box**. On each trial its box yields a private inner
signal only it can see. **We** control the boxes, so "are two agents' beetles the
same?" has a God's-eye answer the agents never get. The box is delivered as a real
network input — so it *could* drive behavior. Whether it does, and whether its
private *form* matters, is then measured.

### The four box conditions (`beetlebox.boxes.BoxScheme`)

| Condition | What the box contains | §293 role |
|---|---|---|
| `shared` | Every agent encodes the God's-eye state with the **same** code | A literally shared inner state (comparable across agents) |
| `divergent` | Each agent encodes it with its **own** private code | The literal beetle-box: same game, incomparable private forms |
| `empty` | A constant (carries nothing) | No beetle at all |
| `noise` | Random, independent of the state | A "beetle" that does no work |

**Earnable cancellation (§3.3).** For any claim that "the beetle cancels out" to
be *found* rather than rigged, informative boxes (`shared`/`divergent`) must be
*able* to beat empty ones — i.e. the box can influence output. The analysis
enforces this as a pre-registered check for the box-only games below.

---

## The three games (config: `experiment.game`)

### 1. `private_referent` — the centerpiece (recommended)

The sender's **only** access to the referent `r` is its box (a private code for
`r`). It emits a public invented symbol; a box-aware receiver identifies `r`.

```
r ~ referents (God's-eye)
box_S = code_S(r)          sender sees ONLY its box
msg   = Sender(box_S)      invented symbol channel
guess = Receiver(msg, box_R)
reward = (guess == r)
```

- **Beetle-box result** = `acc(shared) − acc(divergent)`. ≈0 ⇒ the private *form*
  is idle (**the beetle drops out**); >0 ⇒ a shared inner state helps
  (**against strong deflation**).
- **Earnable** because `empty`/`noise` collapse to chance — the box demonstrably
  carries the signal.
- **Trade-off:** the cleanest, most literal §293 model; the receiver's own box is
  the channel through which "shared" could beat "divergent".
- **Observed (clean-room, K=8):** shared 1.00, divergent 1.00, empty ≈0.12,
  noise ≈0.13 → *beetle drops out*, earned.

### 2. `sensation_matching` — reporting sensations

Two symmetric agents each privately sense a **type**; the God's-eye label says
whether the two sensations are the *same type*. Each emits a public word; each
judges same/different from its own box + the partner's word.

```
t_A, t_B ~ types;  y = (same type?)     (God's-eye)
m_A = A.speak(box_A);  m_B = B.speak(box_B)
reward = A.judge(box_A, m_B)==y  AND  B.judge(box_B, m_A)==y
```

- Closest to "reporting a sensation," but the *same across agents* ground truth is
  **experimenter-imposed** (a philosophical caveat: we define "same beetle").
- **Trade-off:** richest game; the shared-vs-divergent gap tends to be *positive*
  and small — a different verdict from `private_referent` on the same apparatus.
- **Observed (clean-room):** shared ≈0.95, divergent ≈0.89, empty/noise ≈0.50 →
  *shared inner state measurably helps* (chance 0.50).

### 3. `public_referent_aux` — auxiliary box (contrast/control)

Standard referential game — the referent is **public** to the sender — plus a
private box as an extra side-channel.

- **Trade-off (weakest wiring):** the network can solve the task through the public
  channel and ignore the box, so `empty`/`noise` stay *high*. This makes the
  earnable-cancellation check inapplicable and the centerpiece claim unconvincing.
  Included deliberately as a **control**: it shows what "the box doesn't matter"
  looks like when the box genuinely isn't needed.
- **Observed (clean-room):** all conditions high (empty ≈0.87, noise ≈0.90).

---

## Clean-room vs. rich (frontier) mode

Both arms run E3 (plan §3.2); never let one masquerade as the other.

- **Clean-room** (`mode=clean`, default): from-scratch PyTorch agents, invented
  channel — minimal inherited meaning. All three games, all four conditions.
- **Rich** (`mode=rich`): frontier Anthropic-API agents coordinate *in context*
  over rounds with feedback (`private_referent` implemented). Studies how
  pretrained agents *redeploy* inherited concepts. **Costs money** (~2×rounds API
  calls); defaults are small; model configurable (`rich.model`).

---

## How to run

```bash
# Clean-room sweep over the four conditions (centerpiece game):
python experiments/e3_beetle_box/run.py -m box=shared,divergent,empty,noise

# A different game:
python experiments/e3_beetle_box/run.py experiment=sensation_matching box=divergent

# Rich mode (frontier; keep small — costs money):
python experiments/e3_beetle_box/run.py mode=rich box=shared rich.num_rounds=10

# Score a condition sweep against the frozen prereg:
python -m beetlebox.analysis.e3 results/<hashShared>/seed0 results/<hashDivergent>/seed0 \
                                results/<hashEmpty>/seed0 results/<hashNoise>/seed0
```

---

## Reporting discipline

The analysis prints the prereg's *licenses* text with every report. Keep to it:
E3 makes §293 **a number**; it does **not** settle whether the agents have inner
states or what they are like. The four conditions are our God's-eye control — the
agents never get one. That asymmetry is the point, not a limitation.
