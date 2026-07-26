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

- **Notebook:** [`../notebooks/e3_beetle_box.ipynb`](../notebooks/e3_beetle_box.ipynb) · **Exemplary run:** [`../results/exemplary/e3_beetle_box/`](../results/exemplary/e3_beetle_box/)
- **Frozen pre-registration:** [`../prereg/e3_beetle_box.yaml`](../prereg/e3_beetle_box.yaml) (v2 — see the design-history note under `private_referent`)

See also: `plan/beetle-box.md` §4.3 (E3 target), §3.3 (earnable cancellation).

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
`r`). It emits a public invented symbol; the receiver must pick `r` out of **all
candidate referents** — each represented only by the receiver's own box for it —
using the message. This is a **discrimination game**: no single box tells the
receiver "this is the target," so the public message is *forced* to carry the
signal.

```
r ~ referents (God's-eye)
box_S = code_S(r)                         sender sees ONLY its box
msg   = Sender(box_S)                      invented symbol channel
guess = Receiver(msg, [box_R(c) for c])    pick target among candidates
reward = (guess == r)
```

- **Beetle-box result** = `acc(shared) − acc(divergent)`. ≈0 ⇒ the private *form*
  is idle (**the beetle drops out**); >0 ⇒ a shared inner state helps
  (**against strong deflation**).
- **Earnable** because `empty`/`noise` collapse to chance — the box demonstrably
  carries the signal.
- **No-leak guard (mandatory).** `E3RunManager.channel_ablation()` verifies the
  public message is load-bearing: with the message zeroed, accuracy must fall to
  chance. See the design-history note below.
- **Observed (clean-room, K=8):** shared ≈1.00, divergent ≈0.87, empty/noise
  ≈chance; message-zeroed ≈chance in every condition. Reading: informative boxes
  are required (earnable), and the *shared* code is easier than *divergent* (which
  forces the receiver to learn a cross-code translation) — the private form
  affects learning, not the ceiling. Run it to see the current verdict; the
  transcript makes the exchange visible.

> **Design-history note — the receiver-side leak (fixed).** An earlier version of
> `private_referent` handed the receiver *its own box for the target referent* and
> asked it to classify `r` directly. Because that box encodes `r`, the receiver
> learned to read its own box and **ignored the message entirely** — a channel
> ablation showed message-zeroed accuracy stayed at 100%, i.e. no communication
> was happening and "the beetle drops out" was not *earned*, it was rigged (the
> §3.3 hazard). The discrimination reformulation above removes the leak, and
> `channel_ablation()` is kept as a permanent guard so it cannot recur. This is
> exactly the kind of operationalization failure the project exists to catch — see
> `plan/beetle-box.md` §3.3 and §8.

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

Standard referential game — the referent is **public** to the sender (it sees the
referent's features) — plus a private box as an extra side-channel **to the sender
only**. The receiver decodes the referent from the message alone (no receiver box,
so no leak).

- **Trade-off (weakest wiring):** the sender can solve the task through the public
  feature channel and ignore its box, so accuracy stays high across box conditions.
  This makes the earnable-cancellation check inapplicable and the centerpiece claim
  unconvincing. Included deliberately as a **control**: it shows what "the box
  doesn't matter" looks like when the box genuinely isn't needed.
- **Observed (clean-room):** high across conditions (box ignorable).

---

## Clean-room vs. rich (frontier) mode

Both arms run E3 (plan §3.2); never let one masquerade as the other.

- **Clean-room** (`mode=clean`, default): from-scratch PyTorch agents, invented
  channel — minimal inherited meaning. All three games, all four conditions.
- **Rich** (`mode=rich`): frontier Anthropic-API agents coordinate *in context*
  over rounds with feedback (`private_referent` implemented). Studies how
  pretrained agents *redeploy* inherited concepts. **Costs money** (~2×rounds API
  calls); defaults are small; model configurable (`rich.model`).

> ⚠️ **Known limitation — the rich `private_referent` runner still uses the older
> (leaky) framing:** the receiver is told its own sensation code for the target and
> asked to name it, so over rounds it can learn to read its own box rather than the
> message (the same leak the clean-room game had before the discrimination fix). The
> earlier frontier sweep (shared 0.83 / divergent 0.88 / empty 0.25) is therefore
> **not a trustworthy communication result** and should be treated as a
> not-yet-corrected data point. Porting the discrimination design to the in-context
> runner (and re-running) is a tracked follow-up; until then, rely on the clean-room
> arm for the beetle-box claim.

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
