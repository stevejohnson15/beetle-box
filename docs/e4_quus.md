<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Beetle-Box contributors
-->
# E4 — Quus and the teachable rule (rule-following + interpretability)

*Kripke's rule-following skepticism, made manipulable — and the spine linking
Beetle-Box to the grokking / mechanistic-interpretability literature. E4 has two
layers, and its payload is the **gap between them**: behavioral underdetermination
above, mechanistic (non-)determinacy below.*

- **Approach (this doc)** · **Notebook:** [`../notebooks/e4_quus.ipynb`](../notebooks/e4_quus.ipynb) · **Exemplary run:** [`../results/exemplary/e4_quus/`](../results/exemplary/e4_quus/)
- **Frozen pre-registration:** [`../prereg/e4_quus.yaml`](../prereg/e4_quus.yaml)
- **Design document:** `plan/beetle-box.md` §4.4, §9 (references)

---

## 1. The target

Kripke's ``quus`` operator agrees with ``plus`` on every pair anyone has ever
computed and diverges only past a **bend-point** ``k``:

```
a quus b = a + b        if max(a, b) < k
         = quus_value    otherwise
```

If your evidence is a finite set of sums all below the bend, nothing in it settles
whether you have been following ``plus`` or ``quus`` — or any of infinitely many
other rules that agree there. "Memorizing the table" and "grasping the rule" are
behaviorally indistinguishable on the data seen. That is the skeptical point.

E4 makes it manipulable in two layers.

## 2. Layer 1 — Behavioral: which rule gets extrapolated?

**Setup** (`beetlebox.envs.quus`, `beetlebox.harness.e4_manager`). Train several
seeded students only on the below-bend pairs (where plus = quus), then read out
what each predicts on the above-bend pairs, where the rules finally disagree.

**The manipulation is the operand encoding** — it decides whether a shared prior
can even reach across the bend:

- **`scalar`**: operands are given as their values, so the function is
  representable past the bend and a simplicity prior can extend ``plus``.
- **`onehot`**: each operand value is an independent input unit, so above-bend
  values are unconstrained by anything in training.

**Metrics** (`beetlebox.analysis.e4`): `plus_rate` (fraction of above-bend pairs a
student extrapolates as plus), `quus_rate`, and `cross_seed_agreement` (fraction of
pairs on which all students agree).

**Result** (exemplary run, 8 seeds):

| encoding | plus_rate | cross-seed agreement | reading |
|---|---|---|---|
| `scalar` | **0.73** | **0.63** | a shared inductive prior converges toward plus |
| `onehot` | **0.00** | **0.21** | students recover neither plus nor each other — underdetermination, unresolved |

Both faces the plan describes, toggled by one config: **divergence makes the
underdetermination visible; convergence shows shared inductive priors quietly
resolving it.** In **rich mode** (`beetlebox.harness.rich_e4`) a frontier model —
carrying the strongest such prior, ordinary arithmetic as an inherited form of
life — extrapolates plus at rate ≈1.0 on the same underdetermined examples.

## 3. Layer 2 — Mechanistic: is the algorithm determinate?

**Setup** (`beetlebox.mech`). Train a small **from-scratch transformer** on
``(a, b) → (a + b) mod p`` with weight decay — the **grokking** regime. It first
*memorizes* the training pairs (train accuracy saturates, val accuracy at chance),
then much later *generalizes* (val accuracy jumps), compressing onto a crisp
circuit. We record the accuracy curve and a description-length proxy (parameter
norm), then read the learned circuit out of the weights directly.

**Circuit read-out** (`beetlebox.mech.circuits`). A grokked network implements
modular addition with a **Fourier construction** — the token embeddings become
sinusoidal at a few **key frequencies** (Nanda et al.'s "Clock"). We extract the
embedding power spectrum, the dominant frequencies, and — across two models trained
on identical data — their **algorithmic agreement** (overlap of key-frequency
sets). "The Clock and the Pizza" reports that two such models often implement
*different* algorithms.

**Result** (exemplary run, p=53, two seeds): both seeds grok — train accuracy
saturates by ~step 200 while val accuracy lags for thousands of steps then climbs
to ~1.0 (seed 0: 0.998, seed 1: 0.996), as the parameter norm (a description-length
proxy) falls (43.5 → 30.7). Yet each learns a *different* sparse Fourier circuit:
seed 0's key frequencies are **{17, 8, 5}**, seed 1's are **{2, 22, 5}** — they
share only one, a **cross-seed algorithmic agreement of 0.20**. Two models on
identical data, both perfectly correct, implement different algorithms. See the
[exemplary run](../results/exemplary/e4_quus/) for the curves, spectra, and report.

> The mechanistic layer is heavier compute than the rest of Beetle-Box (a real grok
> is ~25k steps). It is a **launch-it-deliberately** job, like rich mode; unit tests
> use tiny/fast configs, and the committed exemplary run is a real grok.

## 4. What E4 does and does not license

From the frozen prereg (printed with every scored report):

> This does **not** resolve Kripke's skepticism. Convergence toward plus does not
> show the students "grasp plus" — it shows a shared inductive prior selecting one
> of the infinitely many rules the data allows; the prior is a longer finite fact,
> not an independent criterion. The payload is the **contrast** between the
> behavioral and mechanistic layers: even opening the box, two models on identical
> data need not implement the same algorithm. **Opening the box does not dissolve
> the skepticism — the circuit is a longer finite fact — but it relocates the debate
> onto new, examinable ground.** State exactly that.

## 5. How to run

```bash
# Behavioral layer:
uv run python experiments/e4_quus/run.py -m quus=scalar,onehot
uv run python -m beetlebox.analysis.e4 results/<config_hash>/seed0

# Mechanistic layer (heavy; launch deliberately):
python -c "from beetlebox.config import GrokkingConfig; from beetlebox.mech import GrokkingRun; \
           print(GrokkingRun(GrokkingConfig(modulus=53, num_steps=25000)).run()['final_val_acc'])"
```

See the [notebook](../notebooks/e4_quus.ipynb) for the behavioral layer live plus a
fast partial-grok demonstration of the mechanistic machinery.

## 6. References

- Kripke, *Wittgenstein on Rules and Private Language* (1982).
- Power et al., "Grokking" (2022); Nanda et al., "Progress Measures for Grokking
  via Mechanistic Interpretability" (2023); "The Clock and the Pizza" (2023);
  DeMoss et al., "The Complexity Dynamics of Grokking" (2024). See `plan/beetle-box.md` §9.
