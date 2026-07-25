<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Beetle-Box contributors
-->
# E1 — Convention from use (the Lewis signaling game)

*Meaning-as-use in its thinnest form. E1 is the warm-up and the baseline: the
simplest game in which a shared, stable convention can **emerge from use alone**,
with no meanings assigned in advance. Every later experiment perturbs this
baseline, so it is worth getting exactly right.*

- **Approach (this doc)** · **Notebook:** [`../notebooks/e1_signaling.ipynb`](../notebooks/e1_signaling.ipynb) · **Exemplary run:** [`../results/exemplary/e1_signaling/`](../results/exemplary/e1_signaling/)
- **Frozen pre-registration:** [`../prereg/e1_signaling.yaml`](../prereg/e1_signaling.yaml)
- **Design document:** `plan/beetle-box.md` §4.1 (E1), §7 (roadmap: E1 first)

---

## 1. The Wittgensteinian target

Wittgenstein's slogan *meaning is use* says the meaning of a word lives in the
public practice of using it — the "language-game" — not in a private object it
supposedly names. A [**Lewis signaling game**](https://en.wikipedia.org/wiki/Lewis_signaling_game)
(David Lewis, *Convention*, 1969) is the barest possible arena for that claim: a
speaker and a hearer must coordinate on the use of signals that start out
**meaningless**, and any meaning the signals acquire is *constituted by* the
coordinated practice, nothing else.

E1 asks the thin question first: **can meaning-as-use happen at all, from
scratch, between agents given no prior conventions?** If it can, we have a
running instrument — a language-game we built and can now perturb. If it can't,
nothing downstream is worth attempting.

E1 deliberately says **nothing** about "understanding," inner states, or forms
of life. Those are the subject of E3 (§293) and E5 (grounding). E1 establishes
only the thin thesis and the baseline the others move.

## 2. Setup

A **sender** sees a referent (one of `K` objects) and emits a **message** drawn
from a constrained, non-natural-language symbol set. A **receiver** sees only the
message and must identify the referent. When it succeeds, both are rewarded. No
meanings are assigned in advance; the mapping *referent → message → referent* has
to be invented jointly, through repeated play.

```
r ~ referents (uniform)            God's-eye ground truth
message = Sender(features(r))      invented symbols, no English
guess   = Receiver(message)
reward  = 1 if guess == r else 0   shared success signal
```

Two design choices make this a **clean-room** study of emergence (`plan/beetle-box.md` §3.2):

- **Invented channel.** Messages are integer symbols from a vocabulary of size
  `V`, in fixed-length strings of length `L` (bandwidth `V^L`). There are no
  English tokens, so no inherited meaning is smuggled in — the convention must be
  built, not recalled. (`beetlebox.channels.SymbolChannel`)
- **From-scratch agents.** Sender and receiver are small PyTorch networks
  initialized from random weights (`beetlebox.agents.NeuralSender` /
  `NeuralReceiver`). The sender is a policy trained by **REINFORCE** (its message
  is a discrete sample); the receiver is a classifier trained by
  **cross-entropy** to reconstruct the referent. They share the success reward.

The referent space is configurable (`beetlebox.envs.SignalingEnv`):

- `flat` — `K` unstructured one-hot objects (the baseline).
- `grid` — an *attribute × value* grid, so referents carry a distance structure.
  This is what makes **compositional** structure measurable (see §4).

## 3. Manipulations (independent variables)

All are Hydra config overrides and support `--multirun` sweeps.

| Manipulation | Config | What it tests |
|---|---|---|
| **Feedback vs. none** | `experiment=e1_default` / `e1_ablation` | Whether the *game* (not a leak) drives coordination. Without a success signal, no learning occurs and accuracy must stay at chance. |
| **Agent turnover** | `experiment=e1_turnover` | Whether a convention **survives its founders**: freeze the converged sender, drop in a fresh receiver mid-run, and see if the incumbent code is re-adopted. A language outliving its speakers is meaning-as-public-practice made visible. |
| **Channel bandwidth** | `channel=small` / `wide` (`V`, `L`) | How vocabulary size and message length shape what conventions are expressible. |
| **Referent structure** | `env=flat8` / `grid` | Whether structured referents + `L>1` give rise to compositional codes. |

## 4. Metrics (frozen in the pre-registration)

Scoring uses only the thresholds in [`prereg/e1_signaling.yaml`](../prereg/e1_signaling.yaml)
— fixed before running, never tuned to a result (`plan/beetle-box.md` §3.2).
Computed by `beetlebox.analysis.metrics` and applied by `beetlebox.analysis.e1`.

- **Emergence rate.** Greedy communication accuracy over all referents, compared
  to chance `1/K`. Emergence = accuracy well above chance (`emergence_accuracy`,
  `emergence_ratio_over_chance`).
- **Convention stability.** Agreement of the greedy *referent → message* map
  across the final evaluations. 1.0 means the convention froze; low values mean
  it was still wandering.
- **Transmission fidelity across turnover.** Post-turnover accuracy relative to
  the pre-turnover level — did the fresh agent re-adopt the convention?
- **Topographic similarity.** Spearman correlation between pairwise *referent*
  distance and pairwise *message* distance. High ρ means similar referents get
  similar messages — a signature of **compositional** structure. Reported only in
  `grid` mode with `L>1`, where referent distance is meaningful; elsewhere it is
  explicitly *not applicable* rather than a spurious number.

## 5. What E1 does and does not license

From the frozen prereg (printed with every scored report):

> Demonstrates the thin meaning-as-use thesis and establishes the baseline that
> later experiments perturb. It does **not**, by itself, say anything about
> "understanding," inner states, or whether the agents share a form of life. A
> stable convention here is public practice, nothing more. (`plan/beetle-box.md` §4.1)

The ablation is the guardrail: if the no-feedback condition also coordinated, the
result would be an artifact (a leak), not emergence from use. It must sit at
chance for the emergence result to mean anything.

## 6. How to run

```bash
uv sync --extra dev

# Baseline: convention emerges from feedback-driven use
uv run python experiments/e1_signaling/run.py seed=0

# Score the run against the FROZEN pre-registration
uv run python -m beetlebox.analysis.e1 results/<config_hash>/seed0

# Conditions
uv run python experiments/e1_signaling/run.py experiment=e1_ablation      # no feedback -> chance
uv run python experiments/e1_signaling/run.py experiment=e1_turnover      # survives its founders
uv run python experiments/e1_signaling/run.py env=grid channel=wide       # compositional structure

# Sweep seeds / conditions
uv run python experiments/e1_signaling/run.py -m seed=0,1,2 experiment=e1_default,e1_ablation
```

Runs are reproducible from `seed` + config and land under
`results/<config_hash>/seed<seed>/` (`events.jsonl` + `manifest.json`). The
config hash excludes `seed`/`device`/`output_dir`, so one condition's seeds group
under a single hash.

For a live, narrated walkthrough see the [notebook](../notebooks/e1_signaling.ipynb);
for a fully described representative run see the
[exemplary run](../results/exemplary/e1_signaling/).

## 7. References

- David Lewis, *Convention: A Philosophical Study* (1969) — the signaling game.
- Wittgenstein, *Philosophical Investigations* — meaning-as-use (§§1–43 and
  throughout); §7 introduces the "language-game."
- Emergent-communication background: `plan/beetle-box.md` §9 (LLM-Language-Games
  and the emergent-communication literature).
