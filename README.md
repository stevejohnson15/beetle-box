# Beetle-Box

*An experimental program using LLM agents and small from-scratch models to
**operationalize** — not to prove or refute — Ludwig Wittgenstein's arguments
about private language, meaning-as-use, rule-following, and forms of life.*

> The single most important commitment: **this project does not aim to prove or
> disprove Wittgenstein.** His claims are grammatical, not empirical. The aim is
> to turn his thought experiments into manipulable instruments — intuition pumps
> we can crank — where surprising agent behavior becomes an occasion for
> philosophical work, never a verdict.

The full research-design document lives in [`plan/beetle-box.md`](plan/beetle-box.md).
Read Sections 1–3 there before contributing: they contain the honesty
commitments (grammatical-not-empirical framing, the pretraining confound,
pre-registration discipline) that the code is built to enforce.

## Status

**All six experiments (E1–E6) are implemented** — each with harness code,
clean-room and/or rich (frontier) modes, a frozen pre-registration, an approach
doc, a working notebook, and a committed exemplary run; see
[`docs/README.md`](docs/README.md) for the index and
[`docs/supplemental_reading.md`](docs/supplemental_reading.md) for the background.

**Milestone 1 — Harness + E1 (clean-room signaling).** The roadmap's first
buildable step (`plan/beetle-box.md` §7): a multi-agent orchestration harness
with declarative configs, strict seeding, constrained communication channels,
structured replayable logging, and pre-registered scoring — exercised end-to-end
on **E1**, a Lewis/referential signaling game played by small from-scratch
PyTorch agents over an *invented* (non-natural-language) symbol channel.

**Milestone 2 — E3, the beetle-box (§293).** The centerpiece, built as a
*framework for exploration*: **three** selectable game designs
(`private_referent`, `sensation_matching`, `public_referent_aux`) × **four** box
conditions (`shared` / `divergent` / `empty` / `noise`), in both **clean-room**
(from-scratch neural) and **rich** (frontier Anthropic-API, in-context) modes.
The box is a real input, so "the beetle cancels out" is an *earnable* result, not
an assumption (§3.3). See [`docs/e3_design.md`](docs/e3_design.md) for the full
options catalog and trade-offs.

**E2 — the private diarist (§258).** A single agent names a private percept stream
with self-invented terms under *no external correction*; the **memory toggle** is
the central manipulation. It makes §258's dilemma a number: a full diary
manufactures perfect "same again" consistency (≈1.0) that no memory cannot (≈0.0)
— but the diary only ever checks the agent's own past impressions, so whether that
is a real check or a longer private impression is exactly what the number cannot
settle. Clean-room + rich modes. See [`docs/e2_diarist.md`](docs/e2_diarist.md).

**E4 — quus / rule-following (§ Kripke).** Two layers. *Behavioral:* seeded
students trained only on below-bend examples (where plus and quus agree)
extrapolate above the bend — converging toward plus under a representable encoding
(shared prior resolving the underdetermination) or diverging under a one-hot one.
*Mechanistic:* a from-scratch transformer **groks** modular addition, then its
Fourier circuit is read out and two seeds are compared — they need not implement
the same algorithm. The payload is the gap between behavioral underdetermination
and mechanistic (non-)determinacy. See [`docs/e4_quus.md`](docs/e4_quus.md).

**E5 — forms of life / grounding (§ capstone).** The same signaling game run
*ungrounded* (symbols are pure labels) vs. *grounded* (the receiver acts; payoffs
carry per-referent stakes). Grounding changes the language's character: it becomes
**selective** — encoding only the payoff-relevant distinction (selectivity gap 1.0,
lexicon 4/16) where bare identification encodes full identity (gap 0.0, lexicon
15/16) — evidence the form of life is load-bearing. See
[`docs/e5_forms_of_life.md`](docs/e5_forms_of_life.md).

**E6 — the reflexive layer (§ capstone / finale).** The meta-example made a
condition: two frontier agents coordinate, then are turned to examine whether they
share meaning; we measure whether that self-examination *changes the coordination*
(a control-subtracted, seed-averaged behavioral effect) rather than reading the
transcripts. The reflection produces fluent talk of shared understanding and leaves
the practice untouched — the irony of §1 reproduced as a number. Its frozen
pre-registration is the strictest in the project (transcripts are never evidence).
See [`docs/e6_reflexive.md`](docs/e6_reflexive.md).

A consolidated, per-experiment bibliography is in
[`docs/supplemental_reading.md`](docs/supplemental_reading.md).

Run E5:

```bash
uv run python experiments/e5_forms_of_life/run.py -m experiment=e5_grounded,e5_ungrounded
uv run python -m beetlebox.analysis.e5 results/<hash>/seed0
```

Run E4 (behavioral):

```bash
uv run python experiments/e4_quus/run.py -m quus=scalar,onehot
uv run python -m beetlebox.analysis.e4 results/<hash>/seed0
```

Run E2:

```bash
uv run python experiments/e2_diarist/run.py -m diarist=full,windowed,none
uv run python -m beetlebox.analysis.e2 results/<hash>/seed0
```

Run E3:

```bash
# clean-room condition sweep (centerpiece game)
uv run python experiments/e3_beetle_box/run.py -m box=shared,divergent,empty,noise
# score a sweep against the frozen pre-registration
uv run python -m beetlebox.analysis.e3 results/<hash>/seed0 ...   # one dir per condition
# rich mode (frontier agents; costs money — keep small)
uv run python experiments/e3_beetle_box/run.py mode=rich box=shared rich.num_rounds=10
```

## Install

Requires [`uv`](https://docs.astral.sh/uv/) and Python ≥ 3.11.

```bash
uv sync --extra dev            # library + test tooling
uv sync --extra notebooks      # add the demonstration-notebook tooling
```

## Development & testing

```bash
uv run pytest --cov            # full suite with the coverage gate (fail_under=90; ~96%)
uv run ruff check              # lint
# verify a demonstration notebook executes end-to-end:
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/e1_signaling.ipynb
```

All code is held to a documentation + test standard (module/public-API docstrings,
inline *why* comments, unit tests for every public behavior, and the coverage gate);
see [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Run E1

```bash
# Run the signaling game (writes results/<config_hash>/seed<seed>/)
uv run python experiments/e1_signaling/run.py seed=0

# Score a run against the FROZEN pre-registration
uv run python -m beetlebox.analysis.e1 results/<config_hash>/seed0
```

Manipulations (feedback on/off, channel bandwidth `V`/`L`, population size,
agent turnover) are Hydra config overrides and support `--multirun` sweeps.

## Repository layout

```
configs/            declarative experiment + condition configs (Hydra)
prereg/             pre-registered hypotheses & scoring thresholds (frozen, versioned)
src/beetlebox/
  config.py         dataclass config schema (config_hash delegated to reprolog)
  seeding.py        -> reprolog (re-export shim)
  harness/          experiment-specific run managers (built on agentharness)
  agents/           from-scratch neural sender/receiver; Agent/ApiAgent -> agentharness
  memory/           -> agentharness (re-export shim)
  channels/         -> emcomkit (re-export shim)
  envs/             experiment envs (percept/grounded/quus); SignalingEnv -> emcomkit
  runlog/           -> reprolog (re-export shim)
  analysis/         scoring; metrics -> emcomkit, prereg -> frozenprereg
  mech/             mechanistic sub-stack (E4 grokking transformer)
experiments/        one directory per experiment (E1–E6 implemented)
results/            run outputs, keyed by config hash + seed (gitignored)
```

## Extracted libraries

Four domain-agnostic components have been extracted into standalone, Apache-2.0,
PyPI-ready libraries; Beetle-Box now consumes them (pinned to their `v0.1.0`
GitHub tags via `[tool.uv.sources]` until they are published to PyPI). Each
attributes Beetle-Box as its origin.

| Library | What it provides |
|---|---|
| [reprolog](https://github.com/stevejohnson15/reprolog) | Reproducible run logging: deterministic seeding, config-hash run keys, append-only JSONL event/manifest logs. |
| [frozenprereg](https://github.com/stevejohnson15/frozenprereg) | Pre-registration + frozen scoring: load versioned criteria that must be fixed before a run. |
| [emcomkit](https://github.com/stevejohnson15/emcomkit) | Emergent-communication toolkit: invented symbol channels, referent generators, compositionality metrics, transcript formatters. |
| [agentharness](https://github.com/stevejohnson15/agentharness) | Multi-agent harness: framework-free `Agent`/`ChoiceAgent`, toggleable memory, a backend registry with an Anthropic frontier backend, and a generic run loop. |

Existing `beetlebox.*` import paths continue to work: the extracted modules are
thin re-export shims over the libraries.

## License & attribution

Beetle-Box is licensed under the **Apache License 2.0** — see [`LICENSE`](LICENSE)
and [`NOTICE`](NOTICE). It permits full reuse and requires only proper
attribution.

Several components (the multi-agent harness, the emergent-communication toolkit,
the pre-registration/scoring framework, and the reproducible run-logging layer)
have been extracted into **standalone open-source libraries** — see
[Extracted libraries](#extracted-libraries) above. Each remains Apache-2.0 and
attributes Beetle-Box as its origin.

Every source file carries an `SPDX-License-Identifier: Apache-2.0` header; please
keep these headers on new files. See [`CONTRIBUTING.md`](CONTRIBUTING.md).
