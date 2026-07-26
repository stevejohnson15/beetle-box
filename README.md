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

Experiments E4, E5, E6 are scaffolded but not yet implemented.

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
  config.py         dataclass config schema + stable config_hash (Hydra-independent)
  seeding.py        deterministic global/per-run seeding
  harness/          multi-agent turn loop + run manager        [extraction candidate]
  agents/           Agent ABC; from-scratch neural sender/receiver
  memory/           toggleable per-agent memory (interface; used fully in E2)
  channels/         invented-vocabulary communication channels  [extraction candidate]
  envs/             signaling-game environment + referent generators
  runlog/           structured, replayable JSONL run logs        [extraction candidate]
  analysis/         scoring; imports only frozen prereg criteria [extraction candidate]
  mech/             mechanistic sub-stack placeholder (E4)
experiments/        one directory per experiment (E1 implemented)
results/            run outputs, keyed by config hash + seed (gitignored)
```

## License & attribution

Beetle-Box is licensed under the **Apache License 2.0** — see [`LICENSE`](LICENSE)
and [`NOTICE`](NOTICE). It permits full reuse and requires only proper
attribution.

Several components (the orchestration harness, the emergent-communication
channel toolkit, the pre-registration/scoring framework, and the reproducible
run-logging layer) are written behind clean module boundaries so they can later
be extracted as **standalone open-source libraries**. Any such spun-out library
remains Apache-2.0 and attributes Beetle-Box as its origin.

Every source file carries an `SPDX-License-Identifier: Apache-2.0` header; please
keep these headers on new files. See [`CONTRIBUTING.md`](CONTRIBUTING.md).
