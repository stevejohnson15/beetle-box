# Contributing to Beetle-Box

Thanks for your interest. A few conventions keep this project honest and reusable.

## Licensing & attribution

- All contributions are made under the **Apache License 2.0** (see `LICENSE`).
- Every new source file must begin with the header:

  ```python
  # SPDX-License-Identifier: Apache-2.0
  # Copyright (c) 2026 Beetle-Box contributors
  ```

- If you extract a component into a standalone library, keep it Apache-2.0 and
  attribute Beetle-Box as its origin (retain the `NOTICE`).

## The honesty guardrails (non-negotiable)

Read `plan/beetle-box.md` §3 before touching experiment code:

- **Grammatical, not empirical.** No experiment "proves/disproves" Wittgenstein.
  Keep result language operational.
- **Pre-register first.** Hypotheses, conditions, and scoring thresholds go in
  `prereg/` under version control *before* running. Analysis code imports only
  frozen prereg criteria — never inline a threshold in `analysis/`.
- **No post-hoc reading of intention into logs.** What counts as "coordination"
  or "the same term" is decided in advance.
- **Strict seeding & reproducibility.** Every run is reproducible from a
  `seed` + config; results are keyed by config hash + seed.
- **Clean-room vs. rich mode** is a config axis, never a code fork; never let a
  rich-mode result masquerade as a clean-room one.

## Development

```bash
uv sync --extra dev
uv run pytest --cov        # run tests with the coverage gate
uv run ruff check
```

Library code (`harness/`, `channels/`, `runlog/`, `analysis/`) must stay
Hydra-independent — it consumes plain dataclass configs so it can be extracted
later. Hydra is confined to `experiments/*/run.py`.

## Documentation & testing standard (applies to all code)

Every change is held to this bar — it is not optional polish:

**Documentation**
- **Module docstring** on every module: what it is and where it fits in the harness.
- **Docstring on every public class and function** (private helpers optional). State
  args/returns/raises when they aren't obvious from the signature.
- **Inline comments explain the *why*** for any non-obvious step — especially the
  design choices that keep the experiments honest (e.g. the no-leak wiring, the
  earnable-cancellation guard). Comment the reasoning, not the syntax.
- **Prose docs** stay in sync: the project `README.md`, `docs/` (the per-experiment
  pattern above), and this file. Markdown files carry the SPDX HTML-comment header.

**Testing**
- **Unit tests for every public behavior** where meaningful: happy path, the
  documented error/guard paths, and determinism/seeding where relevant.
- **Pure logic is tested directly.** Code that wraps an external service (the
  frontier API) is made testable by **dependency injection or mocking** so its
  logic is covered without a network call (see `tests/test_api_model.py`,
  `tests/test_rich_e3.py`). Live/API-only lines get `# pragma: no cover`.
- **Coverage gate:** `uv run pytest --cov` must pass (`fail_under = 90`, currently
  ~96%). New code is expected to hold or raise the line — don't lower the gate.
- **Notebooks are verified by executing them** (0 cell errors), not by unit tests.

**Hygiene**
- `ruff check` clean; SPDX header on every source file.
- Remove dead code rather than leaving it untested.

## Documentation pattern (every experiment)

Each experiment ships three things — an approach doc, a working notebook, and
(where appropriate) a committed exemplary run. The convention and the running
instructions are in [`docs/README.md`](docs/README.md); follow it for E1–E6 and
anything added later. New demonstration notebooks must execute end-to-end
(`uv run jupyter nbconvert --to notebook --execute --inplace notebooks/<nb>.ipynb`).

## Documentation pattern (every experiment)

Each experiment ships three things — an approach doc, a working notebook, and
(where appropriate) a committed exemplary run. The convention and the running
instructions are in [`docs/README.md`](docs/README.md); follow it for E1–E6 and
anything added later. New demonstration notebooks must execute end-to-end
(`uv run jupyter nbconvert --to notebook --execute --inplace notebooks/<nb>.ipynb`).
