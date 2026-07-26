<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Beetle-Box contributors
-->
# Supplemental Reading

*An academic collection of the background materials needed to understand and extend
Beetle-Box, organized by experiment. Beetle-Box **operationalizes** Wittgenstein's
arguments as manipulable instruments — it does not prove or refute them (see
`plan/beetle-box.md` §3). These readings supply the philosophical claims each
experiment is built to probe and the technical literature each method draws on.*

Primary philosophical texts are best read in hardcopy; each entry notes **why it
matters here** and, where relevant, **how to extend** the framework with it.

**Contents:** [Foundational](#foundational-cross-cutting) · [E1](#e1-convention-from-use-signaling) ·
[E2](#e2-the-private-diarist-258) · [E3](#e3-the-beetle-box-293) ·
[E4](#e4-quus-rule-following--interpretability) · [E5](#e5-forms-of-life--grounding) ·
[E6](#e6-the-reflexive-layer) · [Methodology & cross-cutting technical](#methodology--cross-cutting-technical)

---

## Foundational (cross-cutting)

These underlie the whole program; every experiment leans on them.

- **Wittgenstein, L. — *Philosophical Investigations*.** The source text. Key
  regions: **rule-following §§138–242**; **private language §§243–315**; the
  **diarist §258**; the **beetle-box §293**; **meaning-as-use** (§§1–43 and
  throughout, incl. the "language-game" at §7); **forms of life** (§§19, 23, 241).
  *Why it matters:* the grammatical claims each experiment turns into a manipulable
  instrument. *Guardrail:* the claims are grammatical, not empirical — read §3 of
  the design doc before treating any result as a verdict.
- **Kripke, S. — *Wittgenstein on Rules and Private Language* (1982).** The
  "skeptical paradox": no finite set of past instances fixes which rule one
  follows (plus vs. quus). *Why it matters:* the spine of E4, and the sharpest
  statement of the underdetermination that E1–E3 also brush against.

---

## E1 — Convention from use (signaling)

*Target: meaning-as-use in its thinnest form — can a shared convention emerge from
use alone?* Approach doc: [`e1_signaling.md`](e1_signaling.md).

- **Lewis, D. — *Convention: A Philosophical Study* (1969).** Introduces the
  signaling game E1 is built on: coordination on initially meaningless signals,
  with meaning constituted by the coordinated practice. *Why it matters:* the
  formal backbone of E1 and the baseline the later experiments perturb.
- **Emergent-communication literature** (see [Methodology](#methodology--cross-cutting-technical)):
  the modern multi-agent instantiation of Lewis games, and the source of E1's
  metrics (emergence, compositionality/topographic similarity, transmission across
  agent turnover).

## E2 — The private diarist (§258)

*Target: the private-language argument — does "same again" stabilize without an
external check, and on what does any stability rest?* Approach doc:
[`e2_diarist.md`](e2_diarist.md).

- **Wittgenstein, *PI* §258** (and §§243–315). The diarist writes "S" for a
  recurring private sensation with no public criterion for "same again." *Why it
  matters:* E2 makes the "memory as the only check" dilemma a number — and reports
  it as a dilemma, per the license.
- **Kripke (1982)** — the rule-following reading of §258 that E2 sets up and E4
  formalizes. *How to extend:* E2's memory toggle is the natural place to probe
  "is context-memory an independent check or a longer private impression?"

## E3 — The beetle-box (§293)

*Target: whether the private contents of the "box" do any work in the public
language-game.* Approach doc: [`e3_design.md`](e3_design.md).

- **Wittgenstein, *PI* §293.** Everyone has a box; each calls what is inside it a
  "beetle"; the box's contents "cancel out" of the public use of the word. *Why it
  matters:* E3 is a near-literal build of §293, with the four box conditions as the
  God's-eye control the agents lack. *Guardrail:* §293 is not behaviorism and not a
  denial that sensations exist — "not a something, but not a nothing either."
- **"The Bewitching AI: The Illusion of Communication with Large Language
  Models,"** *Philosophy & Technology* (2025).
  <https://link.springer.com/article/10.1007/s13347-025-00893-6>. *Why it matters:*
  the deflationary-vs-shadow reading of human–LLM coordination that E3's shared /
  divergent / empty conditions operationalize.

## E4 — Quus (rule-following + interpretability)

*Target: Kripke's rule-following skepticism, and the gap between behavioral
underdetermination and mechanistic (non-)determinacy.* Approach doc:
[`e4_quus.md`](e4_quus.md).

- **Kripke (1982)** — the plus/quus underdetermination E4's behavioral layer makes
  visible.
- **Y. Li, "Language Models, Arithmetic, and Rule-following" (2025).**
  <https://sites.nd.edu/yuanshan-li/2025/03/17/language-models-arithmetic-and-rule-following/>.
  *Why it matters:* the Kripke-meets-interpretability hinge — the exact move E4
  makes by pairing a quus task with a grokked arithmetic circuit.
- **"Rule-Following and Artificial Intelligence: A Kripkean Perspective"**
  (Atomos, 2025).
  <https://atomos.org/2025/01/rule-following-and-artificial-intelligence-a-kripkean-perspective/>.
  *Why it matters:* situates rule-following skepticism for AI systems directly.
- **Grokking & mechanistic-interpretability set** (the mechanistic layer) — see
  [Methodology](#methodology--cross-cutting-technical): Power et al. (2022); Nanda
  et al. (2023); "The Clock and the Pizza" (2023); DeMoss et al. (2024). *How to
  extend:* E4's `mech/` sub-stack (transformer + Fourier circuit read-out +
  cross-seed agreement) is where these connect to code.

## E5 — Forms of life / grounding

*Target: the crux — public moves alone vs. embedding in a form of life; points most
directly at the LLM question.* Approach doc: [`e5_forms_of_life.md`](e5_forms_of_life.md).

- **Wittgenstein, *PI* on forms of life** (§§19, 23, 241) and meaning-as-use woven
  into "the natural history of a creature." *Why it matters:* E5 asks whether
  grounding words in real consequences (a shared environment with stakes) changes
  the character of the emergent language — the load-bearing question for LLMs.
- **"Mechanistic Indicators of Understanding in Large Language Models,"**
  *Philosophical Studies* (2026).
  <https://link.springer.com/article/10.1007/s11098-026-02513-1> · arXiv:
  <https://arxiv.org/abs/2507.08017>. *Why it matters:* frames what "understanding"
  vs. "a convincing shadow" could mean mechanistically — the fault line E5 probes
  behaviorally (grounded vs. ungrounded) and E4 probes mechanistically.
- **Emergent-communication / grounding literature** — see
  [Methodology](#methodology--cross-cutting-technical); the grounding manipulations
  (tool use, resource/survival tasks, stakes) and the transfer/robustness metrics.

## E6 — The reflexive layer

*Target: the meta-example — agents examining whether their own coordination
constitutes shared meaning or its shadow (observer folded into observed). Planned;
strictest pre-registration.* See `plan/beetle-box.md` §4.6.

- **"The Bewitching AI"** (2025) and **"Mechanistic Indicators of Understanding"**
  (2026) — both above; the shadow-vs-understanding framing E6 turns on itself.
- **Wittgenstein, *PI* §293** — the first-person predicament E6 dramatizes: from
  inside the exchange, the deflationary and form-of-life readings are
  indistinguishable.

---

## Methodology & cross-cutting technical

Read alongside `plan/beetle-box.md` §3 (methodological stance) and §5 (architecture).

**Emergent communication**

- **LLM-Language-Games** — code: <https://github.com/PoorPeer/LLM-Language-Games> ·
  thesis: <https://www.diva-portal.org/smash/get/diva2:1955696/FULLTEXT01.pdf>.
  *Why it matters:* a reference implementation and survey of LLM-driven language
  games; relevant to E1's rich mode and E5's grounding.

**Grokking & mechanistic interpretability** (the E4 backbone; relevant to E6's
"opening the box")

- **Power et al., "Grokking: Generalization Beyond Overfitting on Small Algorithmic
  Datasets" (2022).** <https://arxiv.org/abs/2201.02177>. *Why it matters:* the
  memorize→generalize phenomenon E4's `GrokkingRun` reproduces.
- **Nanda et al., "Progress Measures for Grokking via Mechanistic
  Interpretability" (2023).** <https://arxiv.org/abs/2301.05217>. *Why it matters:*
  the Fourier ("Clock") circuit E4's `mech.circuits` reads out of the embeddings.
- **"The Clock and the Pizza: Two Stories in Mechanistic Explanation of Neural
  Networks" (2023).** <https://arxiv.org/abs/2306.17844>. *Why it matters:* same
  task, different learned algorithms — exactly the cross-seed algorithmic
  (non-)agreement E4 measures.
- **DeMoss et al., "The Complexity Dynamics of Grokking" (2024).**
  <https://arxiv.org/abs/2412.09810>. *Why it matters:* the description-length /
  MDL account behind E4's weight-norm complexity proxy at the grokking transition.

**Philosophy of LLMs** (cross-cutting; especially E3, E5, E6)

- **"The Bewitching AI" (2025)** and **"Mechanistic Indicators of Understanding"
  (2026)** — listed under E3 and E5 above; both bear on the whole program's central
  question of communication-vs-illusion and understanding-vs-shadow.

---

## How to cite / extend

When adding an experiment or a method, add its load-bearing references here under
the relevant experiment (with a one-line "why it matters"), and cite them from that
experiment's approach doc (`docs/eN_*.md` References section) and, where a threshold
or claim depends on a result, from the frozen `prereg/eN_*.yaml`. Keep this file the
single consolidated index; the per-experiment docs carry the short, local lists.
