# Beetle-Box

*An experimental program using LLM agents to **operationalize** — not to prove or refute — Ludwig Wittgenstein's arguments about private language, meaning-as-use, rule-following, and forms of life.*

---

## 0. For whoever picks this up (including Claude Code)

This is a research-design document, not a spec for a product. Read Sections 1–3 before writing any code: they contain the commitments that keep this project honest. The experiments in Section 4 are the buildable core. Section 5 is the architecture, Section 6 a proposed repo layout, Section 7 the order of work.

The single most important instruction: **this project does not aim to prove or disprove Wittgenstein.** His claims are grammatical, not empirical. The aim is to turn his thought experiments into manipulable instruments — intuition pumps we can crank — where surprising agent behavior becomes an occasion for philosophical work, never a verdict. If at any point the code starts to imply "we have shown that meaning is/isn't X," stop and reread Section 3.

---

## 1. The doorway: a meta-example

This project began in a conversation between a human and an LLM about whether an LLM could share meaning with a human. Partway through, the obvious irony surfaced: the conversation was itself an instance of the thing under investigation.

We never secured access to each other's inner states. The human had no read-out of whatever was or wasn't happening "inside" the model; the model had no access to the human's inner life. And yet coordination succeeded — we converged on concepts, on judgments, on what to build. By Wittgenstein's own lights (see §293 below), that is not a limitation we worked around; it is the point arriving in the first person. The coordination happened and the "boxes" never came up. In effect, the conversation was a live run of Experiment 3, and the beetle appeared to cancel out.

But it lands as *irony*, not as proof — and the reason why is the entire research program in miniature:

- **The deflationary reading** says: public practice was sufficient, meaning happened, whatever was in the boxes was idle. Human and LLM found common ground on equal terms.
- **The form-of-life reading** says: it worked only because the human's "box" is stuffed with a lifetime of being a creature that winces, grieves, and files taxes, and the model's is stuffed with the compressed traces of millions of people who did those things. The "common ground" may be shared practice — or the model returning a high-fidelity shadow of the human's form of life, with the human silently supplying all the grounding while experiencing it as mutual.

**From inside the exchange, those two stories are indistinguishable** — which is precisely what §293 predicts you will not be able to tell, and precisely what mechanistic interpretability was supposed to adjudicate and (for any real model, today) cannot.

The methodological upshot is not a flourish; it is the motivating fact of the project:

> The investigator and one entire class of subject are the same kind of thing the investigation is about. The participant's-eye view cannot settle the question — which is *why* the question must be built into conditions, metrics, and pre-registration rather than introspected or simply asked. The irony is the argument for the rigor.

This also means the agents are never neutral instruments; each is also a specimen of the phenomenon. Rather than treat that only as a contaminant, we make it a deliberate layer (Experiment 6): agents turned to examine whether their own coordination constitutes shared meaning or its shadow — observer folded into observed.

---

## 2. Background: the philosophical touchstones

Compressed, and only as needed to design against. Primary texts are the anchors; read them in hardcopy.

- **The beetle-box — *Philosophical Investigations* §293.** Everyone has a box; each calls what is inside it a "beetle"; no one can look into another's box. Whatever is in the box "cancels out" — if "beetle" has a public use, that use cannot be as the name of the private object, because the private object does no work. The beetle is a *sensation* (pain is the running example), and the argument targets the "object and designation" model of meaning. **Guardrail:** this is *not* a denial that sensations exist, and *not* behaviorism. "It is not a something, but not a nothing either." The privacy is real; its *semantic relevance* is the illusion. The point is grammatical.

- **The private-language argument — §§243–315, esp. the diarist at §258.** A diarist writes "S" whenever a private sensation recurs, with no public criterion for "same again." Wittgenstein's claim: absent an independent check, there is no difference between "following the rule for S" and merely seeming to — "whatever is going to seem right to me is right."

- **Meaning-as-use.** Meaning lives in the public practice — the language-game — not in a private referent behind it.

- **Forms of life.** "Use" is not bare statistics of word-succession. It is use woven into the natural history of a creature — pain-behavior, care for the injured, the whole surrounding activity. **This is the crux for LLMs and the fault line the whole program keeps hitting:** is a language-game individuated by its public moves alone (model is in), or by its embedding in a form of life (model has a convincing shadow of the game, not the game)? The beetle-box pushes toward the first answer; the surrounding sections pull toward the second. The tension is arguably unresolved *in Wittgenstein himself*.

- **Rule-following & the quus problem — Kripke, *Wittgenstein on Rules and Private Language* (1982).** No finite set of past instances fixes which rule one is following; "plus" and the deviant "quus" agree on all examples seen so far. Memorization and genuine rule-grasp are behaviorally indistinguishable. This connects directly to the grokking / mechanistic-interpretability literature (Section 9), where the memorize→generalize transition and circuit read-outs make the distinction *empirically visible* for toy models — without dissolving the skeptical point (the circuit is just a longer finite fact).

---

## 3. Methodological stance (read before building)

1. **Grammatical, not empirical.** No experiment here falsifies a Wittgensteinian claim. A committed Wittgensteinian is always entitled to say we have changed the subject from grammar to psychology. We accept that and reframe: the deliverable is *operationalization as instrument*, and a defense of that methodology is itself a core intellectual output.

2. **The pretraining confound is central.** Frontier agents arrive pretrained on oceans of human language; nothing "emerges from scratch," and the shared form of life is smuggled in through pretraining. This is both the chief hazard and, itself, a Wittgensteinian phenomenon (inherited practice). Every experiment is therefore run (where feasible) in two modes:
   - **Clean room** — invented, non-natural-language symbol channels, and/or small models trained from scratch, to minimize inherited meaning.
   - **Rich-but-confounded** — frontier agents in natural language, studying how they *redeploy* inherited concepts under novel pressure.
   They answer different questions. Run both; never let a rich-mode result masquerade as a clean-room one.

3. **Guardrails, pre-committed:**
   - **Anthropomorphic over-reading.** Transcripts seduce. Decide in advance what counts as "coordination," "the same term," "a rule followed." No post-hoc reading of intention into logs.
   - **Experimenter degrees of freedom.** The desired result is trivially easy to build into a reward function or a prompt. **Pre-register** hypotheses, conditions, and scoring before running. Keep the pre-registration under version control.
   - **The cancellation must be earnable.** In any beetle-box-style design, wire the inner signal so it *could* influence output. If the private state is architecturally incapable of affecting behavior, "it cancels out" is rigged, not found.

---

## 4. The experiments

Each is specified by: **target** (the notion), **setup**, **manipulations**, **metrics**, and **what outcomes do and do not license**.

### 4.1 — E1: Convention from use (Lewis signaling game)

- **Target:** meaning-as-use, in its thinnest form. The warm-up and baseline.
- **Setup:** A sender sees a referent and must emit signals drawn from a constrained, non-English token set; a receiver must identify the referent. No meanings assigned in advance.
- **Manipulations:** feedback vs. none; **agent turnover** — swap participants mid-run and test whether a convention survives its founders (a language outliving its speakers is meaning-as-public-practice made visible); population size; channel bandwidth.
- **Metrics:** emergence rate, convention stability, transmission fidelity across turnover, presence/absence of compositional structure.
- **Licenses:** demonstrates the thin thesis; establishes the baseline that later experiments perturb. Does **not** by itself say anything about "understanding."

### 4.2 — E2: The private diarist (§258)

- **Target:** the private-language argument.
- **Setup:** One agent receives a private percept stream only it can see, and names recurring types with self-invented terms, under no external correction.
- **Manipulations:** **memory access** — can the agent consult its own past diary or not?; percept-space structure (clean vs. noisy "same again" judgments); time gaps.
- **Metrics:** term–percept consistency over time; drift; whether "same again" stabilizes or wanders.
- **Licenses:** the interesting case is that context-memory may *manufacture* the stability Wittgenstein says is impossible. That does not refute him — it forces the real question: **is context-memory an independent check, or merely a longer private impression masquerading as one?** Report it as that dilemma, not as a resolution.

### 4.3 — E3: The beetle-box (centerpiece)

- **Target:** §293 directly. Buildable almost literally.
- **Setup:** A population plays a sensation-language-game — reporting and coordinating on "inner states" through public words — and **we control the boxes.**
- **Conditions:** (a) shared inner signal; (b) divergent signals across agents; (c) **empty boxes**; (d) pure noise.
- **Design subtlety (mandatory):** the inner signal must be wired so it *could* propagate to outputs (see Section 3.3), or the cancellation is assumed rather than tested.
- **Metrics:** coordination success on the public game, compared across the four conditions.
- **Licenses:** if empty-box and divergent-box agents coordinate as fluently as shared-box agents, you have a working model of "the beetle drops out." If a shared inner state *measurably* improves coordination, that is a result against the strong deflationary reading. Either way the core claim of §293 becomes a number. Does **not** settle whether the agents "have" inner states.

### 4.4 — E4: Quus and the teachable rule (rule-following + interpretability)

- **Target:** Kripke's rule-following skepticism; the spine linking this project to the grokking thread.
- **Two layers:**
  - **Behavioral:** a teacher gives finite examples consistent with multiple rules (all below the quus bend-point); students extrapolate. Measure convergence vs. quus-like divergence across seeds and models. Divergence makes the underdetermination visible; convergence shows shared inductive priors (inherited form of life) quietly resolving it.
  - **Mechanistic:** in a small **from-scratch** transformer on modular arithmetic (the grokking regime), read out the learned circuit. Ask whether "which rule" has a determinate answer at the mechanism level, and whether two models on identical data converge on the *same* algorithm (the "Clock and the Pizza" result says: often not).
- **Metrics:** behavioral convergence/divergence rates; circuit identification; cross-seed algorithmic agreement; correlation of the memorize→generalize transition with description-length drop (per DeMoss et al.).
- **Licenses:** the payload is the **gap between behavioral underdetermination and mechanistic (non-)determinacy.** Opening the box does not dissolve the skepticism (the circuit is a longer finite fact) — but it relocates the debate onto new, examinable ground. State exactly that.

### 4.5 — E5: Forms of life / grounding (capstone)

- **Target:** the crux — public moves vs. embedding in a form of life. Points most directly at the LLM question.
- **Setup:** take E1 and run it **ungrounded** (tokens connect to nothing) vs. **grounded** in a shared environment where words drive real consequences — tool use, a resource/survival task, stakes.
- **Metrics:** stability, compositionality, robustness to perturbation, whether meanings "stick" and transfer to new agents and new tasks.
- **Licenses:** if grounding measurably changes the character of the emergent language, that is evidence the form of life is load-bearing — the sharpest operational handle on the fault line. Run last, once the harness is mature.

### 4.6 — E6: The reflexive layer (observer folded into observed)

- **Target:** the meta-example of Section 1, made into a condition.
- **Setup:** agents from any prior experiment are turned to examine whether *their own* coordination constitutes shared meaning or its shadow.
- **Metric of interest:** does self-examination change the coordination, or merely add a plausible story on top of it?
- **Licenses:** this is the most interpretively dangerous experiment (maximal seduction toward over-reading) and should be run only with the strictest pre-registration. Its value is conceptual, not confirmatory.

---

## 5. Architecture

- **Orchestration harness:** multi-agent turn management; pluggable agent backends (frontier API models for rich mode; small local models for clean-room mode).
- **Per-agent persistent memory:** explicit, inspectable, and *toggleable* — memory access is an independent variable (see E2), not an implementation detail.
- **Constrained communication channels:** first-class control over the token/symbol vocabulary, bandwidth, and whether the channel is natural language or an invented code. Clean-room vs. rich mode is a channel + backend configuration.
- **Environment layer:** for grounded conditions (E5) — a minimal world with referents, tools, tasks, and consequences.
- **Mechanistic sub-stack:** small transformers trained from scratch (PyTorch), plus interpretability tooling in the lineage of the grokking-circuits work (e.g., TransformerLens-style hooks) for E4.
- **Logging & analysis:** exhaustive, structured, replayable logs; every run reproducible from a seed + config; analysis separated from execution so scoring can be pre-registered and frozen.

Suggested stack: Python throughout; Anthropic API (and/or other providers) for agent backends; PyTorch + an interpretability library for the mechanistic layer; a config system (e.g., Hydra) so conditions are declarative; strict seeding.

---

## 6. Proposed repo structure

```
beetle-box/
  README.md                 # this document (or a pointer to it)
  pyproject.toml
  configs/                  # declarative experiment + condition configs
  prereg/                   # pre-registered hypotheses & scoring, version-controlled
  src/beetlebox/
    harness/                # multi-agent orchestration, turn loop
    agents/                 # backends: api_model, local_model, from_scratch
    memory/                 # persistent, toggleable per-agent memory
    channels/               # communication channels, vocabularies (nl / invented)
    envs/                   # grounded environments for E5
    mech/                   # small transformers + interpretability (E4)
    logging/                # structured run logs
    analysis/               # scoring (imports frozen prereg criteria only)
  experiments/
    e1_signaling/
    e2_diarist/
    e3_beetle_box/
    e4_quus/
    e5_forms_of_life/
    e6_reflexive/
  notebooks/                # exploration only; never the source of a reported result
  results/                  # run outputs, keyed by config hash + seed
```

---

## 7. Roadmap

1. **Harness + E1** — get the orchestration loop, channels, memory, and logging working end-to-end on the simplest game. Fastest clean signal.
2. **E3 (beetle-box)** — the centerpiece; second because it exercises the same harness with the box-wiring twist.
3. **E2 (diarist)** — small, sharp; tests the memory-as-independent-variable machinery.
4. **E4 (quus)** — stand up the mechanistic sub-stack; this is the interpretability backbone and the link to the grokking work.
5. **E5 (forms of life)** — capstone; needs the environment layer and a mature harness.
6. **E6 (reflexive)** — last; strict pre-registration.

Cross-cutting, from day one: pre-registration discipline, dual-mode (clean-room / rich) runs, and the two guardrails in Section 3.

---

## 8. Open questions & risks

- **Is context-memory the independent check Wittgenstein denies is possible, or a longer private impression?** (E2) — likely irreducible; report as a dilemma.
- **How far do toy-model interpretability results extend to real models?** (E4) — open and contested; a critic will press this hard. Do not overclaim.
- **Can any "empty box" be wired to genuinely *could*-influence output without covertly making it non-empty?** (E3) — a real design subtlety; get it reviewed.
- **Fast-moving, partly non-peer-reviewed literature** (grokking / interpretability) — tell load-bearing results from hype.
- **The permanent temptation to read the transcripts as a verdict.** The whole of Section 3 exists to resist this.

---

## 9. References

**Philosophy — primary (hardcopy):**
- Wittgenstein, *Philosophical Investigations* — rule-following §§138–242; private language §§243–315; beetle-box §293.
- Kripke, *Wittgenstein on Rules and Private Language* (1982).

**Philosophy — LLM-facing:**
- "Rule-Following and Artificial Intelligence: A Kripkean Perspective" (Atomos, 2025). https://atomos.org/2025/01/rule-following-and-artificial-intelligence-a-kripkean-perspective/
- "The Bewitching AI: The Illusion of Communication with Large Language Models," *Philosophy & Technology* (2025). https://link.springer.com/article/10.1007/s13347-025-00893-6
- Y. Li, "Language Models, Arithmetic, and Rule-following" (2025) — the Kripke-meets-interpretability hinge. https://sites.nd.edu/yuanshan-li/2025/03/17/language-models-arithmetic-and-rule-following/
- "Mechanistic Indicators of Understanding in Large Language Models," *Philosophical Studies* (2026). https://link.springer.com/article/10.1007/s11098-026-02513-1 · arXiv: https://arxiv.org/abs/2507.08017

**Emergent communication:**
- LLM-Language-Games (code): https://github.com/PoorPeer/LLM-Language-Games · thesis: https://www.diva-portal.org/smash/get/diva2:1955696/FULLTEXT01.pdf

**Grokking & mechanistic interpretability:**
- Power et al., "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets" (2022). https://arxiv.org/abs/2201.02177
- Nanda et al., "Progress Measures for Grokking via Mechanistic Interpretability" (2023). https://arxiv.org/abs/2301.05217
- DeMoss et al., "The Complexity Dynamics of Grokking" (2024) — description-length / MDL account. https://arxiv.org/abs/2412.09810
- "The Clock and the Pizza: Two Stories in Mechanistic Explanation of Neural Networks" (2023) — same task, different learned algorithms. https://arxiv.org/abs/2306.17844

---

*Working title: **Beetle-Box**. The name is a promise to keep the box shut where Wittgenstein says it must be, and to be honest about what we can and cannot see when we try to open it.*
