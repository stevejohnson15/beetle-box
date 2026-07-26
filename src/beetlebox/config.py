# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""Declarative configuration schema for Beetle-Box experiments.

The schema is plain ``dataclasses`` with no dependency on Hydra or OmegaConf, so
that library code (harness, channels, runlog, analysis) stays extractable. Hydra
is confined to the experiment entrypoints (``experiments/*/run.py``), which build
these dataclasses from a composed ``DictConfig`` via :func:`from_dict`.

Runs are keyed by ``config_hash`` + ``seed``. The hash is computed over the
*scientifically meaningful* configuration only: ``seed``, ``output_dir``, and
``device`` are deliberately excluded so that the same condition run under
different seeds/hardware lands under one config directory.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, is_dataclass
from typing import Any

# Deterministic (de)serialization + condition hashing now live in the extracted
# `reprolog` library; re-exported here so `beetlebox.config` remains the import
# site for the whole codebase. reprolog's config_hash excludes the same top-level
# fields (seed/output_dir/device) by default.
from reprolog import canonical_json, config_hash, to_dict

__all__ = ["canonical_json", "config_hash", "to_dict"]


@dataclass
class ChannelConfig:
    """An invented, non-natural-language discrete symbol channel.

    ``vocab_size`` (V) symbols, messages of length ``message_length`` (L); the
    channel bandwidth is ``V ** L``. Using integer symbols with no English
    tokens is the clean-room guarantee for E1.
    """

    vocab_size: int = 8
    message_length: int = 1

    @property
    def bandwidth(self) -> int:
        """Number of distinct messages the channel can express: ``V ** L``."""
        return self.vocab_size**self.message_length


@dataclass
class EnvConfig:
    """The signaling-game referent space.

    ``mode="flat"`` yields ``num_referents`` (K) unstructured one-hot objects.
    ``mode="grid"`` yields an attribute x value grid (K = ``num_values **
    num_attributes``), whose structure makes topographic-similarity analysis of
    compositional structure meaningful when ``message_length > 1``.
    """

    mode: str = "flat"  # "flat" | "grid"
    num_referents: int = 8  # used when mode == "flat"
    num_attributes: int = 2  # used when mode == "grid"
    num_values: int = 4  # used when mode == "grid"

    @property
    def num_classes(self) -> int:
        """Total number of referents ``K`` (``num_values ** num_attributes`` on a grid)."""
        if self.mode == "grid":
            return self.num_values**self.num_attributes
        return self.num_referents


@dataclass
class AgentConfig:
    """Hyperparameters for the from-scratch neural sender/receiver."""

    embed_dim: int = 32
    hidden_dim: int = 64
    learning_rate: float = 1e-3
    entropy_coef: float = 0.01  # sender exploration regularizer (REINFORCE)


@dataclass
class ExperimentConfig:
    """The E1 signaling-game training protocol and manipulations."""

    name: str = "e1_signaling"
    feedback: bool = True  # reward the agents on success (vs. ablation with none)
    num_steps: int = 3000
    batch_size: int = 32
    eval_every: int = 100
    eval_batches: int = 20
    # Agent turnover: freeze the converged sender and reinitialize the receiver
    # partway through, then continue -- a convention outliving its founders.
    turnover: bool = False
    turnover_at: float = 0.5  # fraction of num_steps at which to swap
    population_size: int = 1  # reserved for multi-pair populations (>1 later)


@dataclass
class RunConfig:
    """Top-level run configuration."""

    seed: int = 0
    device: str = "cpu"
    output_dir: str = "results"
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)


# --------------------------------------------------------------------------- #
# E3 (beetle-box) configuration
# --------------------------------------------------------------------------- #
@dataclass
class BoxConfig:
    """The private "box" each agent carries (see :mod:`beetlebox.boxes`)."""

    condition: str = "shared"  # shared | divergent | empty | noise
    box_dim: int = 16


@dataclass
class E3ExperimentConfig:
    """The E3 protocol: which sensation-language-game, and the training regime."""

    name: str = "e3_beetle_box"
    # Which of the three operationalizations to run (all are supported; this is a
    # framework for exploration, so the game is a first-class config axis).
    game: str = "private_referent"  # private_referent | sensation_matching | public_referent_aux
    feedback: bool = True
    num_steps: int = 3000
    batch_size: int = 32
    eval_every: int = 100
    eval_batches: int = 20


@dataclass
class E3RunConfig:
    """Top-level E3 run configuration."""

    seed: int = 0
    device: str = "cpu"
    output_dir: str = "results"
    experiment: E3ExperimentConfig = field(default_factory=E3ExperimentConfig)
    box: BoxConfig = field(default_factory=BoxConfig)
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)


# --------------------------------------------------------------------------- #
# E2 (the diarist) configuration
# --------------------------------------------------------------------------- #
@dataclass
class PerceptConfig:
    """The private percept stream (see :mod:`beetlebox.envs.percept`)."""

    num_types: int = 6  # K latent sensation types (God's-eye ground truth)
    dim: int = 8  # percept vector dimension
    noise: float = 0.25  # "clean vs. noisy same-again": std of per-percept noise


@dataclass
class DiaristConfig:
    """The diarist naming policy and its memory access.

    ``policy`` selects the namer; ``memory`` is E2's central manipulation and maps
    to :func:`beetlebox.memory.make_memory` (``none`` -> NoMemory, ``windowed`` ->
    bounded EpisodicMemory of ``window`` entries, ``full`` -> unbounded).
    """

    policy: str = "prototype"  # prototype | fixed_quantizer | noisy_impression
    memory: str = "full"  # none | windowed | full  (prototype policy only)
    window: int = 20  # capacity when memory == "windowed"
    threshold: float = 2.0  # max distance to call a percept "the same again" (prototype)
    quantizer_bins: int = 4  # bins per dim for fixed_quantizer
    quantizer_scale: float = 3.0  # half-range the quantizer bins cover
    impression_temp: float = 1.5  # stochasticity of the noisy-impression namer


@dataclass
class E2ExperimentConfig:
    """The E2 protocol: how long the diary runs, and evaluation split."""

    name: str = "e2_diarist"
    num_steps: int = 2000
    # Fraction boundary between the "early" and "late" halves for the drift metric.
    drift_split: float = 0.5


@dataclass
class E2RunConfig:
    """Top-level E2 run configuration."""

    seed: int = 0
    output_dir: str = "results"
    experiment: E2ExperimentConfig = field(default_factory=E2ExperimentConfig)
    percept: PerceptConfig = field(default_factory=PerceptConfig)
    diarist: DiaristConfig = field(default_factory=DiaristConfig)


# --------------------------------------------------------------------------- #
# E4 (quus / rule-following) configuration
# --------------------------------------------------------------------------- #
@dataclass
class QuusConfig:
    """The quus task (see :mod:`beetlebox.envs.quus`)."""

    max_operand: int = 12  # operands range over [0, max_operand)
    bend: int = 8  # bend-point k: training pairs have max(a,b) < k
    modulus: int = 24  # outputs taken mod this (>= 2*max_operand => plain addition)
    quus_value: int = 0  # what quus returns at/above the bend
    # How operands are represented -- the axis that decides whether shared priors
    # can resolve the underdetermination: "scalar" (value, extrapolable ->
    # simplicity prior favors plus) or "onehot" (each operand independent, so
    # above-bend values are unconstrained -> total divergence).
    encoding: str = "scalar"


@dataclass
class RuleLearnerConfig:
    """The from-scratch learner that extrapolates the underdetermined rule."""

    hidden_dim: int = 64
    learning_rate: float = 5e-3
    num_steps: int = 4000
    weight_decay: float = 0.0  # >0 biases toward simpler (more plus-like) rules


@dataclass
class E4ExperimentConfig:
    """The E4 behavioral protocol: how many seeded students to compare."""

    name: str = "e4_quus"
    num_seeds: int = 8  # independent students on identical below-bend data


@dataclass
class E4RunConfig:
    """Top-level E4 (behavioral) run configuration."""

    seed: int = 0
    device: str = "cpu"
    output_dir: str = "results"
    experiment: E4ExperimentConfig = field(default_factory=E4ExperimentConfig)
    quus: QuusConfig = field(default_factory=QuusConfig)
    learner: RuleLearnerConfig = field(default_factory=RuleLearnerConfig)


@dataclass
class GrokkingConfig:
    """The mechanistic sub-stack: a small transformer grokking modular addition."""

    modulus: int = 53  # prime p; task is (a + b) mod p
    train_frac: float = 0.5  # fraction of the p*p pairs used for training
    d_model: int = 128
    n_heads: int = 4
    num_steps: int = 20000
    learning_rate: float = 1e-3
    weight_decay: float = 1.0  # the weight decay that drives grokking
    batch_size: int = 512  # <=0 means full-batch
    eval_every: int = 200
    seed: int = 0
    device: str = "cpu"
    output_dir: str = "results"


# --------------------------------------------------------------------------- #
# E5 (forms of life / grounding) configuration
# --------------------------------------------------------------------------- #
@dataclass
class E5ExperimentConfig:
    """The E5 protocol: the same signaling game, grounded vs. ungrounded.

    ``grounded=False`` is E1 (reward = identify the referent). ``grounded=True``
    makes words drive real consequences: the receiver chooses an action whose payoff
    depends on the referent, with per-referent **stakes** (a resource/survival task).
    """

    name: str = "e5_forms_of_life"
    grounded: bool = True
    num_steps: int = 4000
    batch_size: int = 64
    eval_every: int = 200
    eval_batches: int = 40
    robustness_noise: float = 0.25  # channel-flip prob for the robustness eval
    turnover: bool = True  # transfer-to-a-new-agent probe
    turnover_at: float = 0.6


@dataclass
class E5RunConfig:
    """Top-level E5 run configuration (reuses the E1 channel/env/agent configs)."""

    seed: int = 0
    device: str = "cpu"
    output_dir: str = "results"
    experiment: E5ExperimentConfig = field(default_factory=E5ExperimentConfig)
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)


# --------------------------------------------------------------------------- #
# E6 (the reflexive layer) configuration -- rich (frontier) mode only
# --------------------------------------------------------------------------- #
@dataclass
class E6Config:
    """The reflexive experiment: two frontier agents coordinate, then examine it.

    ``intervention`` is the manipulation: ``reflect`` (agents examine whether they
    share meaning), ``control`` (a matched task-irrelevant interlude), or ``none``.
    The pre-registered metric is the behavioral change in coordination (post − pre),
    contrasted against control -- never the content of the reflection.
    """

    intervention: str = "reflect"  # reflect | control | none
    num_referents: int = 4
    vocab_size: int = 6
    rounds_per_block: int = 6  # coordination rounds before and after the intervention
    model: str | None = None  # None -> the frontier default
    seed: int = 0
    output_dir: str = "results"


# --------------------------------------------------------------------------- #
# (de)serialization + hashing
# --------------------------------------------------------------------------- #
def from_dict(data: dict[str, Any]) -> RunConfig:
    """Build a :class:`RunConfig` from a plain dict (e.g. a Hydra DictConfig).

    Unknown keys are ignored; missing keys fall back to dataclass defaults.
    """

    def _build(cls, values: dict[str, Any] | None):
        values = values or {}
        kwargs = {}
        for f in dataclasses.fields(cls):
            if f.name not in values:
                continue
            v = values[f.name]
            if is_dataclass(f.type) or f.name in ("experiment", "channel", "env", "agent"):
                sub_cls = {
                    "experiment": ExperimentConfig,
                    "channel": ChannelConfig,
                    "env": EnvConfig,
                    "agent": AgentConfig,
                }.get(f.name)
                kwargs[f.name] = _build(sub_cls, dict(v)) if sub_cls else v
            else:
                kwargs[f.name] = v
        return cls(**kwargs)

    return _build(RunConfig, dict(data))


def from_dict_e3(data: dict[str, Any]) -> E3RunConfig:
    """Build an :class:`E3RunConfig` from a plain dict (e.g. a Hydra DictConfig)."""
    sub = {
        "experiment": E3ExperimentConfig,
        "box": BoxConfig,
        "channel": ChannelConfig,
        "env": EnvConfig,
        "agent": AgentConfig,
    }

    def _build(cls, values: dict[str, Any] | None):
        values = values or {}
        kwargs = {}
        for f in dataclasses.fields(cls):
            if f.name not in values:
                continue
            v = values[f.name]
            if f.name in sub:
                kwargs[f.name] = _build(sub[f.name], dict(v))
            else:
                kwargs[f.name] = v
        return cls(**kwargs)

    return _build(E3RunConfig, dict(data))


def from_dict_e2(data: dict[str, Any]) -> E2RunConfig:
    """Build an :class:`E2RunConfig` from a plain dict (e.g. a Hydra DictConfig)."""
    sub = {
        "experiment": E2ExperimentConfig,
        "percept": PerceptConfig,
        "diarist": DiaristConfig,
    }

    def _build(cls, values: dict[str, Any] | None):
        values = values or {}
        kwargs = {}
        for f in dataclasses.fields(cls):
            if f.name not in values:
                continue
            v = values[f.name]
            kwargs[f.name] = _build(sub[f.name], dict(v)) if f.name in sub else v
        return cls(**kwargs)

    return _build(E2RunConfig, dict(data))


def from_dict_e4(data: dict[str, Any]) -> E4RunConfig:
    """Build an :class:`E4RunConfig` from a plain dict (e.g. a Hydra DictConfig)."""
    sub = {
        "experiment": E4ExperimentConfig,
        "quus": QuusConfig,
        "learner": RuleLearnerConfig,
    }

    def _build(cls, values: dict[str, Any] | None):
        values = values or {}
        kwargs = {}
        for f in dataclasses.fields(cls):
            if f.name not in values:
                continue
            v = values[f.name]
            kwargs[f.name] = _build(sub[f.name], dict(v)) if f.name in sub else v
        return cls(**kwargs)

    return _build(E4RunConfig, dict(data))


def from_dict_e5(data: dict[str, Any]) -> E5RunConfig:
    """Build an :class:`E5RunConfig` from a plain dict (e.g. a Hydra DictConfig)."""
    sub = {
        "experiment": E5ExperimentConfig,
        "channel": ChannelConfig,
        "env": EnvConfig,
        "agent": AgentConfig,
    }

    def _build(cls, values: dict[str, Any] | None):
        values = values or {}
        kwargs = {}
        for f in dataclasses.fields(cls):
            if f.name not in values:
                continue
            v = values[f.name]
            kwargs[f.name] = _build(sub[f.name], dict(v)) if f.name in sub else v
        return cls(**kwargs)

    return _build(E5RunConfig, dict(data))


def from_dict_e6(data: dict[str, Any]) -> E6Config:
    """Build an :class:`E6Config` from a plain dict (flat; unknown keys ignored)."""
    fields = {f.name for f in dataclasses.fields(E6Config)}
    return E6Config(**{k: v for k, v in dict(data).items() if k in fields})
