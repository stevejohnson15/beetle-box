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
import hashlib
import json
from dataclasses import dataclass, field, is_dataclass
from typing import Any

# Fields excluded from config_hash: they do not change the condition under test.
_HASH_EXCLUDED_TOP_LEVEL = ("seed", "output_dir", "device")


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
# (de)serialization + hashing
# --------------------------------------------------------------------------- #
def to_dict(cfg: Any) -> dict[str, Any]:
    """Recursively convert a dataclass config to a plain dict."""
    if is_dataclass(cfg) and not isinstance(cfg, type):
        return {f.name: to_dict(getattr(cfg, f.name)) for f in dataclasses.fields(cfg)}
    if isinstance(cfg, (list, tuple)):
        return [to_dict(v) for v in cfg]
    if isinstance(cfg, dict):
        return {k: to_dict(v) for k, v in cfg.items()}
    return cfg


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


def canonical_json(cfg: Any) -> str:
    """Deterministic JSON string of a config (sorted keys, no whitespace drift)."""
    return json.dumps(to_dict(cfg), sort_keys=True, separators=(",", ":"))


def config_hash(cfg: RunConfig, length: int = 12) -> str:
    """Stable content hash of the *condition* (excludes seed/device/output_dir)."""
    d = to_dict(cfg)
    for k in _HASH_EXCLUDED_TOP_LEVEL:
        d.pop(k, None)
    payload = json.dumps(d, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]
