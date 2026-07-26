# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""Signaling-game environment -- ``SignalingEnv`` re-exported from `emcomkit`.

The generic referent generator ``SignalingEnv`` was extracted into the standalone,
Apache-2.0 ``emcomkit`` package (https://github.com/stevejohnson15/emcomkit);
re-exported here so existing ``beetlebox.envs`` imports keep working. Note that
``emcomkit``'s ``SignalingEnv`` takes plain parameters -- build one from a
:class:`beetlebox.config.EnvConfig` with ``SignalingEnv.from_config(cfg.env)``.

The experiment-specific environments (:mod:`beetlebox.envs.percept`,
:mod:`~beetlebox.envs.grounded`, :mod:`~beetlebox.envs.quus`) remain in Beetle-Box.
"""

from __future__ import annotations

from emcomkit.referents import SignalingEnv

__all__ = ["SignalingEnv"]
