# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""E4 entrypoint: quus / the teachable rule (behavioral layer).

Examples::

    python experiments/e4_quus/run.py -m quus=scalar,onehot
    python experiments/e4_quus/run.py quus=scalar learner.weight_decay=0.01

Score a run with:
    python -m beetlebox.analysis.e4 results/<config_hash>/seed<seed>

The mechanistic sub-stack (a transformer grokking modular addition) is heavier
compute; run it via ``beetlebox.mech`` -- see docs/e4_quus.md.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

from beetlebox.config import config_hash, from_dict_e4, to_dict
from beetlebox.harness import E4RunManager
from beetlebox.runlog import RunLogger, run_dir


@hydra.main(version_base=None, config_path="../../configs", config_name="e4_config")
def main(dcfg: DictConfig) -> None:
    cfg = from_dict_e4(OmegaConf.to_container(dcfg, resolve=True))
    chash = config_hash(cfg)
    directory = run_dir(cfg.output_dir, chash, cfg.seed)
    with RunLogger(directory) as logger:
        logger.write_manifest(to_dict(cfg), seed=cfg.seed, config_hash=chash)
        summary = E4RunManager(cfg, logger=logger).run()
    print(f"[E4] encoding={summary['encoding']} num_seeds={summary['num_seeds']} "
          f"-> {directory}")
    print(f"[E4] score it with:  python -m beetlebox.analysis.e4 {directory}")


if __name__ == "__main__":
    main()
