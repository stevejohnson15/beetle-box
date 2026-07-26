# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""E6 entrypoint: the reflexive layer (rich mode -- costs API calls).

Run the reflect and control conditions, then contrast them::

    python experiments/e6_reflexive/run.py -m intervention=reflect,control
    python -m beetlebox.analysis.e6 --reflect results/<reflect>/seed0 \
                                    --control results/<control>/seed0

Keep blocks small: a run makes ~4*rounds_per_block + 2 API calls.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

from beetlebox.config import config_hash, from_dict_e6, to_dict
from beetlebox.harness.rich_e6 import RichReflexiveRunner
from beetlebox.runlog import RunLogger, run_dir


@hydra.main(version_base=None, config_path="../../configs", config_name="e6_config")
def main(dcfg: DictConfig) -> None:
    cfg = from_dict_e6(OmegaConf.to_container(dcfg, resolve=True))
    chash = config_hash(cfg)
    directory = run_dir(cfg.output_dir, chash, cfg.seed)
    with RunLogger(directory) as logger:
        logger.write_manifest(to_dict(cfg), seed=cfg.seed, config_hash=chash)
        summary = RichReflexiveRunner(cfg, logger=logger).run()
    print(f"[E6] intervention={summary['intervention']} pre={summary['pre_accuracy']:.3f} "
          f"post={summary['post_accuracy']:.3f} delta={summary['delta']:+.3f} -> {directory}")


if __name__ == "__main__":
    main()
