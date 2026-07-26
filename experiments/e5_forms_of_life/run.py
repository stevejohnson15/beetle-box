# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""E5 entrypoint: forms of life / grounding (capstone).

Examples::

    python experiments/e5_forms_of_life/run.py -m experiment=e5_grounded,e5_ungrounded
    python experiments/e5_forms_of_life/run.py experiment=e5_grounded channel=wide env=grid

Score a run with:
    python -m beetlebox.analysis.e5 results/<config_hash>/seed<seed>
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

from beetlebox.config import config_hash, from_dict_e5, to_dict
from beetlebox.harness import E5RunManager
from beetlebox.runlog import RunLogger, run_dir


@hydra.main(version_base=None, config_path="../../configs", config_name="e5_config")
def main(dcfg: DictConfig) -> None:
    cfg = from_dict_e5(OmegaConf.to_container(dcfg, resolve=True))
    chash = config_hash(cfg)
    directory = run_dir(cfg.output_dir, chash, cfg.seed)
    with RunLogger(directory) as logger:
        logger.write_manifest(to_dict(cfg), seed=cfg.seed, config_hash=chash)
        summary = E5RunManager(cfg, logger=logger).run()
    print(f"[E5] grounded={summary['grounded']} performance={summary['performance']:.3f} "
          f"-> {directory}")
    print(f"[E5] score it with:  python -m beetlebox.analysis.e5 {directory}")


if __name__ == "__main__":
    main()
