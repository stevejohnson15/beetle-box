# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""E2 entrypoint: the private diarist (§258).

Examples::

    # Sweep the memory access (the central manipulation):
    python experiments/e2_diarist/run.py -m diarist=full,windowed,none
    # Noisy "same again":
    python experiments/e2_diarist/run.py diarist=full percept=noisy
    # Contrast policies:
    python experiments/e2_diarist/run.py diarist=fixed_quantizer

Score a run with:
    python -m beetlebox.analysis.e2 results/<config_hash>/seed<seed>
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

from beetlebox.config import config_hash, from_dict_e2, to_dict
from beetlebox.harness import E2RunManager
from beetlebox.runlog import RunLogger, run_dir


@hydra.main(version_base=None, config_path="../../configs", config_name="e2_config")
def main(dcfg: DictConfig) -> None:
    cfg = from_dict_e2(OmegaConf.to_container(dcfg, resolve=True))
    chash = config_hash(cfg)
    directory = run_dir(cfg.output_dir, chash, cfg.seed)
    with RunLogger(directory) as logger:
        logger.write_manifest(to_dict(cfg), seed=cfg.seed, config_hash=chash)
        summary = E2RunManager(cfg, logger=logger).run()
    print(f"[E2] policy={summary['policy']} memory={summary['memory']} "
          f"noise={summary['noise']}: distinct_terms={summary['distinct_terms']} "
          f"-> {directory}")
    print(f"[E2] score it with:  python -m beetlebox.analysis.e2 {directory}")


if __name__ == "__main__":
    main()
