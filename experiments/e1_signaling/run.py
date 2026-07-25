# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""E1 entrypoint: convention from use (Lewis signaling game).

This is the Hydra app layer -- the *only* place Hydra appears. It composes a
config, converts it to the library's plain dataclass schema, runs the harness,
and writes a reproducible run under ``results/<config_hash>/seed<seed>/``.

Examples::

    python experiments/e1_signaling/run.py seed=0
    python experiments/e1_signaling/run.py experiment=e1_ablation
    python experiments/e1_signaling/run.py env=grid channel=wide
    python experiments/e1_signaling/run.py -m seed=0,1,2   # multirun sweep
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

from beetlebox.config import config_hash, from_dict, to_dict
from beetlebox.harness import RunManager
from beetlebox.runlog import RunLogger, run_dir


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(dcfg: DictConfig) -> None:
    cfg = from_dict(OmegaConf.to_container(dcfg, resolve=True))
    chash = config_hash(cfg)
    directory = run_dir(cfg.output_dir, chash, cfg.seed)

    with RunLogger(directory) as logger:
        logger.write_manifest(to_dict(cfg), seed=cfg.seed, config_hash=chash)
        summary = RunManager(cfg, logger=logger).run()

    print(f"[E1] wrote run to {directory}")
    print(f"[E1] final_accuracy={summary['final_accuracy']:.3f} "
          f"(chance={summary['chance']:.3f})")
    print(f"[E1] score it with:  python -m beetlebox.analysis.e1 {directory}")


if __name__ == "__main__":
    main()
