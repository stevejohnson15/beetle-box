# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""E3 entrypoint: the beetle-box (§293), clean-room or rich (frontier) mode.

Examples::

    # Clean-room condition sweep (from-scratch neural agents):
    python experiments/e3_beetle_box/run.py -m box=shared,divergent,empty,noise
    python experiments/e3_beetle_box/run.py experiment=sensation_matching box=divergent

    # Rich mode (frontier API agents, in-context) — costs money, keep small:
    python experiments/e3_beetle_box/run.py mode=rich box=shared rich.num_rounds=10

Score a condition sweep with:
    python -m beetlebox.analysis.e3 results/<hashA>/seed0 results/<hashB>/seed0 ...
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

from beetlebox.config import config_hash, from_dict_e3, to_dict
from beetlebox.harness import E3RunManager
from beetlebox.runlog import RunLogger, run_dir


@hydra.main(version_base=None, config_path="../../configs", config_name="e3_config")
def main(dcfg: DictConfig) -> None:
    raw = OmegaConf.to_container(dcfg, resolve=True)
    mode = raw.get("mode", "clean")

    if mode == "rich":
        from beetlebox.harness.rich_e3 import RichPrivateReferentRunner

        r = raw.get("rich", {})
        condition = raw["box"]["condition"]
        # Key rich runs by a stable hash of their salient params + condition.
        chash = config_hash(from_dict_e3(raw))
        directory = run_dir(raw["output_dir"], f"rich-{chash}", raw["seed"])
        with RunLogger(directory) as logger:
            logger.write_manifest(raw, seed=raw["seed"], config_hash=f"rich-{chash}")
            summary = RichPrivateReferentRunner(
                condition, num_referents=r.get("num_referents", 4),
                vocab_size=r.get("vocab_size", 6), num_rounds=r.get("num_rounds", 12),
                model=r.get("model", "claude-opus-4-8"), seed=raw["seed"], logger=logger,
            ).run()
        print(f"[E3-rich] {condition}: acc={summary['final_accuracy']:.3f} "
              f"(chance={summary['chance']:.3f}) -> {directory}")
        return

    cfg = from_dict_e3(raw)
    chash = config_hash(cfg)
    directory = run_dir(cfg.output_dir, chash, cfg.seed)
    with RunLogger(directory) as logger:
        logger.write_manifest(to_dict(cfg), seed=cfg.seed, config_hash=chash)
        summary = E3RunManager(cfg, logger=logger).run()
    print(f"[E3] game={summary['game']} condition={summary['condition']}: "
          f"acc={summary['final_accuracy']:.3f} (chance={summary['chance']:.3f}) -> {directory}")


if __name__ == "__main__":
    main()
