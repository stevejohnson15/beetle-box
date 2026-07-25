# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Beetle-Box contributors
"""The ``beetlebox`` command-line entrypoint.

A thin dispatcher for utilities that should not depend on Hydra (experiment
*runs* are Hydra apps under ``experiments/*/run.py``). Currently exposes scoring::

    beetlebox analyze results/<config_hash>/seed0 [--prereg PATH]
"""

from __future__ import annotations

import argparse
import sys

from beetlebox import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="beetlebox", description="Beetle-Box utilities.")
    parser.add_argument("--version", action="version", version=f"beetlebox {__version__}")
    sub = parser.add_subparsers(dest="command")

    p_analyze = sub.add_parser("analyze", help="score an E1 run against its frozen prereg")
    p_analyze.add_argument("run_dir")
    p_analyze.add_argument("--prereg", default=None)

    args = parser.parse_args(argv)

    if args.command == "analyze":
        from beetlebox.analysis.e1 import _format_report, score_run
        from beetlebox.analysis.prereg import DEFAULT_PREREG

        result = score_run(args.run_dir, args.prereg or DEFAULT_PREREG)
        print(_format_report(result))
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
