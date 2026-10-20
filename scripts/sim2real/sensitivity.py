#!/usr/bin/env python3
"""lambda sensitivity of a stage (S2R_06 F-1): the stage at x0.1 and x0.001.

Runs the stage config at each ``--factors`` value of
``loss.prior_weight_factor`` (run folders ``<stem>_lambda_x<f>``; resumed if
present), then writes ``sensitivity.{md,json}`` into the stage's own run
folder and rebuilds its ``report.md``, which then carries the sensitivity
columns. Every conclusion that changes across the decade is flagged.
``--no-save`` prints the table and writes nothing (the runs must exist).

Example::

    python scripts/sim2real/sensitivity.py --config config/sim2real/iiwa_s1_rigid.yaml
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, os.fspath(_REPO / "src"))

from elastic_sim.sim2real.config import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--factors", type=float, nargs="+", default=[0.1, 0.001])
    parser.add_argument("--run-id", help="the stage's run folder (default: the config stem)")
    parser.add_argument("--no-save", action="store_true", help="print only (the runs must already exist)")
    args = parser.parse_args(argv)

    from elastic_sim.sim2real.calibration import calibrate, report_only
    from elastic_sim.sim2real.report import sensitivity, write_json

    config = load_config(args.config)
    main_folder = (config.output / (args.run_id or config.source.stem)).resolve()
    if not (main_folder / "metrics.json").is_file():
        parser.error(f"{main_folder} has no finished run: run the stage first (calibrate.py --config ...)")
    others = {}
    for factor in args.factors:
        folder = config.output / f"{config.source.stem}_lambda_x{factor:g}"
        if not (folder / "metrics.json").is_file():
            if args.no_save:
                parser.error(f"{folder} missing (--no-save runs nothing)")
            variant = dataclasses.replace(config, loss=dataclasses.replace(config.loss, prior_weight=None,
                                                                           prior_weight_factor=factor),
                                          resolved={**config.resolved, "loss": {
                                              **dict(config.resolved.get("loss") or {}), "prior_weight": "auto",
                                              "prior_weight_factor": factor}})
            calibrate(variant, run_id=folder.name, resume=(folder / "history.jsonl").is_file(), figures=False,
                      command=f"sensitivity.py x{factor:g}")
        others[factor] = folder
    table, payload = sensitivity(main_folder, others)
    print(table)
    if not args.no_save:
        (main_folder / "sensitivity.md").write_text(table, encoding="utf-8")
        write_json(main_folder / "sensitivity.json", payload)
        report_only(main_folder, figures=False)
        print(f"written: {main_folder / 'sensitivity.md'}; report rebuilt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
