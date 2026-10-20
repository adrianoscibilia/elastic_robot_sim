#!/usr/bin/env python3
"""Identifiability report of a config at its start theta (S2R_01 T-4.3).

Writes ``identifiability.{json,md}`` to ``<config output>/identifiability_<stem>/``
(``--no-save`` prints the markdown only). R0: the analytic Jacobian.

Example::

    python scripts/sim2real/identifiability.py --config config/sim2real/kuka_lbr_iiwa_14_r820_base.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, os.fspath(_REPO / "src"))

from elastic_sim.sim2real.config import load_config

DEFAULT_CONFIG = _REPO / "config" / "sim2real" / "kuka_lbr_iiwa_14_r820_base.yaml"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--theta", type=Path, help="overlay to analyse at (default: the config's start theta)")
    parser.add_argument("--no-save", action="store_true", help="print only")
    args = parser.parse_args(argv)

    from elastic_sim.sim2real import identifiability as ida
    from elastic_sim.sim2real.calibration import analysed_names, identifiability_at
    from elastic_sim.sim2real.parameters import Registry, check_write_target
    from elastic_sim.sim2real.plant import PlantBase
    from elastic_sim.sim2real.problem import CalibrationProblem, load_runs
    from elastic_sim.sim2real.report import write_json

    config = load_config(args.config)
    base = PlantBase.from_config(config)
    registry = Registry.from_config(config, base)
    theta = registry.values() if args.theta is None else registry.with_overlay(args.theta).values()
    problem = CalibrationProblem.from_config(config, registry, base, load_runs(config))
    block = identifiability_at(problem, theta, analysed_names(registry, base))
    text = ida.to_markdown(block, title=f"Identifiability: {config.source.name}")
    print(text)
    if not args.no_save:
        out = config.output / f"identifiability_{config.source.stem}"
        check_write_target(out / "identifiability.json")
        write_json(out / "identifiability.json", block)
        (out / "identifiability.md").write_text(text, encoding="utf-8")
        print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
