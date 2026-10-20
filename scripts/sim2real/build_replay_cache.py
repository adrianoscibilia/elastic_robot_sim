#!/usr/bin/env python3
"""Build the replay sidecar cache of a real run and print its checks (S2R_01 T-1.4).

The run is a run id of the config (``A``, ``B``) or a run folder. The cache
goes to ``data/sim2real/cache/<robot>/<run>/`` and is keyed by the input
digests: a second call with unchanged inputs is a no-op (``--force``
rebuilds). ``--no-save`` builds the sidecar in memory and prints the checks
without writing anything.

Examples
--------
Cache both iiwa runs of the base config::

    python scripts/sim2real/build_replay_cache.py A B

Check a run folder without writing the cache::

    python scripts/sim2real/build_replay_cache.py \
        data/real_robot/kuka_lbr_iiwa_14_r820/20261014/iiwa_fast_all_01 --no-save
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, os.fspath(_REPO / "src"))

from elastic_sim.sim2real.config import load_config
from elastic_sim.sim2real.real_runs import _load_base, build_replay_cache, build_sidecar

DEFAULT_CONFIG = _REPO / "config" / "sim2real" / "kuka_lbr_iiwa_14_r820_base.yaml"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs", nargs="+", help="run ids of the config, or run folders")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="sim2real config (run ids, sha256)")
    parser.add_argument("--cache-root", type=Path, default=None, help="default: data/sim2real/cache")
    parser.add_argument("--force", action="store_true", help="rebuild even if the cache key matches")
    parser.add_argument("--no-save", action="store_true", help="build in memory, print, write nothing")
    parser.add_argument("--json", action="store_true", help="print windows.json instead of the summary")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    for name in args.runs:
        if name in config.runs:
            spec = config.runs[name]
            path, sha, run_id = spec.path, spec.sha256, name
        else:
            path, sha, run_id = Path(name), None, None
        if args.no_save:
            run = _load_base(path, expected_sha256=sha, run_id=run_id)
            _, windows = build_sidecar(run)
            target, built = None, False
        else:
            target, windows, built = build_replay_cache(path, cache_root=args.cache_root, force=args.force,
                                                        expected_sha256=sha)
        if args.json:
            print(json.dumps(windows, indent=2, default=float))
            continue
        status = "not saved" if target is None else ("built" if built else "unchanged (no-op)")
        print(f"== {run_id or path.name}: {path}")
        print(f"   cache: {target or '-'} [{status}]")
        print_summary(windows)
    return 0


def print_summary(windows: dict) -> None:
    header = "bag".ljust(24) + "kind".ljust(20) + "rows".rjust(7) + "  consec  latency       lag max|q-cmd|  cmd chg  run"
    print("   " + header)
    for name, entry in windows["bags"].items():
        tracking = entry.get("tracking", {})
        latency = entry.get("latency_cycles", "")
        if "latency_vs_excitation_cycles" in entry:
            latency = f"{latency} ({entry['latency_vs_excitation_cycles']:+.1f})"
        lag = entry["plan_lag"]["cycles"] if "plan_lag" in entry else ""
        if "plan_lag_vs_excitation_cycles" in entry:
            lag = f"{lag} ({entry['plan_lag_vs_excitation_cycles']:+.1f})"
        matched = entry.get("matched_rows", entry["rows"])
        cadence = entry.get("command_cadence")
        change = "" if cadence is None else f"{cadence['changed_fraction']:.4f}"
        longest = "" if cadence is None else cadence["longest_identical_rows"]
        print("   " + f"{name:<24}{entry['kind']:<20}{matched:>7}  {str(entry['consecutive_cycles']):<6}  "
              f"{str(latency):>11} {str(lag):>9} {_mrad(tracking, 'cmd'):>10} {change:>8} {longest:>4}")
    summary = windows["summary"]
    tracking = summary["tracking"]
    print(f"   excitations ({tracking['excitation_bags']} bags), per joint, mrad:")
    for key in ("rms_q_minus_cmd_rad", "max_abs_q_minus_cmd_rad", "max_abs_q_minus_plan_rad"):
        if key in tracking:
            print(f"     {key:<26}" + " ".join(f"{1e3 * v:7.3f}" for v in tracking[key]))
    if "excitation_latency_cycles" in summary:
        lat = summary["excitation_latency_cycles"]
        print(f"   excitation event latency, cycles: median {lat['median']:g}, min {lat['min']}, max {lat['max']}")
    if "excitation_plan_lag_cycles" in summary:
        lag = summary["excitation_plan_lag_cycles"]
        print(f"   excitation cmd-vs-plan lag, cycles: median {lag['median']:g}, min {lag['min']}, max {lag['max']}")
    cadence = summary.get("command_cadence")
    if cadence:
        print(f"   command echo cadence ({cadence['bags']} bags): changed on {cadence['changed_fraction']:.4f} of rows "
              f"(min {cadence['changed_fraction_min']:.4f}), longest identical run {cadence['longest_identical_rows']}, "
              f"median run {cadence['median_identical_rows']:g}")
    holds = summary.get("holds")
    if holds:
        print(f"   holds ({holds['n_holds']}, rows {holds['rows']}), per joint:")
        print("     sigma_ft N·m           " + " ".join(f"{v:7.4f}" for v in holds["sigma_ft"]))
        if "sigma_q" in holds:
            print("     sigma_q rad            " + " ".join(f"{v:7.1e}" for v in holds["sigma_q"]))
        print("     gravity slope          " + " ".join(f"{v:7.4f}" for v in holds["gravity_slope"]))
        print("     gravity r2             " + " ".join(f"{v:7.4f}" for v in holds["gravity_r2"]))


def _mrad(tracking: dict, name: str) -> str:
    values = tracking.get(f"max_abs_q_minus_{name}_rad")
    return "-" if values is None else f"{1e3 * float(np.max(values)):.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
