#!/usr/bin/env python3
"""Measure what each controller mode's data contains (``R5_00`` Q-C).

Answers Q-C without training anything: for every controller mode, build a few
short bags, then report the collinearity of ``tau_cmd`` with the state
regressor, the conditioning of the achieved motion, the decomposition of the
target's unexplained part, the probe-band content and the deflection's
signal-to-noise.  The whole point of Q-C is that these are cheap -- a handful
of rollouts and linear algebra, not a multi-day training -- so the defaults are
deliberately small.

Two ways to run it:

* ``--generate``: build the datasets here, one per mode, from a config, with
  the trajectory count and robot count cut down.  Everything but
  ``simulation.controller.mode`` is held fixed across modes, so a difference in
  the table is a difference the controller made.
* ``--dataset``: read datasets that already exist (any mix of modes, simulated
  or recorded) and only run the measurements.

Examples
--------
    # The round-5 comparison, UR10, four modes, ~5 bags each
    python scripts/diagnose_controller_modes.py --generate \
        --config config/identification/ur10_table_round5.yaml \
        --modes exact_ct nominal_ct pd_gravity velocity_pi \
        --trajectories 1 --robots 2 --out reports/round5/qc_ur10

    # Only measure, on datasets built earlier
    python scripts/diagnose_controller_modes.py \
        --dataset data/identification/qc_exact_ct.parquet \
        --dataset data/identification/qc_velocity_pi.parquet
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(_REPO / "src"))

import numpy as np
import pandas as pd

from elastic_sim.assets import AssetRegistry, load_asset_spec
from elastic_sim.controllers import CONTROLLER_MODES
from elastic_sim.dataset import PlantExtrasSampling, build_tiers, generate, load_config, write_dataset
from elastic_sim.diagnostics import dataset_diagnostics, load_dataset, summarize_diagnostics

#: The columns worth putting in front of a reader; the CSV keeps all of them.
_HEADLINE = [
    "controller_mode", "plant_extras", "tier_kind", "tau_vs_reference_feedforward",
    "tau_unexplained_fraction", "tau_partial_information", "condition_inflation", "tracking_rms", "rigid_nominal_residual_rms", "target_rms",
    "differentiation_share", "link_friction_share", "noise_share",
    "probe_band_deflection", "deflection_rms", "deflection_over_noise",
    "baseline_rms_state", "baseline_rms_state_tau", "baseline_rms_tau", "baseline_rms_mean",
]


def _load_asset(reference: str):
    candidate = Path(reference)
    if candidate.is_file():
        return load_asset_spec(candidate)
    return AssetRegistry.for_repository(_REPO).load(reference)


def _shrink(config, *, trajectories: int, robots: int, backends: tuple[str, ...]):
    """Cut a production config down to a Q-C-sized one.

    Only the *size* is changed: priors, excitation, payload, gains, the sensor
    model and the plant extras are exactly the config's own, so the table
    compares controllers rather than two differently-tuned builds.
    """
    sampling = replace(config.transmission, robots=robots)
    split = config.split
    if split.mode == "holdout_robots" and split.test_robots + split.val_robots > robots:
        # A Q-C run has a handful of robots, far fewer than a production
        # config holds out.  The split is irrelevant here -- nothing is
        # trained -- so fall back rather than force the caller to copy the
        # config, the same fallback generate_identification_dataset.py makes.
        split = replace(split, mode="contiguous")
    shrunk = replace(
        config, transmission=sampling, n_trajectories=trajectories, backends=backends,
        n_friction_samples=1, split=split,
        tiers=build_tiers(config.rigid_reference, sampling, config.seed),
    )
    return shrunk


def _generate_for_mode(config, asset, mode: str, output: Path, jobs: int, verbose: bool,
                       *, plant_extras: bool = True):
    spec = replace(config.controller, mode=mode)
    if mode == "exact_ct":
        # `exact_ct` with a non-exact nominal block is a contradiction the
        # controller factory refuses; comparing it against the same config's
        # other modes means dropping the mismatch, not the mode.
        spec = replace(spec, nominal=replace(spec.nominal, knows_payload=True, friction_scale=1.0,
                                             rotor_inertia_scale=1.0, inertia_scale=1.0))
    mode_config = replace(config, controller=spec, output=str(output))
    if not plant_extras:
        # R5_03 Sec 2.2's control row: the same build with no link-side friction,
        # nonlinear spring or ripple.  Two things need it.  First, the ranking
        # "velocity_pi tracks better than exact_ct" is an artefact of the
        # *unmodelled* friction -- computed torque droops by d / (M kp) against a
        # constant disturbance while the PI loop's integral carries it -- so
        # without this row the table cannot say whether it is measuring the
        # controller's structure or its disturbance rejection.  Second, a model
        # class that only wins where friction dominates shows up as a win here
        # and nowhere else (R5_Q Q-3's answer).
        mode_config = replace(mode_config, plant_extras=PlantExtrasSampling())
    frame, manifest, comparison = generate(mode_config, asset, verbose=verbose, jobs=jobs)
    write_dataset(frame, manifest, output, comparison, metadata_columns=mode_config.metadata_columns)
    return frame, manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="identification config to build the Q-C datasets from")
    parser.add_argument("--generate", action="store_true", help="build one dataset per mode before measuring")
    parser.add_argument("--modes", nargs="+", default=list(CONTROLLER_MODES), choices=list(CONTROLLER_MODES))
    parser.add_argument("--dataset", action="append", default=[], help="an existing dataset to measure (repeatable)")
    parser.add_argument("--trajectories", type=int, default=1)
    parser.add_argument("--robots", type=int, default=2)
    parser.add_argument("--full", action="store_true",
                        help="Build at the config's own size instead of the Q-C-sized default. "
                             "This is how R5_03 T-2's matched dataset pairs are produced: every "
                             "draw but the controller's is keyed on streams the controller does not "
                             "touch, so two --generate runs differing only in --modes share their "
                             "robots, payloads, trajectories and friction exactly")
    parser.add_argument("--backends", nargs="+", default=["mujoco"])
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--out", default="reports/round5/qc", help="output prefix for the CSVs written")
    parser.add_argument("--data-dir", default="data/identification/round5_qc",
                        help="where --generate writes its datasets")
    parser.add_argument("--no-extras-control", action="store_true",
                        help="Skip the plant_extras-off control build (R5_03 Sec 2.2); by default "
                             "--generate adds one per mode so the table can separate the "
                             "controller's structure from its rejection of the unmodelled terms")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    if args.generate and not args.config:
        parser.error("--generate needs --config")
    if not args.generate and not args.dataset:
        parser.error("give --generate --config ... or at least one --dataset")

    verbose = not args.quiet
    diagnostics: list[pd.DataFrame] = []

    if args.generate:
        config = load_config(args.config)
        asset = _load_asset(config.asset)
        shrunk = config if args.full else _shrink(
            config, trajectories=args.trajectories, robots=args.robots, backends=tuple(args.backends),
        )
        data_dir = Path(args.data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        variants = [(True, "")] if args.no_extras_control else [(True, ""), (False, "_noextras")]
        for mode in args.modes:
            for with_extras, suffix in variants:
                if not with_extras and config.plant_extras.is_empty:
                    continue  # nothing to turn off
                output = data_dir / f"{Path(config.output).stem}_{mode}{suffix}.parquet"
                if verbose:
                    print(f"\n=== {mode}{suffix} -> {output} ===")
                frame, manifest = _generate_for_mode(
                    shrunk, asset, mode, output, args.jobs, verbose, plant_extras=with_extras,
                )
                table = dataset_diagnostics(frame, manifest, asset)
                table["plant_extras"] = with_extras
                diagnostics.append(table)

    for reference in args.dataset:
        frame, manifest = load_dataset(reference)
        asset = _load_asset(str(manifest.get("asset") or ""))
        if verbose:
            print(f"\n=== measuring {reference} ({(manifest.get('controller') or {}).get('mode')}) ===")
        table = dataset_diagnostics(frame, manifest, asset)
        table["plant_extras"] = not (manifest.get("plant_extras") is None)
        diagnostics.append(table)

    table = pd.concat(diagnostics, ignore_index=True)
    summary = summarize_diagnostics(table)

    out_prefix = Path(args.out)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    per_bag_path = out_prefix.with_name(out_prefix.name + "_per_bag.csv")
    summary_path = out_prefix.with_name(out_prefix.name + "_summary.csv")
    table.to_csv(per_bag_path, index=False)
    summary.to_csv(summary_path, index=False)

    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print("\n" + "=" * 100)
        print("Q-C summary (median per controller mode and tier kind)")
        print("=" * 100)
        columns = [c for c in _HEADLINE if c in summary.columns]
        print(summary[columns].to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"\nwrote {per_bag_path}\n      {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
