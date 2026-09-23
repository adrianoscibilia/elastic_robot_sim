#!/usr/bin/env python3
"""Generate a dynamic-model-identification dataset from simulation only.

Excitation trajectories are optimized for regressor conditioning, executed as
torque-driven rollouts in MuJoCo and/or Newton across a rigid tier and a
set of sampled elastic robots, and written as one flat CSV in the schema that
``dynamic_model_nn`` consumes.  When more than one backend runs, every bag
pair that differs only by backend is compared and the result written next to
the dataset.

Settings come from a YAML file under ``config/identification/``; every one of
them can be overridden on the command line.  See
``scripts/run_identification_simulation.py`` to inspect or watch a single
rollout without writing a dataset.

Examples
--------
    python scripts/generate_identification_dataset.py
    python scripts/generate_identification_dataset.py --backends mujoco --trajectories 8 --robots 12
    python scripts/generate_identification_dataset.py --visualize --no-save
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(_REPO / "src"))

from elastic_sim.assets import AssetRegistry, load_asset_spec
from elastic_sim.controllers import CONTROLLER_MODES
from elastic_sim.dataset import DEFAULT_CONFIG, build_tiers, generate, load_config, write_dataset


def _load_asset(reference: str):
    candidate = Path(reference)
    if candidate.is_file():
        return load_asset_spec(candidate)
    return AssetRegistry.for_repository(_REPO).load(reference)


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else _REPO / candidate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="YAML defaults (see config/identification/)")
    parser.add_argument("--asset", default=None)
    parser.add_argument("--backends", nargs="+", choices=("mujoco", "newton"), default=None)
    parser.add_argument("--robots", type=int, default=None,
                        help="Number of sampled elastic robots; 0 for the rigid reference only")
    parser.add_argument("--split-mode", choices=("contiguous", "holdout_robots"), default=None,
                        help="Override split.mode; auto-falls back to contiguous when --robots is too "
                             "small for the config's holdout_robots test_robots+val_robots")
    parser.add_argument("--no-rigid", action="store_true", help="Omit the rigid reference tier")
    parser.add_argument("--trajectories", type=int, default=None)
    parser.add_argument("--friction-samples", type=int, default=None)
    parser.add_argument("--friction-scale", nargs=2, type=float, default=None)
    parser.add_argument("--harmonics", type=int, default=None)
    parser.add_argument("--base-frequency", type=float, default=None)
    parser.add_argument("--periods", type=int, default=None)
    parser.add_argument("--max-acceleration", type=float, default=None)
    parser.add_argument("--candidates", type=int, default=None)
    parser.add_argument("--control-frequency", type=float, default=None)
    parser.add_argument("--controller-mode", default=None, choices=list(CONTROLLER_MODES),
                        help="Override simulation.controller.mode (R5_00 Sec 6.1); everything else in the "
                             "config is untouched, so two runs differing only in this flag differ only in "
                             "their controller")
    parser.add_argument("--position-side", default=None, choices=("link", "motor"),
                        help="Override dataset.signals.position_side: which side of the spring q0../dq0.. are")
    parser.add_argument("--clean-columns", action="store_true",
                        help="Also write the exact simulator values as *_clean columns (debug/diagnostics)")
    parser.add_argument("--sample-step", type=float, default=None, help="Output sampling step [s]")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-save", action="store_true",
                        help="Run everything but write nothing, for debugging")
    parser.add_argument("--visualize", action="store_true", help="Open the native viewer for each rollout")
    parser.add_argument("--realtime-scale", type=float, default=None, help="1.0 is real time")
    parser.add_argument("--jobs", type=int, default=1,
                        help="Parallel worker processes for the bag loop (forced to 1 under --visualize)")
    parser.add_argument("--quiet", action="store_true")
    return parser


def resolve_config(args: argparse.Namespace, parser: argparse.ArgumentParser):
    """Turn parsed CLI args plus the YAML defaults into one ``DatasetConfig``.

    Factored out of ``main`` so a test can assert
    ``resolve_config(no_overrides).excitation == load_config(DEFAULT_CONFIG).excitation``
    without running any simulation -- the regression class this guards
    against (a field silently dropped by a from-scratch
    ``FourierExcitationConfig(...)`` reconstruction instead of
    ``dataclasses.replace``) is exactly the kind of bug that class of test
    catches and the old, inline version of this code could not be tested for
    at all.
    """
    if args.jobs < 1:
        parser.error("--jobs must be >= 1")
    if args.visualize and args.jobs > 1:
        parser.error("--jobs > 1 is incompatible with --visualize")

    config = load_config(_resolve(args.config))

    if args.robots is not None and args.robots < 0:
        parser.error("--robots must be >= 0")
    try:
        transmission = replace(config.transmission, robots=config.transmission.robots if args.robots is None else args.robots)
    except ValueError as exc:
        parser.error(str(exc))
    rigid_reference = config.rigid_reference and not args.no_rigid
    seed = config.seed if args.seed is None else args.seed
    try:
        tiers = build_tiers(rigid_reference, transmission, seed)
    except ValueError as exc:
        parser.error(str(exc))

    split = config.split
    if args.split_mode is not None:
        split = replace(split, mode=args.split_mode)
    elif (split.mode == "holdout_robots"
          and split.test_robots + split.val_robots > transmission.robots):
        # A rigid-only or small --robots run (e.g. an ad-hoc Newton
        # cross-check) cannot satisfy the shipped config's holdout counts;
        # falling back instead of raising is what a rigid-only build (which
        # has no elastic robots to hold out from at all) needs to work
        # without a config copy (R4_14 Sec 3, C-1).
        print(f"note: --robots {transmission.robots} is smaller than "
              f"split.test_robots+val_robots ({split.test_robots + split.val_robots}); "
              "falling back to split.mode=contiguous for this run")
        split = replace(split, mode="contiguous")

    sample_step = args.sample_step or config.sample_time_step
    if args.max_acceleration is not None and config.regime.enabled:
        parser.error(
            "--max-acceleration is silently overridden per-trajectory by excitation.regime "
            "(enabled in this config); edit the YAML's regime.max_acceleration range instead, "
            "or run with a config that has regime.enabled: false"
        )
    if args.control_frequency is not None and config.control_gains.enabled:
        parser.error(
            "--control-frequency is silently overridden per-trajectory by simulation.control_gains "
            "(enabled in this config); edit the YAML's control_gains.natural_frequency range instead, "
            "or run with a config that has control_gains.enabled: false"
        )
    # dataclasses.replace, not a from-scratch FourierExcitationConfig(...):
    # reconstructing field-by-field silently drops any field not named here
    # (centre_jitter, probe_harmonics, probe_acceleration_fraction all fell
    # back to their dataclass defaults this way -- 0.0, (), 0.2 -- so every
    # dataset built from this script had centre_jitter=0 and no probe
    # regardless of what the YAML said).
    excitation = replace(
        config.excitation,
        n_harmonics=args.harmonics or config.excitation.n_harmonics,
        base_frequency=args.base_frequency or config.excitation.base_frequency,
        n_periods=args.periods or config.excitation.n_periods,
        time_step=sample_step,
        max_acceleration=args.max_acceleration or config.excitation.max_acceleration,
    )
    config = replace(
        config,
        asset=args.asset or config.asset,
        backends=tuple(args.backends) if args.backends else config.backends,
        rigid_reference=rigid_reference,
        transmission=transmission,
        tiers=tiers,
        split=split,
        n_trajectories=args.trajectories or config.n_trajectories,
        n_friction_samples=args.friction_samples or config.n_friction_samples,
        friction_scale_range=(tuple(args.friction_scale) if args.friction_scale
                              else config.friction_scale_range),
        seed=seed,
        excitation=excitation,
        candidates=args.candidates or config.candidates,
        control_frequency=args.control_frequency or config.control_frequency,
        # replace(), never a from-scratch ControllerSpec/SignalPolicy: same
        # silent-field-drop class as the excitation reconstruction above.
        controller=(config.controller if args.controller_mode is None
                    else replace(config.controller, mode=args.controller_mode)),
        signals=replace(
            config.signals,
            position_side=args.position_side or config.signals.position_side,
            clean_columns=bool(args.clean_columns) or config.signals.clean_columns,
        ),
        sample_time_step=sample_step,
        output=args.output or config.output,
        visualize=bool(args.visualize) or config.visualize,
        realtime_scale=(args.realtime_scale if args.realtime_scale is not None
                        else config.realtime_scale),
    )
    if config.n_trajectories < 1 or config.n_friction_samples < 1:
        parser.error("trajectories and friction-samples must be positive")
    return config


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = resolve_config(args, parser)

    asset = _load_asset(config.asset)
    asset.resolve_active_joints()
    asset.validate_resources()
    if not args.quiet:
        print(f"asset {asset.name}: tiers {[t.name for t in config.tiers]} "
              f"x backends {list(config.backends)} x {config.n_trajectories} trajectories "
              f"x {config.n_friction_samples} friction samples")

    frame, manifest, comparison = generate(config, asset, verbose=not args.quiet, jobs=args.jobs)
    if args.no_save:
        print(f"\n{len(frame)} samples in {manifest['n_bags']} bags; nothing written (--no-save).")
        return
    csv_path, manifest_path, comparison_path = write_dataset(
        frame, manifest, _resolve(config.output), comparison, metadata_columns=config.metadata_columns,
    )
    print(f"\nWrote {len(frame)} samples in {manifest['n_bags']} bags to {csv_path}")
    print(f"Wrote manifest to {manifest_path}")
    if comparison_path is not None:
        print(f"Wrote backend comparison to {comparison_path}")


if __name__ == "__main__":
    main()
