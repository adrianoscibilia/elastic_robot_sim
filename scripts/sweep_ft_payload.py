#!/usr/bin/env python3
"""Sweep the calibration payload's mass against the flange cell's noise floor.

``R5_07`` T-14.  A force/torque cell at the flange measures only what is
*distal* to it: with nothing mounted past the sensor it reads identically
zero, and the arm's own link inertia, the gravity of its links and the joint
springs themselves are all proximal and invisible.  So the elastic content of
an ``ee_wrench_joint`` target is

    delta f  ~  m_payload * ( a_true - a_rigid )

at the flange -- the payload mass times the acceleration error the joint
deflections cause.  The payload mass is therefore the one knob that sets how
much the target carries, and it trades against the torque budget: the same
mass that makes the signal also eats the drives' headroom and forces the
excitation down.

``R5_04`` Sec 3 predicted this quantity for FMRR and was wrong by an order of
magnitude, so nothing is predicted here.  The measurement is the same one
``R5_05`` Sec 6.4 made: per joint, the RMS difference between the elastic and
the rigid target, over the RMS of what the cell's own instrument model adds.

For each mass the script rewrites the asset's ``calibration_payload`` mass in
a temporary sibling URDF (the trick ``payload_asset`` uses, so every relative
mesh reference keeps resolving), builds a small matched dataset with the
rigid reference tier in it, and reports:

    elastic / noise   per joint, the ratio that decides observability
    effort headroom   peak |tau_motor| over the joint's rated effort

Example
-------
    python scripts/sweep_ft_payload.py \
        --config config/identification/ur10_table_round5.yaml \
        --masses 0 2 5 10 --robots 2 --trajectories 1 \
        --out reports/round5/t14_ur10_payload_sweep.csv
"""

from __future__ import annotations

import argparse
import contextlib
import os
import re
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(_REPO / "src"))

import numpy as np
import pandas as pd

from elastic_sim.assets import AssetRegistry, AssetSpec, load_asset_spec
from elastic_sim.dataset import RIGID_TIER, PayloadSampling, build_tiers, generate, load_config


@contextlib.contextmanager
def payload_mass_asset(asset: AssetSpec, link: str, mass: float):
    """``asset`` with ``link``'s declared mass replaced by ``mass``.

    The inertia tensor is scaled with the mass, which is what a box of the
    same size and a different material is; the centre of mass is unchanged.
    """
    source = Path(asset.urdf_path)
    text = source.read_text(encoding="utf-8")
    pattern = re.compile(
        rf'(<link\s+name="{re.escape(link)}">.*?<mass\s+value=")([0-9.eE+-]+)(")', re.S,
    )
    match = pattern.search(text)
    if match is None:
        raise ValueError(f"asset {asset.name!r}: no <link name=\"{link}\"> with a mass to rewrite")
    original = float(match.group(2))
    scale = 0.0 if original == 0.0 else float(mass) / original
    text = pattern.sub(rf'\g<1>{mass:.9g}\g<3>', text, count=1)

    def _scale_inertia(inertia_match: re.Match[str]) -> str:
        body = inertia_match.group(0)
        return re.sub(
            r'(i(?:xx|yy|zz|xy|xz|yz)=")([0-9.eE+-]+)(")',
            lambda m: f"{m.group(1)}{float(m.group(2)) * scale:.9g}{m.group(3)}",
            body,
        )

    link_pattern = re.compile(rf'<link\s+name="{re.escape(link)}">.*?</link>', re.S)
    text = link_pattern.sub(lambda m: re.sub(r"<inertia [^>]*/>", _scale_inertia, m.group(0)), text, count=1)

    handle = tempfile.NamedTemporaryFile(
        mode="w", suffix=".urdf", prefix="ft_payload_sweep_",
        dir=source.resolve().parent, delete=False, encoding="utf-8",
    )
    try:
        handle.write(text)
        handle.close()
        yield replace(asset, urdf_path=Path(handle.name))
    finally:
        Path(handle.name).unlink(missing_ok=True)


def _load_asset(reference: str) -> AssetSpec:
    candidate = Path(reference)
    if candidate.is_file():
        return load_asset_spec(candidate)
    return AssetRegistry.for_repository(_REPO).load(reference)


def _measure(frame: pd.DataFrame, n_dof: int) -> dict[str, float]:
    """Elastic content and instrument noise per joint, in the target's units.

    The rigid reference tier runs the *same* trajectory as the elastic tiers
    (the sweep forces a shared trajectory and no sampled payload, so the only
    difference between the two is the transmission), which is what makes the
    difference of the two clean targets the elastic content and nothing else.
    """
    target = [f"ft{i}" for i in range(n_dof)]
    clean = [f"ft_clean{i}" for i in range(n_dof)]
    rigid = frame[frame["tier"] == RIGID_TIER]
    elastic = frame[frame["tier"] != RIGID_TIER]
    if rigid.empty or elastic.empty:
        raise ValueError("the sweep needs both the rigid reference tier and at least one elastic tier")
    rigid_target = rigid.groupby("t")[clean].mean()
    row: dict[str, float] = {}
    elastic_rms = np.zeros(n_dof)
    noise_rms = np.zeros(n_dof)
    count = 0
    for _, bag in elastic.groupby("bag", sort=False):
        bag = bag.set_index("t")
        shared = bag.index.intersection(rigid_target.index)
        difference = bag.loc[shared, clean].to_numpy() - rigid_target.loc[shared].to_numpy()
        noise = bag.loc[shared, target].to_numpy() - bag.loc[shared, clean].to_numpy()
        elastic_rms += np.sqrt(np.mean(difference**2, axis=0))
        noise_rms += np.sqrt(np.mean(noise**2, axis=0))
        count += 1
    elastic_rms /= count
    noise_rms /= count
    for index in range(n_dof):
        row[f"elastic_rms_{index}"] = float(elastic_rms[index])
        row[f"noise_rms_{index}"] = float(noise_rms[index])
        row[f"ratio_{index}"] = float(elastic_rms[index] / noise_rms[index]) if noise_rms[index] else float("inf")
    row["ratio_best"] = float(np.nanmax([row[f"ratio_{i}"] for i in range(n_dof)]))
    row["ratio_median"] = float(np.nanmedian([row[f"ratio_{i}"] for i in range(n_dof)]))
    row["axes_above_5"] = int(sum(row[f"ratio_{i}"] >= 5.0 for i in range(n_dof)))
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--masses", type=float, nargs="+", default=[0.0, 2.0, 5.0, 10.0])
    parser.add_argument("--link", default="calibration_payload",
                        help="the URDF link past the cell whose mass is swept")
    parser.add_argument("--robots", type=int, default=2)
    parser.add_argument("--trajectories", type=int, default=1)
    parser.add_argument("--backends", nargs="+", default=["mujoco"])
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--probe", type=float, nargs=3, metavar=("LOW_HZ", "HIGH_HZ", "LINES"),
                        default=None,
                        help="Replace the config's modal probe with LINES log-spaced lines between "
                             "LOW_HZ and HIGH_HZ, at the same acceleration budget (R5_06 T-9): the "
                             "per-line acceleration, not the band, is what rings the mode, so a "
                             "narrow comb of few lines carries far more per line")
    parser.add_argument("--label", default=None, help="extra column identifying this run in the CSV")
    parser.add_argument("--out", default="reports/round5/t14_payload_sweep.csv")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if config.signals.target != "ee_wrench_joint":
        parser.error(f"{args.config}: this sweep is about an ee_wrench_joint target, not "
                     f"{config.signals.target!r}")
    asset = _load_asset(config.asset)
    n_dof = len(asset.joint_names)

    if args.probe is not None:
        from elastic_sim.excitation import log_spaced_probe_harmonics

        low, high, lines = float(args.probe[0]), float(args.probe[1]), int(args.probe[2])
        config = replace(config, excitation=replace(
            config.excitation,
            probe_harmonics=tuple(log_spaced_probe_harmonics(
                low, high, lines, config.excitation.base_frequency)),
        ))
        if not args.quiet:
            print(f"probe: {lines} lines over {low:g}-{high:g} Hz "
                  f"-> {config.excitation.probe_harmonics}")

    sampling = replace(config.transmission, robots=args.robots)
    # A shared trajectory and no sampled payload: the rigid and the elastic
    # bags must differ by the transmission alone, or the difference of their
    # targets is not the elastic content.
    base = replace(
        config, transmission=sampling, n_trajectories=args.trajectories,
        trajectories_per_robot=False, n_friction_samples=1,
        payload=PayloadSampling(), backends=tuple(args.backends),
        rigid_reference=True, split=replace(config.split, mode="contiguous"),
        signals=replace(config.signals, clean_columns=True),
        tiers=build_tiers(True, sampling, config.seed),
    )

    rows = []
    for mass in args.masses:
        if not args.quiet:
            print(f"\n=== calibration payload {mass:g} kg ===")
        with payload_mass_asset(asset, args.link, mass) as swept:
            frame, manifest, _ = generate(
                replace(base, output=f"data/identification/ft_sweep_{mass:g}.parquet"),
                swept, verbose=not args.quiet, jobs=args.jobs,
            )
        row = {"payload_mass": mass}
        if args.label:
            row["label"] = args.label
        probe = np.asarray(config.excitation.probe_harmonics) * config.excitation.base_frequency
        row["probe_low_hz"] = float(probe.min()) if probe.size else 0.0
        row["probe_top_hz"] = float(probe.max()) if probe.size else 0.0
        row["probe_lines"] = int(probe.size)
        row.update(_measure(frame, n_dof))
        effort = np.asarray([
            joint.effort if joint.effort is not None else np.inf for joint in asset.resolve_active_joints()
        ], dtype=float)
        commanded = frame[[f"tau{i}" for i in range(n_dof)]].to_numpy()
        row["peak_effort_fraction"] = float(np.max(np.abs(commanded) / effort[None, :]))
        rows.append(row)
        if not args.quiet:
            print({key: round(value, 4) for key, value in row.items() if key.startswith("ratio")})

    table = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 60):
        print("\n" + "=" * 100)
        print("T-14 payload sweep: elastic content over cell noise, per joint")
        print("=" * 100)
        print(table[["payload_mass"] + [f"ratio_{i}" for i in range(n_dof)]
                    + ["ratio_best", "axes_above_5", "peak_effort_fraction"]].to_string(index=False))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
