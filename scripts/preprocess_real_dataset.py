#!/usr/bin/env python3
"""Build a training CSV from real-robot parquet recordings.

Reads all parquet files that match ``real_*.parquet`` under the recordings
directory (searched recursively) and produces a single CSV that is directly
loadable by ``CustomDataset`` in dynamic_model_nn/dataset.py.

Column mapping (real robot → dataset)
--------------------------------------
  t                    ← t               (time, seconds)
  q0, q1, q2          ← q_motor_{x,y,z} (joint positions from encoder)
  dq0, dq1, dq2       ← dq_motor_{x,y,z}(joint velocities)
  tau1, tau2, tau3    ← tau_motor_{x,y,z}(commanded joint effort, from /joint_states)
  fx, fy, fz          ← tau_link_{x,y,z} (F/T sensor forces)

Note: on the real robot the elastic deflection is not directly observable, so
``q_link`` is recorded as a copy of ``q_motor``. Joint positions are therefore
taken from ``q_motor`` directly (unlike the sim where q_motor + q_link are
summed to get the total deflection).

Provenance check: recordings made before the ros_recorder.py fix (see git
history) have ``tau_motor`` wrongly duplicated from the F/T sensor
(``tau_link``) instead of the real commanded joint effort. This script flags
any file where tau_motor == tau_link for the whole trajectory, since that is
a strong signature of stale, pre-fix data — such files should be re-recorded
rather than trusted for tau1/tau2/tau3.

Usage
-----
# All sessions under data/recordings/, default output name
python scripts/preprocess_real_dataset.py

# Specific session, explicit output path
python scripts/preprocess_real_dataset.py \\
    --recordings-dir data/recordings/session_latest \\
    --output data/dataset_real_session_latest.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a training CSV from real-robot parquet recordings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--recordings-dir", "-r",
        default=os.path.join(_REPO, "data", "recordings"),
        help="Root directory to search for parquet files (searched recursively).",
    )
    p.add_argument(
        "--output", "-o",
        default=None,
        help="Output CSV path (default: data/dataset_real_HHMMSS.csv).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    recordings_dir = Path(args.recordings_dir)
    if not recordings_dir.exists():
        print(f"ERROR: recordings directory not found: {recordings_dir}")
        sys.exit(1)

    if args.output is None:
        ts = datetime.now().strftime("%H%M%S")
        output_path = Path(_REPO) / "data" / f"dataset_real_{ts}.csv"
    else:
        output_path = Path(args.output)

    parquet_files = sorted(recordings_dir.rglob("real_*.parquet"))
    print(f"Found {len(parquet_files)} parquet files under {recordings_dir}")

    if not parquet_files:
        print("No parquet files found. Exiting.")
        sys.exit(1)

    all_dfs = []
    stale_files = []
    for bag_id, path in enumerate(parquet_files):
        print(f"  [{bag_id:3d}] {path.relative_to(recordings_dir)}")

        df = pd.read_parquet(path)

        # Detect pre-fix recordings: tau_motor was wrongly copied from tau_link
        # (F/T sensor) instead of holding the real commanded joint effort.
        if np.allclose(
            df[["tau_motor_x", "tau_motor_y", "tau_motor_z"]].to_numpy(),
            df[["tau_link_x", "tau_link_y", "tau_link_z"]].to_numpy(),
        ):
            stale_files.append(path.name)

        # ── column mapping ──────────────────────────────────────────────────
        # Joint positions: use motor-encoder values (q_link == q_motor on real robot)
        df["q0"] = df["q_motor_x"]
        df["q1"] = df["q_motor_y"]
        df["q2"] = df["q_motor_z"]

        # Joint velocities
        df["dq0"] = df["dq_motor_x"]
        df["dq1"] = df["dq_motor_y"]
        df["dq2"] = df["dq_motor_z"]

        # Commanded joint effort, read from /joint_states (msg.effort)
        df["tau1"] = df["tau_motor_x"]
        df["tau2"] = df["tau_motor_y"]
        df["tau3"] = df["tau_motor_z"]

        # End-effector forces from F/T sensor
        df["fx"] = df["tau_link_x"]
        df["fy"] = df["tau_link_y"]
        df["fz"] = df["tau_link_z"]

        # Metadata (mirrors sim preprocessing conventions)
        df["bag"] = bag_id
        df["experiment"] = path.stem

        all_dfs.append(df)

    dataset = pd.concat(all_dfs, ignore_index=True)
    dataset = dataset.sort_values(["bag", "t"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)

    print(f"\nTotal samples : {len(dataset)}")
    print(f"Bags          : {dataset['bag'].nunique()}")
    print(f"Output        : {output_path}")

    if stale_files:
        print(
            f"\nWARNING: {len(stale_files)} file(s) have tau_motor == tau_link "
            "(pre-fix recordings — tau1/tau2/tau3 are F/T-sensor duplicates, "
            "NOT real commanded effort). Consider re-recording these:"
        )
        for name in stale_files:
            print(f"  - {name}")


if __name__ == "__main__":
    main()
