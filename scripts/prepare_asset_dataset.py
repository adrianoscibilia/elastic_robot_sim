#!/usr/bin/env python3
"""Convert a local KUKA/Baxter source dataset to canonical calibration CSV."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(_REPO / "src"))

from elastic_sim.asset_dataset import load_asset_rollouts, rollout_to_dataframe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Raw MAT/CSV file or directory")
    parser.add_argument("--format", required=True, choices=("kuka_raw_mat", "baxter_csv"))
    parser.add_argument("--output", required=True, help="Canonical CSV output path")
    parser.add_argument("--sample-time", type=float, default=None, help="Required for timestamp-free Baxter CSV")
    parser.add_argument("--joints", default=None, help="Optional comma-separated public joint order")
    args = parser.parse_args()
    joints = tuple(item.strip() for item in args.joints.split(",") if item.strip()) if args.joints else None
    rollouts = load_asset_rollouts(
        args.input, joint_names=joints, dataset_format=args.format, sample_time_s=args.sample_time,
    )
    import pandas as pd
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.concat(
        [rollout_to_dataframe(rollout, bag=f"rollout_{index:03d}") for index, rollout in enumerate(rollouts)],
        ignore_index=True,
    )
    frame.to_csv(output, index=False)
    print(f"Wrote {len(frame)} samples from {len(rollouts)} rollout(s) to {output}")


if __name__ == "__main__":
    main()
