#!/usr/bin/env python3
"""Generate the S0-S3 datasets for the stiffness-range sensitivity study.

The question this answers: does a model's performance depend on the exact
stiffness range it was trained on, or does it generalize? Per
``REFACTOR_SPECS/round3/R3_04_RANGE_JUSTIFICATION_AND_PROVENANCE.md`` Sec A4:

| Run | Train robots drawn from | Test robots drawn from |
|---|---|---|
| S0 (baseline) | nominal x [1/2, 2]   | held-out draws from the same |
| S1 (narrow)   | nominal x [1/1.4, 1.4] | nominal x [1/2, 2] |
| S2 (shifted)  | nominal x [1/2, 2]   | nominal x [1/4, 1/2] (softer) |
| S3 (shifted)  | nominal x [1/2, 2]   | nominal x [2, 4] (stiffer) |

Each variant is two datasets -- a train-side and a test-side, drawn from
different ``stiffness_factor`` ranges and different seeds so they cannot
collide.  This script only generates them and writes an index; it
deliberately does not call into ``dynamic_model_nn`` (a separate repository)
to train or evaluate anything, so the coupling between the two repos stays
one-directional.  It prints the snippet to run there instead.

Examples
--------
    python scripts/run_range_sensitivity.py --robots 8 --trajectories 2
    python scripts/run_range_sensitivity.py --only S1 --output data/sensitivity
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

from elastic_sim.assets import AssetRegistry, load_asset_spec
from elastic_sim.dataset import DEFAULT_CONFIG, build_tiers, generate, load_config, write_dataset

# (name, train_factor, test_factor, seed_offset). Distinct seed offsets keep
# a variant's train and test robots from being drawn from the same stream
# position -- they must be genuinely different samples, not the same numbers
# relabelled.
VARIANTS = (
    ("S0_baseline", (0.5, 2.0), (0.5, 2.0), 0),
    ("S1_narrow", (1.0 / 1.4, 1.4), (0.5, 2.0), 1),
    ("S2_softer", (0.5, 2.0), (0.25, 0.5), 2),
    ("S3_stiffer", (0.5, 2.0), (2.0, 4.0), 3),
)


def _load_asset(reference: str):
    candidate = Path(reference)
    if candidate.is_file():
        return load_asset_spec(candidate)
    return AssetRegistry.for_repository(_REPO).load(reference)


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else _REPO / candidate


def _build_variant_config(base, robots: int, trajectories: int, factor: tuple[float, float], seed: int):
    transmission = replace(base.transmission, robots=robots, stiffness_factor=factor)
    tiers = build_tiers(base.rigid_reference, transmission, seed)
    return replace(base, transmission=transmission, tiers=tiers, n_trajectories=trajectories, seed=seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--asset", default=None)
    parser.add_argument("--robots", type=int, default=8, help="Robots per side (train and test drawn separately)")
    parser.add_argument("--trajectories", type=int, default=2)
    parser.add_argument("--seed", type=int, default=900001, help="Base seed; each variant/side offsets from this")
    parser.add_argument("--output", default="data/identification/sensitivity")
    parser.add_argument("--only", default=None, help="Generate a single variant by name, e.g. S1_narrow")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    base = load_config(_resolve(args.config))
    asset = _load_asset(args.asset or base.asset)
    asset.resolve_active_joints()
    asset.validate_resources()
    output_dir = _resolve(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = [v for v in VARIANTS if args.only is None or v[0] == args.only]
    if not variants:
        parser.error(f"--only {args.only!r} does not match any of {[v[0] for v in VARIANTS]}")

    index: dict[str, dict[str, str]] = {}
    for name, train_factor, test_factor, offset in variants:
        entry = {}
        for side, factor, side_offset in (("train", train_factor, 0), ("test", test_factor, 100)):
            config = _build_variant_config(
                base, args.robots, args.trajectories, factor, args.seed + offset * 1000 + side_offset,
            )
            frame, manifest, comparison = generate(config, asset, verbose=not args.quiet)
            csv_path, *_ = write_dataset(frame, manifest, output_dir / f"{name}_{side}.csv", comparison)
            entry[side] = str(csv_path)
            print(f"{name} [{side}]: stiffness_factor={factor} -> {csv_path}")
        index[name] = entry

    index_path = output_dir / "index.json"
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"\nWrote {index_path}")

    print("\nTo train and evaluate each variant in dynamic_model_nn (run there, not here):")
    for name, entry in index.items():
        print(f"""
# {name}
python - <<'PY'
from dataset import CustomDataset, create_loaders
from model_workflow import DEFAULT_SETTINGS, train_model  # adjust to the model family under test
train_set = CustomDataset("{entry['train']}")
test_set = CustomDataset("{entry['test']}")
train_loader, _ = create_loaders(train_set)
_, held_out_loader = create_loaders(test_set)
# model, _, _, losses, model_name = train_model(YourModelClass, "{name}", "{entry['train']}", DEFAULT_SETTINGS)
# evaluate `model` against held_out_loader and report the degradation vs. S0
PY""")


if __name__ == "__main__":
    main()
