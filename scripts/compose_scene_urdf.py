#!/usr/bin/env python3
"""Generate a table-mounted scene asset from a portable robot asset.

Example
-------
Mount the KUKA LBR iiwa 14 on a 0.75 m table and register it as an asset::

    python scripts/compose_scene_urdf.py --asset kuka_lbr_iiwa_14_r820 \
        --table-height 0.75 --write-asset-yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(_REPO / "src"))

from elastic_sim.assets import AssetRegistry, load_asset_spec
from elastic_sim.scene import write_table_scene


def _load_asset(reference: str):
    candidate = Path(reference)
    if candidate.is_file():
        return load_asset_spec(candidate)
    return AssetRegistry.for_repository(_REPO).load(reference)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--asset", required=True, help="Source asset name or asset.yaml path")
    parser.add_argument("--scene-name", default=None, help="Generated asset name (default: <asset>_table)")
    parser.add_argument("--table-height", type=float, default=0.75)
    parser.add_argument("--table-length", type=float, default=1.2)
    parser.add_argument("--table-width", type=float, default=0.8)
    parser.add_argument("--table-mass", type=float, default=60.0)
    parser.add_argument("--output-root", default=None, help="Asset directory (default: assets/robots/<scene-name>)")
    parser.add_argument("--write-asset-yaml", action="store_true", help="Also write the asset.yaml descriptor")
    args = parser.parse_args()

    asset = _load_asset(args.asset)
    asset.resolve_active_joints()
    scene_name = args.scene_name or f"{asset.name}_table"
    root = Path(args.output_root) if args.output_root else _REPO / "assets" / "robots" / scene_name
    urdf_path = root / "description" / f"{scene_name}.urdf"

    written = write_table_scene(
        asset,
        urdf_path,
        table_height=args.table_height,
        table_size=(args.table_length, args.table_width),
        table_mass=args.table_mass,
        scene_name=scene_name,
    )
    print(f"Wrote {written}")

    if args.write_asset_yaml:
        import yaml

        joints = list(asset.joint_names)
        # Pinocchio merges fixed joints, so the table and the robot base share
        # the universe joint and are auto-excluded as adjacent.  Links beyond
        # the shoulder are genuinely checked against the table top.
        descriptor = {
            "asset": {
                "name": scene_name,
                "urdf": os.path.relpath(written, root).replace(os.sep, "/"),
                "active_joints": joints,
                "gravity": list(asset.gravity),
                "self_collisions": True,
                "default_configuration": list(
                    asset.metadata.get("default_configuration", [0.0] * len(joints))
                ),
                "kinematic_groups": asset.metadata.get("kinematic_groups", {}),
                "collision": {"margin": 0.01, "max_joint_step": 0.05},
                "scene": {
                    "table_height": args.table_height,
                    "table_size": [args.table_length, args.table_width],
                    "table_mass": args.table_mass,
                    "source_asset": asset.name,
                },
            }
        }
        path = root / "asset.yaml"
        path.write_text(yaml.safe_dump(descriptor, sort_keys=False), encoding="utf-8")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
