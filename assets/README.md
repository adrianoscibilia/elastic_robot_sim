# Robot assets

Robot descriptions are stored under `assets/robots/`. An asset definition supplies the URDF, the active-joint order, optional base pose, gravity, collision policy, and mesh/package-path metadata. The same definition is used by trajectory generation, Newton, MuJoCo, and experiment validation.

## Bundled assets

| Registry name | Description | Notes |
|---|---|---|
| `fmrr_tecnobody` | Three-axis Cartesian FMRR/Tecnobody platform | Uses the dedicated Cartesian elastic model; active joints are `joint_x`, `joint_y`, `joint_z`. |
| `ur10` | Universal Robots UR10 | Generic serial-asset model; active joints come from its URDF. |
| `baxter_left` / `baxter_right` | Baxter arm chains | Generic serial-asset model; choose the arm explicitly. |
| `kuka_kr300_r2500_ultra_se` | KUKA KR300 structural approximation | The bundled geometry/inertia are an approximation, not manufacturer CAD. |

The registry can be inspected indirectly with:

```bash
uv run python scripts/run_asset_simulation.py --asset ur10 --dry-run
```

## Asset YAML

A minimal asset file looks like:

```yaml
asset:
  name: ur10
  urdf: ../../assets/robots/ur10/description/ur10.urdf
  active_joints: []   # empty means every non-mimic 1-DoF URDF joint
  base:
    position: [0.0, 0.0, 0.0]
    quaternion: [0.0, 0.0, 0.0, 1.0]
  self_collisions: false
```

Active joints are ordered exactly as listed. That order becomes the public order in `trajectory.json`, Parquet columns, simulator rollouts, and ROS goals. A missing, duplicate, mimic, or non-1-DoF joint is rejected.

Relative URDF and mesh paths are resolved from the asset YAML/URDF, not from the process working directory. `package_roots` metadata can map `package://...` mesh references when needed.

## Provenance and datasets

Each bundled robot description has a `PROVENANCE.md` with source, license, checksum, and retrieval notes. The files under `assets/datasets/` are historical benchmark material and are not required by the primary sim-to-real workflow. Calibration consumes only complete paired experiment artifacts under `data/experiments/`.
