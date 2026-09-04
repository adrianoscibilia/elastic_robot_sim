# Development guide

## Environment

The project requires Python 3.11–3.13 and uses uv. The repository is configured as a non-installed Python project, so the scripts add `src/` when run directly.

```bash
uv sync --all-groups
uv run python -m pytest -q
python -m compileall -q src scripts
```

`pyarrow` is a core dependency because Parquet is part of the experiment contract. CMA-ES, `scikit-optimize`, skrl, Gymnasium, and PyTorch are in the `calibration` dependency group.

## Test layers

The tests cover:

- asset and URDF joint validation;
- YAML defaults and overrides;
- dynamic active-joint counts and ordering;
- exact trajectory JSON round trips;
- shared saved samples across simulator/ROS command paths;
- parameter bounds, normalization, and model overrides;
- optimizer initialization and short budgets;
- experiment storage/discovery and complete/incomplete filtering;
- simulator outputs and finite-value validation;
- ROS topic manifests and capture behavior without requiring a live ROS graph.

Real ROS execution is not part of ordinary pytest because it requires a live controller, sensor, TF graph, rosbag2, and robot.

## Adding an asset

1. Put the URDF and required meshes under `assets/robots/<asset>/`.
2. Add `PROVENANCE.md` with source/license/checksum information.
3. Create an asset YAML with `name`, `urdf`, and an explicit `active_joints` order where automatic discovery is not appropriate.
4. Run `run_asset_simulation.py --asset <name> --dry-run`.
5. Add `config/assets/<name>_sim2real.yaml` with trajectory, model, calibration, and ROS settings.
6. Run the sim2real dry run and at least one MuJoCo simulation.
7. Add tests for the expected active-joint names, trajectory shape, and backend output.

For a serial robot, `active_joints` must contain URDF revolute, continuous, or prismatic non-mimic joints. The declared order is part of the data contract and must match the controller's expected joint order.

## Adding a parameter

Prefer a named registry entry over a new positional calibration vector. Define physical lower/upper bounds, an initial value, scaling, enabled state, and backend target in YAML. Then implement the mapping in the relevant backend adapter.

Mass and inertia changes may require rebuilding a simulator model for an evaluation. Elastic gains can be updated in place only when the backend refreshes any derived solver state correctly. Keep geometry, gravity, solver choice, and integration conditions fixed while identifying dynamic parameters.

## Adding a backend

The backend must accept a `MaterializedTrajectory` and expose equivalent signals:

```python
run_reference_trajectory(
    trajectory: MaterializedTrajectory,
    params: Mapping[str, float],
) -> SimulationRollout
```

The current adapters normalize backend-specific results through `rollout_to_frame()`. A backend must use the provided trajectory times and arrays; generating a new trajectory from the seed creates an invalid comparison.

## Documentation conventions

- Document commands from the repository root.
- Keep one canonical sim2real path in user-facing documentation.
- Label compatibility and synthetic-dataset utilities as secondary.
- Document units and joint ordering whenever adding a signal or parameter.
- Update tests and the relevant guide when adding a configuration field.
