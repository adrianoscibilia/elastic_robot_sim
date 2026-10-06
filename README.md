# elastic_robot_sim

`elastic_robot_sim` builds simulated identification datasets for elastic
robots. It mounts a robot on a scene, designs excitation trajectories that
condition the rigid-body regressor and probe the transmission modes, runs
them as torque-driven rollouts in MuJoCo (with Newton available as an audit
backend), and writes one flat Parquet file per build. A consumer contract
sidecar travels with each file and is read by
[`dynamic_model_nn`](../dynamic_model_nn).

Each elastic joint is modelled as a motor DOF, a transmission spring and a
link DOF. Robots are sampled rather than laddered: stiffness, damping ratio,
reflected rotor inertia, payload, controller gains and excitation regime are
drawn per robot or per trajectory from seeded, independent streams. A rigid
reference tier anchors every build. The recorded torques are exact by
construction, and the sensor model (quantization, noise, gain error, delay)
is applied to the recorded channels afterwards. Clean copies are kept when a
config asks for them.

## Install

```bash
uv sync                 # add --group dev for pytest
```

MuJoCo runs on a CPU. Newton needs a CUDA GPU and is used only by
`scripts/audit_backends.py`.

## ROS 2 real-data recording (iiwa, UR10)

`src/ros2/` holds the ROS 2 stack that records real validation datasets in
the same contract. On Ubuntu 24.04, from a fresh clone:

```bash
source workspace_setup.sh --deps   # first time: ROS 2 Jazzy + deps, workspace, build
source workspace_setup.sh          # every new terminal
```

The colcon workspace (`ros2_ws/`) and the recordings (`data/real_robot/`)
are generated inside the repository and gitignored. Details:
`src/ros2/README.md`.

## The pipeline

```bash
# 1. Mount an asset on a table and register the scene asset (once per platform)
uv run python scripts/compose_scene_urdf.py \
  --asset kuka_lbr_iiwa_14_r820 --table-height 0.75 --write-asset-yaml

# 2. Watch one rollout, save nothing: trajectory, controller and numbers
uv run python scripts/run_identification_simulation.py --visualize

# 3a. Build matched datasets, one per controller mode, plus the Q-C diagnostics
uv run python scripts/diagnose_controller_modes.py --generate --full \
  --config config/identification/ur10_table_round5.yaml --modes velocity_pi \
  --data-dir data/identification/ur10 --out reports/ur10/qc

# 3b. ...or one dataset at the config's own settings
uv run python scripts/generate_identification_dataset.py \
  --config config/identification/ur10_table.yaml

# 4. Audit MuJoCo against Newton on a small build (not part of production)
uv run python scripts/audit_backends.py --config config/identification/ur10_table_round5.yaml
```

`diagnose_controller_modes.py` is the production build path. A default
build is MuJoCo only. It runs trajectory optimization and bags on
`nproc - 1` worker processes (`--jobs`), and it optimizes each trajectory once
and reuses it for every mode × plant-extras row. The manifests record the
trajectory digests, and the script asserts they are identical across rows.
A bag whose rollout diverges is excluded from the file and listed under
`unstable_bags` in the manifest. Divergence means MuJoCo's `BADQACC`/`BADQPOS`/`BADQVEL`
warnings, a non-finite state, or a joint beyond ten times its span.

### Unattended runs

`scripts/round5_weekend.sh` runs datasets, Optuna screening, export,
training and evaluation as resumable stages:

```bash
PLATFORMS=ur10 MODES=velocity_pi bash scripts/round5_weekend.sh doctor     # GO / NO-GO
PLATFORMS=ur10 MODES=velocity_pi bash scripts/round5_weekend.sh preflight  # every stage, tiny
PLATFORMS=ur10 MODES=velocity_pi bash scripts/round5_weekend.sh launch     # detached
```

`launch` refuses in each of these cases:
- another run holds the lock
- the doctor fails
- no green preflight exists for the same git heads, dirty-tree digest and scope

Stage timeouts are sized from that preflight's measured rates.

## What a build writes

For `<stem>_<mode>[_noextras].parquet`:

| file | content |
|---|---|
| `.parquet` | one row per sample on a uniform grid: `q*`/`dq*` (the side `dataset.signals.position_side` selects), `tau*` (motor command), `ft*` (target), motor/link/deflection channels, optional `*_clean*` copies, and per-bag metadata columns |
| `.contract.json` | what the consumer needs: target kind, semantics and instrument, the Savitzky-Golay window (`differentiation`), signal sides, the split, and the per-split linear baselines (`baselines`) |
| `.manifest.json` | every bag's draws and diagnostics, the robots, the trajectory digests, `unstable_bags` |

[docs/IDENTIFICATION_DATASET.md](docs/IDENTIFICATION_DATASET.md) describes
the design, the configuration and the contract in full.
[docs/PARAMETER_PROVENANCE.md](docs/PARAMETER_PROVENANCE.md) records where
every prior comes from.

## Repository map

```text
config/identification/*.yaml       dataset configurations (one per platform/round)
assets/robots/                     robot descriptions and derived scene assets
scripts/diagnose_controller_modes.py  matched per-mode builds + Q-C diagnostics (production)
scripts/generate_identification_dataset.py  one build at the config's settings
scripts/run_identification_simulation.py    one rollout, viewer and diagnostics, no output
scripts/audit_backends.py          MuJoCo vs Newton on clean columns (audit only)
scripts/compare_identification_backends.py  re-check the backend pairs of a written file
scripts/report_parameter_bounds.py, run_range_sensitivity.py, sweep_ft_payload.py  prior studies
scripts/compose_scene_urdf.py      mount an asset on a table as a new scene asset
scripts/round5_weekend.sh, launch_helpers.py  unattended launcher
src/elastic_sim/                   the package; see docs/MODULES.md
tests/                             live test suite (`-m slow` for the physics tests)
legacy/                            archived sim2real/calibration code, see legacy/README.md
```

## Tests

```bash
uv run python -m pytest -q -m "not slow"   # ~1 min
uv run python -m pytest -q -m slow         # ~5 min, integrates physics
```

`pytest` collects `tests/` only. The archived suite runs with
`pytest legacy/tests`.

## Documentation

- [Identification datasets](docs/IDENTIFICATION_DATASET.md): excitation design, torque-driven rollouts, backend parity, the dataset contract.
- [Parameter provenance](docs/PARAMETER_PROVENANCE.md): priors and their classes.
- [Module guide](docs/MODULES.md): what each live module and script does.
- [Asset provenance](assets/README.md): robot descriptions, licenses and sources.
- Archived sim-to-real workflow (code in `legacy/`): [workflow](docs/WORKFLOW.md), [configuration](docs/CONFIGURATION.md), [data format](docs/DATA_FORMAT.md), [ROS 2](docs/ROS2.md), [development](docs/DEVELOPMENT.md).
