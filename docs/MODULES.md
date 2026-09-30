# Module and script guide

The live pipeline builds identification datasets (see
[IDENTIFICATION_DATASET.md](IDENTIFICATION_DATASET.md)). Everything listed
here is reachable from its entry points. The pre-identification
sim-to-real/calibration stack is archived in `legacy/`; see
`legacy/README.md`.

## `src/elastic_sim/`

| Module | Responsibility |
|---|---|
| `dataset.py` | Config (`DatasetConfig`, `load_config`), robot/payload/regime/gain sampling, splits, the per-bag work list, trajectory cache, `run_condition`/`_run_bag`, the unstable-bag guard, `rollout_frame`, `generate`, `write_dataset` and the consumer contract. |
| `torque_runners.py` | Torque-driven rigid and elastic rollouts in MuJoCo and Newton with exact labels; `RolloutDiverged` on MuJoCo instability warnings or non-finite state; link-inertia envelopes; `TransmissionSpec`. |
| `controllers.py` | Controller modes (`exact_ct`, `nominal_ct`, `pd_gravity`, `pd`, `velocity_pi`), their per-trajectory gain draws, and the nominal model a controller is allowed to know. |
| `excitation.py` | Fourier excitation design: limit fitting, the modal probe, regressor conditioning, collision-checked candidate selection. |
| `identification.py` | Pinocchio inverse dynamics, base-parameter regressor and friction model: the analytic reference independent of both simulators. |
| `measurement.py` | Sensor model applied to recorded channels: quantization, noise, gain error, delay; round 6's percentage `NoiseModel` and the in-loop `LoopInstrument` the controller reads. |
| `plant_extras.py` | Non-Lagrangian plant effects: link friction, nonlinear spring (deflection or torque knees), torque ripple, the harmonic drive's transmission error. |
| `dataset_bounds.py` | Round 6's load-time explicit-integration bound, `(b + c/eps + d) h / M_eff <= 1` per DOF, and the `auto` physics steps derived from it. |
| `link_modes.py` | Round 6's link-side mode table `f_link = sqrt(K / M_link) / 2 pi`, the elastically observable joints, and the probe comb aimed at them. |
| `round6_checks.py` | Round 6's controller-influence gates and per-file hard checks, run on a written dataset. |
| `wrench.py` | End-effector force/torque cell and its `J(q)^T` mapping to a per-joint target. |
| `payload.py` | Flange payload injected into the URDF so every consumer sees the same plant. |
| `diagnostics.py` | Q-C measurements per bag: collinearity, conditioning, residual decomposition, probe-band content, deflection SNR, linear baselines. |
| `backend_comparison.py` | Per-pair MuJoCo/Newton agreement, on the clean columns when a dataset has them. |
| `kinematics.py` | Pinocchio/Pink kinematics and Coal self-collision validation (`validate_path`, bisection-capped). |
| `materialized.py` | `MaterializedTrajectory`: exact time/position/velocity/acceleration arrays, analytic evaluation, digests. |
| `assets.py` | Asset YAML loading, URDF joint discovery, the repository asset registry. |
| `scene.py` | Compose an asset and a table into a derived scene asset. |
| `generic_mujoco_runner.py`, `generic_newton_runner.py` | URDF-backed MuJoCo/Newton model builders (elastic transmissions, mesh proxies) used by the torque runners and the viewer. |
| `generic_calibration.py` | Torque-replay rollout container used by the Newton builder. |
| `serial_trajectory.py` | URDF-limit helpers and standalone serial-arm trajectories. |
| `visualization.py` | Native MuJoCo/Newton viewer adapters with path overlays (`--visualize`). |

## Scripts

| Script | Use |
|---|---|
| `build_round6.py` | **Round-6 build**: production, gain-shift and ablation files from one schema-3 config, gated; a failing file is written `*.refused.parquet`. |
| `round6_budget.py` | Round 6's contribution budget: every plant effect toggled once on 2 robots, variance shares of `tau_s`, gated on the elastically observable joints (`R6_02` P2-2). |
| `round6_probe_fraction.py` | Chooses a round-6 platform's probe acceleration fraction: raised from 0.2 until peak `|tau|/effort` reaches 0.8, capped at 0.7. |
| `diagnose_controller_modes.py` | **Round-5 production build**: matched datasets per controller mode (± plant extras), trajectories optimized once and reused, parallel bags, Q-C diagnostics, baselines written into each contract. |
| `generate_identification_dataset.py` | One dataset at a config's own settings. |
| `run_identification_simulation.py` | One rollout, viewer and diagnostics, no dataset written; reproduces any bag by `--tier/--trajectory`. |
| `audit_backends.py` | MuJoCo vs Newton on a 3-robot build, clean columns; not part of any launch. |
| `compare_identification_backends.py` | Re-check the backend pairs of a written dataset. |
| `report_parameter_bounds.py`, `run_range_sensitivity.py`, `sweep_ft_payload.py` | Studies behind the priors. |
| `compose_scene_urdf.py`, `compile_urdf_from_description.py`, `view_urdf_mujoco.py`, `generate_serial_trajectory.py` | Asset preparation and inspection. |
| `round5_weekend.sh`, `launch_helpers.py` | Unattended doctor → preflight → launch runner and its helpers (probe check, dataset stems, stage timeouts). |

## Where to implement common changes

- A new controller mode: `controllers.py` (`CONTROLLER_MODES`, `build_controller`) and its gain draw; tests in `tests/test_round5_controller.py`.
- A new dataset column or contract field: `rollout_frame`/`write_dataset` in `dataset.py`, `docs/IDENTIFICATION_DATASET.md`, and the consumer in `dynamic_model_nn/dataset.py`.
- A new platform: an asset under `assets/robots/`, a scene via `compose_scene_urdf.py`, a config under `config/identification/`, and an entry in the launcher's `CONFIG` table.
- A new prior: the config, with its provenance class in `docs/PARAMETER_PROVENANCE.md`.
