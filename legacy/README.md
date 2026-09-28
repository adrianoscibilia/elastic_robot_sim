# legacy/: archived code (R5_10 T-6)

This is code that no live entry point reaches. It was archived rather than
deleted, and it is excluded from the package and from the default pytest
collection (`testpaths = ["tests"]`).

| path | what it was |
|---|---|
| `src/elastic_sim_legacy/` | The sim-to-real/calibration stack: `experiment`, `sim2real`, `ros_experiment`, `ros_recorder`, `parameter_registry`, `calibration`, `compare`, `rollout`, `trajectory`, `params`, `sim_runner`, `mujoco_runner`, `elastic_settings`, `plotting`, `moveit_validation`, `optimizers/`. It imports the live `elastic_sim` for the modules still in use. |
| `scripts/` | Their entry points (`run_experiment`, `run_calibration`, `record_rollouts`, …) and the standalone FMRR simulator. |
| `real_robot/`, `package.xml`, `CMakeLists.txt` | ROS 2 tooling. The owner decides its future in the post-training chain. |
| `tests/` | Tests of the code above. Run them with `pytest legacy/tests`. |

## The FMRR simulator

`scripts/elastic_cart_robot_{newton,mujoco,isaacsim}.py` stay runnable. They
are the reference for the motor-link / load-link measurement pattern that
round 6 generalises:

```bash
uv run python legacy/scripts/elastic_cart_robot_newton.py --headless
uv run python legacy/scripts/elastic_cart_robot_mujoco.py --headless
```

They read `config/settings.yaml`, `config/elastic_cart_mujoco.xml` and the
`fmrr_tecnobody` asset from the repository root.
