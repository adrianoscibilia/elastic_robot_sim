"""Old JSON (speed_override!=100, no global_speed_factor) must still replay correctly."""

import os
import sys

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_REPO, "src"))

from elastic_sim.trajectory import TrajectoryConfig, _trajectory_from_config


_OLD_JSON = {
    "mode": 1,
    "sim_time": 15.0,
    "seed": 0,
    "joint_limits": {"x": [-1.8, 1.8], "y": [-1.8, 1.8], "z": [-1.0, 1.0]},
    "step_duration": 2.0,
    "params": {
        "x": {"amp": 0.5, "freq": 1.0, "phase": 1.5707963, "offset": 0.0, "cos": False},
        "y": {"amp": 0.4, "freq": 1.2, "phase": 1.5707963, "offset": 0.0, "cos": True},
        "z": {"amp": 0.3, "freq": 0.8, "phase": -1.5707963, "offset": 0.0, "cos": False},
    },
    "speed_override": 50.0,
    "vel_limit_ms": 0.3,
}


def test_legacy_loads_without_error():
    cfg = TrajectoryConfig.from_dict(_OLD_JSON)
    assert cfg.speed_override == 50.0
    assert cfg.vel_limit_ms == 0.3
    assert cfg.global_speed_factor == 1.0  # field absent in old file → default


def test_legacy_applies_time_warp():
    cfg = TrajectoryConfig.from_dict(_OLD_JSON)
    traj = _trajectory_from_config(cfg)
    # At 50% speed, effective_sim_time = 15 * 100/50 = 30
    assert abs(cfg.effective_sim_time - 30.0) < 1e-9

    # Velocity must be scaled by factor 0.5 and clipped at 0.3
    _, dq = traj(5.0)
    assert np.all(np.abs(dq) <= 0.3 + 1e-9)


def test_new_baked_does_not_hit_legacy_path():
    new_json = dict(_OLD_JSON)
    new_json["speed_override"] = 100.0
    new_json["global_speed_factor"] = 0.625
    new_json["nominal_sim_time"] = 15.0
    new_json["vel_limit_ms"] = None
    cfg = TrajectoryConfig.from_dict(new_json)
    # New file: speed_override==100 → legacy branch not entered
    assert cfg.speed_override == 100.0
    assert cfg.global_speed_factor == 0.625
    traj = _trajectory_from_config(cfg)
    # No extra scaling applied — just evaluates the (already-baked) sinusoid
    _, dq = traj(5.0)
    assert dq is not None
