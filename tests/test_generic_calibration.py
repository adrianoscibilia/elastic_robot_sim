import os
import sys

import numpy as np
import pandas as pd

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_REPO, "src"))

from elastic_sim.generic_calibration import (
    ParameterSpace,
    ParameterSpec,
    TorqueReplayCalibrationProblem,
    TorqueReplayRollout,
    compare_torque_replay,
)


def _rollout():
    t = np.arange(5, dtype=float) * 0.01
    q = np.column_stack((t, 2.0 * t))
    return TorqueReplayRollout(t, q, np.ones_like(q), np.ones_like(q), ("j1", "j2"))


def test_state_metric_uses_dynamic_joint_count():
    rollout = _rollout()
    report = compare_torque_replay({"time": rollout.time, "q_link": rollout.q, "dq_link": rollout.dq}, rollout)
    assert report["metric"] == 0.0
    assert set(report["per_joint"]) == {"j1", "j2"}


def test_parameter_space_and_runner_protocol():
    class PerfectRunner:
        def run_torque_replay(self, params, rollout):
            return {"time": rollout.time, "q": rollout.q, "dq": rollout.dq}

    problem = TorqueReplayCalibrationProblem(
        PerfectRunner(), [_rollout()], ParameterSpace([ParameterSpec("joint_1.stiffness", 10.0, 100.0)])
    )
    assert problem.loss(np.array([0.0])) == 0.0
    assert np.isclose(problem.parameter_space.decode(np.array([0.0]))["joint_1.stiffness"], np.sqrt(1000.0))
