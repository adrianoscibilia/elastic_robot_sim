import numpy as np
import pytest
from erd_recording.identification_plan import velocity_sweep
from erd_recording.identification import fit_sweep_friction, friction_torque


@pytest.mark.parametrize('speed',[-.35,.35])
def test_velocity_sweep_is_smooth_bounded_and_has_plateau(speed):
    trajectory = velocity_sweep(np.zeros(6),2,speed,acceleration=1.,jerk=15.,plateau_s=.5,dt=.008,
                                joint_names=tuple(f'j{i}' for i in range(6)))
    assert np.max(np.abs(trajectory.acceleration)) <= 1.
    jerk = np.gradient(trajectory.acceleration,trajectory.time,axis=0)
    assert np.max(np.abs(jerk)) <= 15.
    assert np.max(np.abs(trajectory.velocity[[0,-1]])) < 1e-12
    assert np.count_nonzero(np.isclose(trajectory.velocity[:,2],speed)) >= 60
    assert np.max(np.abs(trajectory.position[:,[0,1,3,4,5]])) == 0
    derivative = np.gradient(trajectory.position,trajectory.time,axis=0)
    assert np.max(np.abs(derivative[1:-1]-trajectory.velocity[1:-1])) < .0003


def test_sweep_friction_predicts_heldout_speeds():
    speeds = np.array([.02,.05,.1,.2,.35,.5])
    parameters = dict(viscous=.4,coulomb=1.,stribeck=.3,stribeck_velocity=.1)
    result = fit_sweep_friction(speeds,friction_torque(speeds,parameters))
    heldout = np.linspace(-.5,.5,101)
    assert np.max(np.abs(friction_torque(heldout,result)-friction_torque(heldout,parameters))) < .01
