"""Frozen, bounded and collision-checked E-iiwa/E-ur recording segments."""
from __future__ import annotations

import numpy as np
from elastic_sim.materialized import MaterializedTrajectory
from .planning import PlanError, PlanSegment, _rename, ladder_scale_trajectory, quintic_segment


def velocity_sweep(pose, joint, speed, *, acceleration, jerk, plateau_s, dt, joint_names):
    """Cosine velocity ramps bracketing a constant-velocity plateau.

    Analytic position/velocity/acceleration, with zero velocity and acceleration
    at each end. The signed displacement is speed * (ramp + plateau).
    """
    ramp = max(np.pi * abs(speed) / (2 * acceleration),
               np.pi * np.sqrt(abs(speed) / (2 * jerk)), 2 * dt)
    ramp = np.ceil(ramp / dt) * dt
    plateau = np.ceil(plateau_s / dt) * dt
    t = np.arange(round((2 * ramp + plateau) / dt) + 1) * dt
    x = np.minimum(t, ramp)
    distance = speed / 2 * (x - ramp / np.pi * np.sin(np.pi * x / ramp))
    velocity = speed / 2 * (1 - np.cos(np.pi * x / ramp))
    accel = speed * np.pi / (2 * ramp) * np.sin(np.pi * x / ramp)
    flat = (t > ramp) & (t <= ramp + plateau)
    distance[flat] = speed * (t[flat] - ramp / 2)
    velocity[flat], accel[flat] = speed, 0
    down = t > ramp + plateau
    x = t[down] - ramp - plateau
    distance[down] = speed * (ramp / 2 + plateau) + speed / 2 * (x + ramp / np.pi * np.sin(np.pi * x / ramp))
    velocity[down] = speed / 2 * (1 + np.cos(np.pi * x / ramp))
    accel[down] = -speed * np.pi / (2 * ramp) * np.sin(np.pi * x / ramp)
    q = np.tile(pose, (len(t), 1)); dq = np.zeros_like(q); ddq = np.zeros_like(q)
    q[:, joint] += distance; dq[:, joint] = velocity; ddq[:, joint] = accel
    return MaterializedTrajectory(time=t, position=q, velocity=dq, acceleration=ddq,
                                  joint_names=joint_names, metadata={"generator": "velocity_sweep",
                                  "joint": joint, "speed": speed, "plateau_start_s": ramp,
                                  "plateau_end_s": ramp + plateau})


def build_identification_segments(config, world, excitations, joint_names, dt, margin):
    arrays = config.limits.as_arrays(config.joint_order)
    segments = []
    home = np.asarray(config.poses.home, dtype=float)
    previous = home.copy()

    def append(sid, kind, trajectory):
        nonlocal previous
        q = trajectory.position
        if np.any(q < arrays['position_lower'] - 1e-9) or np.any(q > arrays['position_upper'] + 1e-9):
            raise PlanError(f"{sid}: identification position limit")
        if np.any(np.abs(trajectory.velocity) > arrays['velocity'] + 1e-8) or np.any(np.abs(trajectory.acceleration) > arrays['acceleration'] + 1e-8):
            raise PlanError(f"{sid}: identification velocity/acceleration limit")
        jerk = np.gradient(trajectory.acceleration, trajectory.time, axis=0)
        if np.any(np.abs(jerk) > arrays['jerk'] + 1e-6):
            raise PlanError(f"{sid}: identification jerk limit")
        report = world.validate_path(q, margin=margin, max_joint_step=0.01)
        if not report.valid:
            raise PlanError(f"{sid}: collision between {report.closest_pair}")
        segments.append(PlanSegment(sid, kind, trajectory, trajectory.digest()))
        previous = q[-1].copy()

    def approach(target, label):
        trajectory = quintic_segment(previous, target, velocity_limit=config.limits.approach.velocity,
                                     acceleration_limit=config.limits.approach.acceleration, time_step=dt)
        jerk = np.max(np.abs(np.gradient(trajectory.acceleration, trajectory.time, axis=0)), axis=0)
        stretch = max(1., float(np.max(jerk / arrays['jerk'])) ** (1 / 3))
        if stretch > 1:
            trajectory = MaterializedTrajectory(
                time=trajectory.time * stretch, position=trajectory.position,
                velocity=trajectory.velocity / stretch, acceleration=trajectory.acceleration / stretch**2,
                joint_names=joint_names, metadata=trajectory.metadata,
            )
        append(label, 'identify_approach', _rename(trajectory, joint_names))

    for i, pose in enumerate(config.poses.standstill):
        approach(pose, f'identify_pose_{i}_approach')
        t = np.arange(round(config.recording.standstill_s / dt) + 1) * dt
        q = np.tile(pose, (len(t), 1))
        hold = MaterializedTrajectory(time=t, position=q, velocity=np.zeros_like(q), acceleration=np.zeros_like(q),
                                      joint_names=joint_names, metadata={'generator': 'identify_hold'})
        append(f'identify_hold_{i}', 'identify_hold', hold)
        approach(home, f'identify_pose_{i}_return')

    # Independent first/last recordings are held out. The middle trajectory
    # uses the second optimized excitation, with a different seed/centre.
    bases = [excitations[0]] if config.robot != 'ur10_cb3' else [excitations[0], excitations[-1], excitations[0]]
    for i, base in enumerate(bases):
        trajectory = ladder_scale_trajectory(base.trajectory, 0.25)
        approach(trajectory.position[0], f'identify_excitation_{i}_approach')
        append(f'identify_excitation_{i}', 'identify_excitation', trajectory)
        approach(home, f'identify_excitation_{i}_return')

    if config.robot == 'ur10_cb3':
        speeds = np.asarray(config.identification.sweep_speeds)
        peak = np.max([np.max(np.abs(s.trajectory.velocity), axis=0) for s in excitations], axis=0)
        for joint, name in enumerate(joint_names):
            bounds = config.identification.sweep_range[name]
            lower = max(bounds[0], arrays['position_lower'][joint])
            upper = min(bounds[1], arrays['position_upper'][joint])
            centre = home.copy(); centre[joint] = (lower + upper) / 2
            effective = speeds * min(1., peak[joint] / max(speeds), arrays['velocity'][joint] / max(speeds))
            for k, speed in enumerate(effective):
                kwargs = dict(acceleration=arrays['acceleration'][joint], jerk=arrays['jerk'][joint],
                              plateau_s=0.5, dt=dt, joint_names=joint_names)
                probe = velocity_sweep(centre, joint, speed, **kwargs)
                span = probe.position[-1, joint] - centre[joint]
                if span > upper - lower:
                    raise PlanError(f'identify sweep {name}: ramp/plateau do not fit configured range')
                start = centre.copy(); start[joint] -= span / 2
                approach(start, f'identify_sweep_{joint}_{k}_approach')
                forward = velocity_sweep(start, joint, speed, **kwargs)
                append(f'identify_sweep_{joint}_{k}_positive', 'identify_sweep', forward)
                reverse = velocity_sweep(previous, joint, -speed, **kwargs)
                append(f'identify_sweep_{joint}_{k}_negative', 'identify_sweep', reverse)
            approach(home, f'identify_sweep_{joint}_return')
    return tuple(segments)
