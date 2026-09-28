"""One identification bag: run a condition and flatten the rollout into the consumer's schema.

Split out of ``dataset.py`` (`R5_10` T-7) with no behaviour change; import
from :mod:`elastic_sim.dataset`.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from .assets import AssetSpec
from .controllers import ControllerDraw, build_controller, describe_controller
from .excitation import effective_position_window
from .identification import FrictionModel
from .kinematics import PortableKinematics
from .materialized import MaterializedTrajectory
from .measurement import IDEAL_MEASUREMENT, MeasurementModel, measure_bag, measure_channel
from .payload import Payload, payload_asset
from .plant_extras import NO_EXTRAS, PlantExtras
from .torque_runners import (
    RolloutDiverged,
    TransmissionSpec,
    link_inertia_envelope,
    run_mujoco_elastic_torque,
    run_mujoco_torque,
    run_newton_elastic_torque,
    run_newton_torque,
)
from .wrench import ForceTorqueSensor, SensorSpec, WRENCH_COLUMNS
from .dataset_config import DatasetConfig, LinkInertia, SignalPolicy, Tier


def elastic_time_step(transmission: TransmissionSpec, config: DatasetConfig) -> float:
    return min(config.max_time_step, transmission.required_time_step())


def _inertia_envelope_bounds(asset: AssetSpec, config: DatasetConfig) -> tuple[tuple[float, float], ...] | None:
    """Per-active-joint ``(lo, hi)`` for ``link_inertia_envelope``/``link_inertia_max``.

    ``None`` (the whole URDF range) unless ``config.excitation.position_window``
    is set, in which case the envelope is sampled over the same effective
    window the excitation trajectories live in, instead of the full URDF
    range (R4_03 Sec 1) -- for the iiwa (no window) this keeps the historical
    call exactly as it was.
    """
    if not config.excitation.position_window:
        return None
    lower, upper = effective_position_window(asset, config.excitation)
    return tuple(zip(lower.tolist(), upper.tolist()))


def run_condition(
    asset: AssetSpec,
    trajectory: MaterializedTrajectory,
    tier: Tier,
    backend: str,
    friction: FrictionModel,
    config: DatasetConfig,
    *,
    link_inertia: LinkInertia | None = None,
    payload: Payload | None = None,
    natural_frequency: float | None = None,
    damping_ratio: float | None = None,
    draw: ControllerDraw | None = None,
    extras: PlantExtras | None = None,
) -> dict[str, Any]:
    """Execute one trajectory under one condition and return the rollout.

    ``link_inertia`` is computed from the asset when not given; pass it when
    running many conditions, since it is the same for all of them.

    ``payload`` is welded to the asset's last link (see ``payload.py``)
    before anything downstream sees it -- Pinocchio, both simulators and the
    collision checker all consume the same modified asset by construction.
    Excitation trajectories are deliberately scored payload-free (their
    kinematic limits and regressor conditioning do not depend on the
    inertias a payload changes; only the required torque does, which is
    checked separately) -- do not "fix" this without re-reading
    ``R3_01 Sec 2.6``.

    ``natural_frequency``/``damping_ratio`` default to
    ``config.control_frequency``/``config.control_damping_ratio`` when not
    given; pass the per-trajectory draw from ``sample_control_gains`` to use
    a randomized closed-loop gain instead (``config.control_gains``).

    ``draw`` carries the velocity-loop gains as well and supersedes those two
    arguments when given (``resolve_bag`` builds it); ``extras`` carries the
    round-5 non-Lagrangian plant effects.  Both default to round 4's
    behaviour: exact computed torque at the config's fixed gains, on a purely
    Lagrangian link side.

    Plant extras are rejected on the rigid tier on purpose.  That tier is the
    dataset's *analytic reference*: its recorded torque must keep satisfying
    ``rnea(achieved state)`` so that the Pinocchio cross-check and the
    MuJoCo/Newton-Featherstone backend comparison stay meaningful.  Link-side
    friction, a nonlinear spring and torque ripple all break that by
    construction, and on a rigid chain the first two have no separate link
    side to act on anyway.
    """
    natural_frequency = config.control_frequency if natural_frequency is None else natural_frequency
    damping_ratio = config.control_damping_ratio if damping_ratio is None else damping_ratio
    if draw is None:
        loop = config.controller.velocity_loop
        draw = ControllerDraw(
            natural_frequency=natural_frequency, damping_ratio=damping_ratio,
            position_gain=loop.position_gain[0], velocity_bandwidth=loop.velocity_bandwidth[0],
            integral_time=loop.integral_time[0],
        )
    natural_frequency, damping_ratio = draw.natural_frequency, draw.damping_ratio
    extras = NO_EXTRAS if extras is None else extras
    n_dof = len(asset.joint_names)
    view = {"visualize": config.visualize, "realtime_scale": config.realtime_scale}
    probe_top_hz = float(trajectory.metadata.get("probe_top_hz", 0.0))

    def _check_control_rate(time_step: float) -> None:
        if config.control_decimation <= 1 or probe_top_hz <= 0.0:
            return
        control_rate = 1.0 / (time_step * config.control_decimation)
        if control_rate <= 10.0 * probe_top_hz:
            raise ValueError(
                f"control_decimation={config.control_decimation} at time_step={time_step:g}s gives a "
                f"{control_rate:g} Hz control rate, not comfortably above the {probe_top_hz:g} Hz probe "
                "top frequency (need > 10x); lower control_decimation or the probe's top harmonic"
            )

    with payload_asset(asset, payload) as asset_p:
        # What the controller is allowed to know.  `asset` here is always the
        # bare, payload-free asset; `asset_p` is the plant.
        nominal_asset = asset_p if config.controller.nominal.knows_payload else asset
        if tier.is_rigid:
            if not extras.is_empty:
                raise ValueError(
                    "simulation.plant_extras cannot be applied to the rigid reference tier "
                    "(its recorded torque must stay consistent with rnea(achieved state) for the "
                    "Pinocchio and backend cross-checks); set rigid_reference: false or drop the extras"
                )
            _check_control_rate(config.rigid_time_step)
            controller = build_controller(
                config.controller, draw, asset=asset_p, nominal_asset=nominal_asset,
                trajectory=trajectory, friction=friction,
            )
            runner = run_mujoco_torque if backend == "mujoco" else run_newton_torque
            result = runner(asset_p, trajectory, controller, time_step=config.rigid_time_step,
                            friction=friction, control_decimation=config.control_decimation, **view)
            result.update(transmission=None, time_step=config.rigid_time_step, payload=payload,
                         natural_frequency=natural_frequency, damping_ratio=damping_ratio,
                         controller_draw=draw, controller_mode=config.controller.mode,
                         extras=NO_EXTRAS)
            return result
        if link_inertia is None:
            link_inertia = link_inertia_envelope(
                asset_p, n_samples=config.transmission.inertia_samples,
                bounds=_inertia_envelope_bounds(asset_p, config),
            )
        transmission = tier.transmission(n_dof, link_inertia)
        time_step = elastic_time_step(transmission, config)
        _check_control_rate(time_step)
        controller = build_controller(
            config.controller, draw, asset=asset_p, nominal_asset=nominal_asset,
            trajectory=trajectory, friction=friction, transmission=transmission,
            # The PD and velocity-loop gains are sized from a nominal inertia.
            # A payload-*unaware* controller must not get it from the
            # payload-fitted envelope: that would hand it the payload back
            # through its gains.  Passing None makes `build_controller` derive
            # it from the nominal asset instead.
            joint_inertia=link_inertia[0] if config.controller.nominal.knows_payload else None,
        )
        runner = run_mujoco_elastic_torque if backend == "mujoco" else run_newton_elastic_torque
        result = runner(asset_p, trajectory, controller, transmission, time_step=time_step,
                        friction=friction, control_decimation=config.control_decimation,
                        extras=extras, **view)
        result.update(transmission=transmission, time_step=time_step, payload=payload,
                     natural_frequency=natural_frequency, damping_ratio=damping_ratio,
                     controller_draw=draw, controller_mode=config.controller.mode,
                     extras=extras)
        return result


def describe_transmission(transmission: TransmissionSpec | None, time_step: float | None = None) -> dict[str, Any]:
    """JSON-serializable per-joint transmission parameters."""
    if transmission is None:
        return {"stiffness": None, "damping": None, "damping_ratio": None, "rotor_inertia": None,
                "mode_frequency_hz": None, "time_step": time_step}
    return {
        "stiffness": transmission.stiffness.tolist(),
        "damping": transmission.damping.tolist(),
        "damping_ratio": None if transmission.damping_ratio is None else transmission.damping_ratio.tolist(),
        "rotor_inertia": transmission.rotor_inertia.tolist(),
        "mode_frequency_hz": transmission.natural_frequency().tolist(),
        "time_step": time_step,
    }


def rollout_frame(
    asset: AssetSpec,
    trajectory: MaterializedTrajectory,
    result: Mapping[str, Any],
    *,
    bag: str,
    tier: Tier,
    backend: str,
    friction: FrictionModel,
    resample_step: float,
    split: str = "train",
    signals: SignalPolicy | None = None,
    measurement: MeasurementModel | None = None,
    measurement_seed: int = 0,
    sensor: ForceTorqueSensor | None = None,
) -> pd.DataFrame:
    """Flatten one rollout onto a uniform grid in the consumer's schema.

    ``dynamic_model_nn`` differentiates positions with a Savitzky-Golay filter
    and only does so when the time step is uniform, so every bag is resampled
    onto the same grid regardless of the step its tier required.

    ``signals`` selects which physical side ``q0..``/``dq0..`` and ``ft0..``
    carry (``SignalPolicy``, ``R5_01`` Amendment 2); ``measurement`` applies
    the sensor model to the recorded channels *after* resampling, since that
    is the rate a recorder runs at and the rate ``delay_samples`` counts in.
    Both default to round 4: the collocated link/link pair, perfect
    instruments.

    ``sensor`` is required when ``signals.target`` is ``ee_wrench_joint`` and is
    built from the asset when not given; building it costs one Pinocchio model,
    so a caller running many bags should build it once and pass it in.
    """
    names = tuple(asset.joint_names)
    signals = SignalPolicy() if signals is None else signals
    measurement = IDEAL_MEASUREMENT if measurement is None else measurement
    time = np.asarray(result["time"], dtype=float)
    grid = np.arange(time[0], time[-1] + 0.5 * resample_step, resample_step)

    def _on_grid(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        return np.column_stack([np.interp(grid, time, values[:, i]) for i in range(values.shape[1])])

    # The end-effector wrench, when that is the target.  Computed here rather
    # than inside the runners because it is a pure function of the achieved
    # link-side motion, which the rollout already records: the cell measures
    # what the bodies past it do, and nothing about it feeds back into the
    # simulation.  On the *resampled* grid, so it is the wrench a recorder at
    # that rate would log.
    wrench = wrench_target = None
    if signals.needs_sensor:
        if sensor is None:
            sensor_spec = SensorSpec.from_asset(asset)
            if sensor_spec is None:
                raise ValueError(
                    f"dataset.signals.target='{signals.target}' needs a force/torque cell, but asset "
                    f"{asset.name!r} declares no `force_torque_sensor: {{frame: ...}}`; add one to its "
                    "asset.yaml or choose another target"
                )
            sensor = ForceTorqueSensor(asset, sensor_spec)
        link_position = _on_grid(result["q_link"])
        link_velocity = _on_grid(result["dq_link"])
        # The consumer recomputes ddq from the recorded velocity, but the *cell*
        # experiences the true acceleration, so the simulated wrench uses the
        # rollout's own ddq rather than a differentiated copy of it.
        link_acceleration = _on_grid(result["ddq_link"])
        mapping = link_position if signals.position_side == "link" else _on_grid(result["q_motor"])
        wrench, wrench_target = sensor.rollout(
            link_position, link_velocity, link_acceleration, q_mapping=mapping,
        )

    clean = {
        "q_link": _on_grid(result["q_link"]),
        "dq_link": _on_grid(result["dq_link"]),
        "q_motor": _on_grid(result["q_motor"]),
        "dq_motor": _on_grid(result["dq_motor"]),
        "tau_motor": _on_grid(result["tau_motor"]),
        "tau_link": _on_grid(result["tau_link"]),
    }
    measured = measure_bag(
        measurement, seed=measurement_seed,
        q_motor=clean["q_motor"], dq_motor=clean["dq_motor"],
        q_link=clean["q_link"], dq_link=clean["dq_link"],
        tau_motor=clean["tau_motor"], tau_link=clean["tau_link"],
    )
    q = measured.q_link if signals.position_side == "link" else measured.q_motor
    dq = measured.dq_link if signals.position_side == "link" else measured.dq_motor
    if signals.target == "ee_wrench_joint":
        # One instrument, one realization (`R5_06` Sec 3).  The cell measures
        # the 6-axis wrench -- with its own noise and gain error, since it is
        # the same *instrument slot* as a link-side torque sensor but a
        # different transducer -- and the Jacobian is software applied after
        # it, at the configuration the encoders report.  Measuring the wrench
        # and the mapped target as two draws, as pass 2 did, gave a consumer
        # two independent realizations of one physical reading.
        measured_wrench = measure_channel(
            measurement, wrench, kind="force_cell", seed=measurement_seed,
        ).values
        target = sensor.joint_torques(q, measured_wrench)
    elif signals.target == "link_torque":
        target = measured.tau_link
    else:
        target = measured.tau_motor

    # Every column goes into one dict and the frame is built once: inserting
    # ~200 columns one at a time fragmented it (thousands of pandas
    # PerformanceWarnings per build, `R5_10` T-3.5).  Scalars broadcast with
    # the same dtype inference as a column assignment, so the file is
    # byte-identical.
    frame: dict[str, Any] = {"t": grid, "bag": bag}
    for index in range(len(names)):
        frame[f"q{index}"] = q[:, index]
    for index in range(len(names)):
        frame[f"dq{index}"] = dq[:, index]
    for index in range(len(names)):
        frame[f"tau{index}"] = measured.tau_motor[:, index]
    # The learning target: whichever side `signals.target` names, one channel
    # per joint.  Its physical meaning is in the contract sidecar.
    for index in range(len(names)):
        frame[f"ft{index}"] = target[:, index]
    for index in range(len(names)):
        frame[f"q_motor{index}"] = measured.q_motor[:, index]
        frame[f"dq_motor{index}"] = measured.dq_motor[:, index]
        frame[f"q_link{index}"] = measured.q_link[:, index]
        frame[f"dq_link{index}"] = measured.dq_link[:, index]
    # Direct readout of the transmission deflection tau/k -- the cleanest
    # target for identifying the elastic parameters themselves (R3_01 Sec 3).
    # Taken from the *measured* positions, so it is the deflection an observer
    # could actually compute; the clean columns below hold the true one.
    for index in range(len(names)):
        frame[f"defl{index}"] = measured.q_motor[:, index] - measured.q_link[:, index]
    if wrench is not None:
        # The raw 6-axis reading beside the mapped target (R5_03 T-1): a
        # consumer may want the wrench itself, and a diagnostic needs it to
        # separate the mapping from the measurement.  `ft` is exactly
        # `J(q)^T w` of these two sets of columns (`R5_06` Sec 3).
        for index, column in enumerate(WRENCH_COLUMNS):
            frame[column] = measured_wrench[:, index]
        # How much of the wrench the mapping can still carry (`R5_07` T-15).
        # On a 6-DoF arm `J^T` is configuration dependent and near a
        # singularity whole wrench directions stop reaching joint space; this
        # is the per-sample record of that, so a bag can be scored on posture
        # instead of the excitation being assumed non-singular.
        frame["sigma_min_j"] = sensor.smallest_singular_value(q)
    if signals.clean_columns:
        # The exact simulator values on the same grid.  A noisy dataset is only
        # decomposable into physics and instrument if these survive (R5_00
        # Sec 6.2); they are debug columns and never model inputs.  The
        # reference is one of them: it is not a measurable signal at all, but
        # tracking error is the headline difference between controller modes and
        # a diagnostic reading a written file cannot recover it otherwise.
        clean = dict(clean, q_ref=_on_grid(result["q_ref"]), dq_ref=_on_grid(result["dq_ref"]))
        if wrench_target is not None:
            clean = dict(clean, ft=wrench_target)
        for key, values in clean.items():
            for index in range(len(names)):
                frame[f"{key}_clean{index}"] = values[:, index]
    frame["experiment"] = bag
    frame["tier"] = tier.name
    frame["backend"] = backend
    frame["split"] = split
    transmission = describe_transmission(result.get("transmission"))
    for index, name in enumerate(names):
        frame[f"viscous__{name}"] = friction.viscous[index]
        frame[f"coulomb__{name}"] = friction.coulomb[index]
        for key in ("stiffness", "damping", "damping_ratio", "rotor_inertia"):
            values = transmission[key]
            frame[f"{key}__{name}"] = np.nan if values is None else values[index]
    payload: Payload | None = result.get("payload")
    frame["payload_mass"] = np.nan if payload is None else payload.mass
    frame["payload_offset_x"] = np.nan if payload is None else payload.offset[0]
    frame["payload_offset_y"] = np.nan if payload is None else payload.offset[1]
    frame["payload_offset_z"] = np.nan if payload is None else payload.offset[2]
    frame["payload_size"] = np.nan if payload is None else payload.size
    exc_metadata = trajectory.metadata
    frame["exc_base_frequency"] = exc_metadata.get("base_frequency")
    frame["exc_max_acceleration"] = exc_metadata.get("max_acceleration")
    frame["exc_velocity_fraction"] = exc_metadata.get("velocity_fraction")
    frame["exc_probe_top_hz"] = exc_metadata.get("probe_top_hz", 0.0)
    draw: ControllerDraw | None = result.get("controller_draw")
    if draw is None:
        frame["control_natural_frequency"] = result.get("natural_frequency")
        frame["control_damping_ratio"] = result.get("damping_ratio")
    else:
        # Only the gains this mode uses; see ControllerDraw.as_row on why the
        # dataset columns and the manifest's record differ here.
        for column, value in draw.as_row(str(result.get("controller_mode", "exact_ct"))).items():
            frame[column] = value
    if not measurement.is_ideal:
        # The per-bag calibration errors, so a consumer or a diagnostic can
        # tell a gain error from a modelling error.
        for index, name in enumerate(names):
            frame[f"gain_motor__{name}"] = measured.gain_motor[index]
            frame[f"gain_link__{name}"] = measured.gain_link[index]
    extras: PlantExtras | None = result.get("extras")
    if extras is not None and not extras.is_empty:
        link_friction = extras.link_friction
        for index, name in enumerate(names):
            frame[f"link_viscous__{name}"] = 0.0 if link_friction is None else link_friction.viscous[index]
            frame[f"link_coulomb__{name}"] = 0.0 if link_friction is None else link_friction.coulomb[index]
        ripple = extras.torque_ripple
        frame["ripple_amplitude"] = 0.0 if ripple is None else float(ripple.amplitude)
        frame["ripple_order"] = np.nan if ripple is None else float(ripple.order)
    return pd.DataFrame(frame)


#: One force/torque cell per (asset name, URDF path, target) inside a worker
#: process.  `_run_bag` runs in a process pool, so this is per-worker and never
#: shared.
#:
#: The URDF path is part of the key because an asset's *name* does not identify
#: its inertias: `payload_asset` and `scripts/sweep_ft_payload.py` both yield an
#: asset with the same name and a rewritten URDF, and keying on the name alone
#: silently hands the second caller the first one's cell.  That cost the T-14
#: sweep its first run -- every payload mass was measured with the 0 kg cell,
#: which reads identically zero (`R5_08` D-13).
_SENSOR_CACHE: dict[tuple[str, str, str], ForceTorqueSensor] = {}


def _sensor_for(asset: AssetSpec, signals: SignalPolicy) -> ForceTorqueSensor | None:
    """The asset's force/torque cell, built once per process.

    Building it costs a Pinocchio model, and every bag of a build needs the same
    one, so it is cached on the asset name rather than rebuilt per bag.  Returns
    ``None`` unless the signal policy actually needs it, so nothing is built for
    a joint-torque target.
    """
    if not signals.needs_sensor:
        return None
    key = (asset.name, str(asset.urdf_path), signals.target)
    if key not in _SENSOR_CACHE:
        spec = SensorSpec.from_asset(asset)
        if spec is None:
            raise ValueError(
                f"dataset.signals.target='{signals.target}' needs a force/torque cell, but asset "
                f"{asset.name!r} declares no `force_torque_sensor: {{frame: ...}}` in its asset.yaml"
            )
        _SENSOR_CACHE[key] = ForceTorqueSensor(asset, spec)
    return _SENSOR_CACHE[key]


def _feedback_ratio(feedforward: np.ndarray, feedback: np.ndarray) -> float | None:
    """``mean|feedback| / mean|feedforward|``, or ``None`` when there is no feedforward."""
    reference = float(np.mean(np.abs(np.asarray(feedforward, dtype=float))))
    if reference <= 0.0:
        return None
    return float(np.mean(np.abs(np.asarray(feedback, dtype=float))) / reference)


def _run_bag(args: tuple) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Execute one bag and return ``(frame, record)``.

    A plain top-level function taking one tuple argument, so it can be sent
    to a process pool: every argument is a simple dataclass/array (config,
    the asset spec, the tier, ...), never a built simulator model, which is
    what makes it picklable across the process boundary.  Bags are fully
    independent -- their RNG streams are keyed on ``(seed, index)``, not on
    call order -- so worker order never affects the result.  The control-
    separation ratio/joint are computed by the caller (``generate()``)
    *before* any bag is dispatched, so an ``action: error`` violation is
    caught before the pool starts, not inside a worker (R4_10 Sec 2.3); this
    function only carries them through to the manifest record.
    """
    (asset, trajectory, tier, backend, friction, config, link_inertia, payload,
     split, bag, bag_index, traj_index, friction_index, control_gains, link_inertia_max_value,
     separation_ratio, separation_joint, draw, extras) = args
    natural_frequency, damping_ratio = control_gains
    try:
        result = run_condition(asset, trajectory, tier, backend, friction, config,
                               link_inertia=link_inertia, payload=payload,
                               natural_frequency=natural_frequency, damping_ratio=damping_ratio,
                               draw=draw, extras=None if tier.is_rigid else extras)
        instability = None
    except RolloutDiverged as error:
        result, instability = None, error.describe()
    if instability is None:
        instability = _state_out_of_range(asset, result)
    if instability is not None:
        # Never reaches rollout_frame or validate_path: a reset or diverged
        # state is not physics, and validate_path's bisection on it is what
        # hung the weekend build for 17 h (`R5_09` F-1).
        return None, {
            "bag": bag, "bag_index": bag_index, "trajectory": traj_index, "tier": tier.name,
            "split": split, "backend": backend, "controller_mode": config.controller.mode,
            **instability,
            "controller": describe_controller(config.controller, draw),
            "control_natural_frequency": float(natural_frequency),
            "control_damping_ratio": float(damping_ratio),
            "control_position_gain": float(draw.position_gain),
            "control_velocity_bandwidth": float(draw.velocity_bandwidth),
            "control_integral_time": float(draw.integral_time),
            "transmission": describe_transmission(
                None if tier.is_rigid else tier.transmission(len(asset.joint_names), link_inertia)
            ),
        }
    frame = rollout_frame(asset, trajectory, result, bag=bag, tier=tier, backend=backend,
                          friction=friction, resample_step=config.sample_time_step, split=split,
                          signals=config.signals, measurement=config.measurement,
                          sensor=_sensor_for(asset, config.signals),
                          # Keyed on the bag index, so a bag's noise is the same
                          # however many bags ran before it and whatever the
                          # worker order (R4_10 Sec 2.3's reproducibility rule).
                          measurement_seed=config.seed + 1_000_003 * bag_index)
    # Effort limits come from the bare asset's URDF <limit effort=...> tags,
    # unaffected by payload injection (a payload changes link inertia, not
    # the motor's rated torque) -- a payload heavy/offset enough, combined
    # with a high regime acceleration, is the most likely way a bag becomes
    # physically meaningless without anything else catching it (R3_10 Sec 3.3).
    effort_limits = np.asarray(
        [joint.effort if joint.effort else np.inf for joint in asset.resolve_active_joints()]
    )
    peak_torque_ratio = float(np.max(np.abs(np.asarray(result["tau_motor"])) / effort_limits[None, :]))
    # The reference trajectory is validated collision-free at build time
    # (optimize_excitation), but tracking error means the *achieved* path
    # can differ, especially on a soft elastic robot -- flagged, not
    # dropped, same policy as the backend comparison (R4_14 Sec 2.3).
    margin = float(asset.metadata.get("collision", {}).get("margin", 0.0))
    max_joint_step = float(asset.metadata.get("collision", {}).get("max_joint_step", 0.05))
    achieved_deviation = float(np.max(np.abs(np.asarray(result["q_link"]) - np.asarray(result["q_ref"]))))
    achieved_clearance = None
    if achieved_deviation >= max_joint_step:
        # Only when the achieved path leaves the reference by more than the
        # check's own resolution: closer than that, every achieved sample is
        # within one resolution step of a validated reference sample, which is
        # the guarantee validate_path gives anyway (`R5_10` T-3.6).  Checked on
        # the output grid, not the physics step: validate_path decimates to
        # max_joint_step itself, so the finer path only cost the decimation.
        time = np.asarray(result["time"], dtype=float)
        grid = np.arange(time[0], time[-1] + 0.5 * config.sample_time_step, config.sample_time_step)
        q_link = np.asarray(result["q_link"], dtype=float)
        path = np.column_stack([np.interp(grid, time, q_link[:, i]) for i in range(q_link.shape[1])])
        with payload_asset(asset, payload) as asset_p:
            achieved_clearance = float(PortableKinematics(asset_p).validate_path(
                path, margin=margin, max_joint_step=max_joint_step,
            ).minimum_distance)
    record = {
        "bag": bag, "bag_index": bag_index, "trajectory": traj_index, "tier": tier.name, "split": split,
        **describe_transmission(result["transmission"], result["time_step"]),
        "payload": None if payload is None or payload.is_empty else payload.as_dict(),
        "friction_index": friction_index, "backend": backend,
        "solver": result.get("solver"), "wall_time": float(result.get("wall_time", 0.0)),
        "samples": int(len(result["time"])),
        "trajectory_digest": trajectory.digest(),
        "trajectory_signal_digest": trajectory.signal_digest(),
        "condition_number": float(trajectory.metadata["condition_number"]),
        "exc_base_frequency": float(trajectory.metadata["base_frequency"]),
        "exc_max_acceleration": float(trajectory.metadata["max_acceleration"]),
        "exc_velocity_fraction": float(trajectory.metadata["velocity_fraction"]),
        "exc_probe_top_hz": float(trajectory.metadata.get("probe_top_hz", 0.0)),
        "control_natural_frequency": float(natural_frequency),
        "control_damping_ratio": float(damping_ratio),
        "controller": describe_controller(config.controller, draw),
        "control_position_gain": float(draw.position_gain),
        "control_velocity_bandwidth": float(draw.velocity_bandwidth),
        "control_integral_time": float(draw.integral_time),
        "plant_extras": None if extras is None or extras.is_empty or tier.is_rigid else extras.describe(),
        "viscous": friction.viscous.tolist(), "coulomb": friction.coulomb.tolist(),
        "tracking_rms": float(np.sqrt(np.mean((np.asarray(result["q_link"]) - np.asarray(result["q_ref"])) ** 2))),
        "max_deflection": float(np.abs(np.asarray(result["q_motor"]) - np.asarray(result["q_link"])).max()),
        # Per-bag ringing measure (R5_Q Q-6): the velocity loop's range now
        # straddles the transmission mode on purpose, so some bags ring.  The
        # RMS -- not the peak -- is what compares across bags, and `generate()`
        # takes the ratio to the median over bags to flag the outliers.
        "deflection_rms": float(np.sqrt(np.mean(
            (np.asarray(result["q_motor"]) - np.asarray(result["q_link"])) ** 2
        ))),
        # None rather than a number divided by ~zero: a model-free PD
        # controller has no feedforward at all, and a ratio of 1e12 would read
        # as a measurement rather than as "not applicable" (R5_02 Sec 2).
        "feedback_ratio": _feedback_ratio(result["tau_feedforward"], result["tau_feedback"]),
        "peak_torque_ratio": peak_torque_ratio,
        "control_separation_min_ratio": separation_ratio,
        "control_separation_joint": separation_joint,
        # None when the check was skipped (achieved path within max_joint_step
        # of the validated reference); the deviation says why.
        "achieved_min_clearance_m": achieved_clearance,
        "achieved_max_deviation_rad": achieved_deviation,
    }
    return frame, record


#: A joint beyond this many spans of its range is diverged, not moving.
_DIVERGED_SPAN_FACTOR = 10.0


def _state_out_of_range(asset: AssetSpec, result: Mapping[str, Any]) -> dict[str, Any] | None:
    """The instability record of a rollout that finished but left physics.

    The runners raise on MuJoCo's own instability flags and on non-finite
    state; this catches what they cannot: a state still finite but more than
    ``_DIVERGED_SPAN_FACTOR`` joint spans out (`R5_10` T-2).  A joint without
    finite limits uses one turn as its span.
    """
    joints = asset.resolve_active_joints()
    span = np.asarray([
        (j.upper - j.lower) if j.lower is not None and j.upper is not None and np.isfinite(j.upper - j.lower)
        and j.upper > j.lower else 2.0 * np.pi
        for j in joints
    ])
    time = np.asarray(result["time"], dtype=float)
    for key in ("q_link", "q_motor"):
        values = np.asarray(result[key], dtype=float)
        bad = ~np.isfinite(values) | (np.abs(values) > _DIVERGED_SPAN_FACTOR * span[None, :])
        if bad.any():
            row, column = (int(v) for v in np.argwhere(bad)[0])
            return {
                "kind": "non_finite" if not np.isfinite(values[row, column]) else "joint_range",
                "time": float(time[row]), "mujoco_dof": None, "joint": asset.joint_names[column],
                "message": f"{key}[{asset.joint_names[column]}] = {values[row, column]:.3g} at "
                           f"t={time[row]:.6f}s, beyond {_DIVERGED_SPAN_FACTOR:g}x its span",
            }
    return None
