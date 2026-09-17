"""Torque-driven rollouts for dynamic-model identification datasets.

The position-tracking runners record the *controller command* they synthesize,
which is not a joint torque consistent with the dynamics and is computed
differently by each backend.  For identification the label has to be exact, so
these runners invert the arrangement: a torque is computed from the model and
injected as a pure generalized force, and the simulator only ever answers the
question "what motion does this torque produce?".

The applied torque is then exact by construction and *identical* across
backends, so any MuJoCo/Newton disagreement shows up purely as state
divergence rather than as an unexplained torque difference.

Passive effects are deliberately neutralized in both backends and re-injected
through the applied torque.  MuJoCo maps a URDF ``<dynamics damping=...>`` to
a passive viscous force while Newton maps the same attribute to a PD
controller gain, so leaving the importers to their own conventions would make
the two backends simulate different machines.  After neutralization both
integrate ``M(q) qdd + C(q,qd) qd + g(q) = tau_solver`` and nothing else.
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

from .assets import AssetSpec
from .identification import COULOMB_EPSILON, FrictionModel
from .materialized import MaterializedTrajectory

TorqueLaw = Callable[[float, np.ndarray, np.ndarray], "TorqueCommand"]


@dataclass(frozen=True)
class TorqueCommand:
    """One controller evaluation, split so its parts stay auditable."""

    total: np.ndarray
    feedforward: np.ndarray
    feedback: np.ndarray


def joint_inertia_floor(pin: Any, model: Any, data: Any, configurations: np.ndarray) -> np.ndarray:
    """Smallest diagonal joint-space inertia seen along a path.

    Distal wrist joints of a 7-DoF arm carry inertias three to four orders of
    magnitude below the shoulder (``izz = 3e-4`` on the iiwa's last link), so
    a single scalar PD gain that is gentle at the base is violently unstable
    at the wrist.  Gains are therefore scaled by this per-joint inertia.
    """
    configurations = np.atleast_2d(np.asarray(configurations, dtype=float))
    diagonals = []
    for q in configurations:
        mass_matrix = np.asarray(pin.crba(model, data, q), dtype=float)
        diagonals.append(np.diag(mass_matrix).copy())
    return np.min(np.asarray(diagonals), axis=0)


class ComputedTorqueController:
    """Computed-torque (inverse-dynamics) tracking controller.

    ``tau = rnea(q, dq, ddq_ref + kd*e_dot + kp*e) + friction(dq)``

    Substituting this into ``M qdd + C qd + g = tau - friction`` cancels the
    model exactly and leaves ``e_ddot + kd e_dot + kp e = 0``, so a single
    scalar gain pair fixes the closed-loop bandwidth of every joint.  That
    matters here: the iiwa's joint-space inertia matrix has a condition
    number around 1.5e4, so independent per-joint PD gains cannot decouple the
    axes and either drift or go unstable.

    The property that makes this usable as an identification label is that the
    commanded torque still satisfies ``tau = rnea(q, dq, qdd) + friction(dq)``
    on the *achieved* state, because the closed loop drives the achieved
    acceleration to the commanded one.  The label therefore stays exactly
    consistent with the dynamics while the trajectory stays bounded.
    """

    def __init__(
        self,
        asset: AssetSpec,
        trajectory: MaterializedTrajectory,
        *,
        friction: FrictionModel | None = None,
        natural_frequency: float = 25.0,
        damping_ratio: float = 1.0,
        effort_limit: np.ndarray | None = None,
        epsilon: float = COULOMB_EPSILON,
    ) -> None:
        from . import identification as idn

        if natural_frequency <= 0.0 or damping_ratio <= 0.0:
            raise ValueError("natural_frequency and damping_ratio must be positive")
        self.asset = asset
        self.trajectory = trajectory
        self.friction = FrictionModel.from_asset(asset) if friction is None else friction
        self.epsilon = float(epsilon)
        self.natural_frequency = float(natural_frequency)
        self.damping_ratio = float(damping_ratio)
        # Error-dynamics gains, in 1/s^2 and 1/s -- not torque gains.
        self.kp = self.natural_frequency**2
        self.kd = 2.0 * self.damping_ratio * self.natural_frequency
        self._pin, self._model, self._data = idn.build_model(asset)
        self._idn = idn
        if effort_limit is None:
            effort_limit = np.asarray([
                joint.effort if joint.effort is not None else np.inf
                for joint in asset.resolve_active_joints()
            ], dtype=float)
        self.effort_limit = np.asarray(effort_limit, dtype=float)

    def stability_margin(self, time_step: float) -> float:
        """Return ``kd * dt``; explicit integration of the error needs it < 2."""
        return float(self.kd * float(time_step))

    def __call__(self, t: float, q: np.ndarray, dq: np.ndarray) -> TorqueCommand:
        q_ref, dq_ref, ddq_ref = self.trajectory(float(t))
        q = np.asarray(q, dtype=float)
        dq = np.asarray(dq, dtype=float)
        feedforward = self._idn.inverse_dynamics(
            self._pin, self._model, self._data, q_ref, dq_ref, ddq_ref,
            friction=self.friction, epsilon=self.epsilon,
        )
        commanded = ddq_ref + self.kd * (dq_ref - dq) + self.kp * (q_ref - q)
        total = self._idn.inverse_dynamics(
            self._pin, self._model, self._data, q, dq, commanded,
            friction=self.friction, epsilon=self.epsilon,
        )
        total = np.clip(total, -self.effort_limit, self.effort_limit)
        return TorqueCommand(total=total, feedforward=feedforward, feedback=total - feedforward)


# Retained name for callers that want the plain feedforward-plus-PD form.
FeedforwardPDController = ComputedTorqueController


class _ViewerPacer:
    """Render at wall-clock speed from a rollout stepping far faster.

    Identification rollouts integrate at 0.5 ms or less; drawing every step
    would both crawl and swamp the viewer.  This renders only often enough to
    hit ``fps`` and sleeps to keep playback near ``realtime_scale``.
    """

    def __init__(self, viewer: Any, time_step: float, realtime_scale: float, fps: float = 60.0) -> None:
        self.viewer = viewer
        self.stride = max(1, int(round(1.0 / (fps * max(time_step, 1e-9) / max(realtime_scale, 1e-9)))))
        self.frame_period = self.stride * time_step / max(realtime_scale, 1e-9)
        self._next = None

    def running(self) -> bool:
        return self.viewer is None or self.viewer.is_running()

    def should_render(self, index: int) -> bool:
        return self.viewer is not None and index % self.stride == 0

    def pace(self) -> None:
        now = _time.perf_counter()
        if self._next is None:
            self._next = now
        self._next += self.frame_period
        remaining = self._next - _time.perf_counter()
        if remaining > 0.0:
            _time.sleep(remaining)
        else:
            self._next = _time.perf_counter()


class _NewtonStep:
    """Replay ``solver.step`` as a CUDA graph on GPU devices.

    A single-world rollout launches dozens of small kernels per step, so
    Python-side launch overhead dominates: uncaptured, one step costs ~3 ms,
    about 8x the captured cost, with bit-identical results.  The loop swaps
    two state objects, so one graph is kept per ``(state_in, state_out)``
    pair.  A capture records kernels without running them, hence the launch
    right after it.  The time step is baked into the graph; any other ``dt``
    and any step with contacts enabled run uncaptured.
    """

    def __init__(self, model: Any, solver: Any, *, enabled: bool = True) -> None:
        import warp as wp

        self._wp, self.solver = wp, solver
        self.enabled = enabled and wp.get_device(model.device).is_cuda
        self.device = model.device
        self._graphs: dict[tuple[int, int], Any] = {}
        self._dt: float | None = None

    def __call__(self, state_in: Any, state_out: Any, control: Any, contacts: Any, dt: float) -> None:
        if not self.enabled or (self._dt is not None and not np.isclose(dt, self._dt, rtol=1e-9, atol=0.0)):
            self.solver.step(state_in, state_out, control, contacts, dt)
            return
        key = (id(state_in), id(state_out))
        graph = self._graphs.get(key)
        if graph is None:
            self._dt = dt
            with self._wp.ScopedCapture(device=self.device) as capture:
                self.solver.step(state_in, state_out, control, contacts, dt)
            graph = self._graphs[key] = capture.graph
        self._wp.capture_launch(graph)


def _open_viewer(kind: str, asset: AssetSpec, trajectory: MaterializedTrajectory, *args: Any) -> Any:
    from . import visualization

    cls = visualization.MujocoVisualizer if kind == "mujoco" else visualization.NewtonVisualizer
    return cls(asset, trajectory).open(*args)


def _close_viewer(viewer: Any) -> None:
    if viewer is not None:
        try:
            viewer.close()
        except Exception:  # pragma: no cover - viewer teardown is best effort
            pass


def _result(
    time: np.ndarray,
    trajectory: MaterializedTrajectory,
    joint_names: tuple[str, ...],
    q: list, dq: list, tau: list, tau_ff: list, tau_fb: list,
    *, q_motor: list | None = None, dq_motor: list | None = None,
    tau_link: list | None = None, ddq: np.ndarray | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    time = np.asarray(time, dtype=float)
    q_array = np.asarray(q, dtype=float)
    dq_array = np.asarray(dq, dtype=float)
    if ddq is None:
        ddq = np.gradient(dq_array, time, axis=0) if len(time) > 2 else np.zeros_like(dq_array)
    planned = trajectory.sample()
    motor_q = q_array if q_motor is None else np.asarray(q_motor, dtype=float)
    motor_dq = dq_array if dq_motor is None else np.asarray(dq_motor, dtype=float)
    tau_array = np.asarray(tau, dtype=float)
    result = {
        "time": time,
        "q_ref": _resample(planned["q"], trajectory.time, time),
        "dq_ref": _resample(planned["dq"], trajectory.time, time),
        "q_link": q_array, "dq_link": dq_array, "ddq_link": np.asarray(ddq, dtype=float),
        "q_motor": motor_q, "dq_motor": motor_dq,
        "tau_motor": tau_array,
        # A rigid chain has no transmission compliance, so the link-side
        # generalized force is the applied joint torque itself.
        "tau_link": tau_array if tau_link is None else np.asarray(tau_link, dtype=float),
        "tau_feedforward": np.asarray(tau_ff, dtype=float),
        "tau_feedback": np.asarray(tau_fb, dtype=float),
        "joint_names": joint_names,
    }
    result.update(extra or {})
    return result


def _resample(values: np.ndarray, source_time: np.ndarray, target_time: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == len(target_time) and np.allclose(source_time[: len(target_time)], target_time):
        return values
    return np.column_stack([np.interp(target_time, source_time, values[:, i]) for i in range(values.shape[1])])


# ---------------------------------------------------------------------------
# MuJoCo
# ---------------------------------------------------------------------------

def run_mujoco_torque(
    asset: AssetSpec,
    trajectory: MaterializedTrajectory,
    controller: TorqueLaw,
    *,
    time_step: float = 0.002,
    friction: FrictionModel | None = None,
    config: Mapping[str, Any] | None = None,
    disable_contacts: bool = True,
    visualize: bool = False,
    realtime_scale: float = 1.0,
) -> dict[str, Any]:
    """Integrate ``asset`` under an injected joint torque in MuJoCo."""
    import mujoco

    from .generic_mujoco_runner import _build_model, _joint_addresses

    if tuple(trajectory.joint_names) != tuple(asset.joint_names):
        raise ValueError("Trajectory joint_names must match asset active_joints exactly")
    friction = FrictionModel.from_asset(asset) if friction is None else friction
    model, _ = _build_model(asset, mujoco, time_step, body_overrides=dict(config or {}).get("body_overrides", {}))
    neutralize_mujoco_passive(model)
    if disable_contacts:
        disable_mujoco_contacts(model)
    data = mujoco.MjData(model)
    active = _joint_addresses(model, mujoco, tuple(asset.joint_names))

    q0, dq0, _ = trajectory(0.0)
    for index, (qpos, dof) in enumerate(active):
        data.qpos[qpos] = q0[index]
        data.qvel[dof] = dq0[index]
    mujoco.mj_forward(model, data)

    grid = np.arange(0.0, trajectory.duration + 0.5 * time_step, time_step)
    q_rows, dq_rows, ddq_rows, tau_rows, ff_rows, fb_rows = [], [], [], [], [], []
    viewer = _open_viewer("mujoco", asset, trajectory, model, data) if visualize else None
    pacer = _ViewerPacer(viewer, time_step, realtime_scale)
    started = _time.perf_counter()
    for index, sample_time in enumerate(grid):
        if not pacer.running():
            break
        q = np.asarray([data.qpos[qpos] for qpos, _ in active], dtype=float)
        dq = np.asarray([data.qvel[dof] for _, dof in active], dtype=float)
        if not (np.isfinite(q).all() and np.isfinite(dq).all()):
            raise RuntimeError(f"MuJoCo torque rollout became non-finite at t={sample_time:.6f}s")
        command = controller(float(sample_time), q, dq)
        data.qfrc_applied[:] = 0.0
        # Friction is modelled explicitly rather than by the importer, so it
        # is subtracted here: the recorded label stays the full joint torque.
        solver_torque = command.total - friction.torque(dq)
        for joint_index, (_, dof) in enumerate(active):
            data.qfrc_applied[dof] = solver_torque[joint_index]
        mujoco.mj_forward(model, data)
        q_rows.append(q)
        dq_rows.append(dq)
        ddq_rows.append(np.asarray([data.qacc[dof] for _, dof in active], dtype=float))
        tau_rows.append(command.total)
        ff_rows.append(command.feedforward)
        fb_rows.append(command.feedback)
        if pacer.should_render(index):
            viewer.render(float(sample_time), q)
            pacer.pace()
        if index + 1 == len(grid):
            break
        mujoco.mj_step(model, data)
    _close_viewer(viewer)

    return _result(
        grid[: len(q_rows)], trajectory, tuple(asset.joint_names),
        q_rows, dq_rows, tau_rows, ff_rows, fb_rows,
        ddq=np.asarray(ddq_rows, dtype=float),
        extra={"backend": "mujoco", "mode": "rigid", "wall_time": _time.perf_counter() - started},
    )


def neutralize_mujoco_passive(model: Any) -> None:
    """Remove importer-supplied damping, friction loss and armature."""
    model.dof_damping[:] = 0.0
    model.dof_frictionloss[:] = 0.0
    model.dof_armature[:] = 0.0


def disable_mujoco_contacts(model: Any) -> None:
    """Turn off all contact generation for an identification rollout.

    Excitation trajectories are validated collision-free against the real
    geometry before they are ever simulated, so contacts can only add forces
    that are not part of the model being identified.

    Disabling them is also a correctness requirement for the elastic chain.
    Inserting a transmission body between two links means MuJoCo no longer
    sees them as a parent/child pair and stops excluding their contact
    automatically, so neighbouring links are detected deeply interpenetrated
    at their shared joint and the resulting forces silently lock the arm.
    """
    model.geom_contype[:] = 0
    model.geom_conaffinity[:] = 0


# ---------------------------------------------------------------------------
# Newton
# ---------------------------------------------------------------------------

def run_newton_torque(
    asset: AssetSpec,
    trajectory: MaterializedTrajectory,
    controller: TorqueLaw,
    *,
    time_step: float = 0.002,
    friction: FrictionModel | None = None,
    config: Mapping[str, Any] | None = None,
    disable_contacts: bool = True,
    visualize: bool = False,
    realtime_scale: float = 1.0,
    solver_order: tuple[str, ...] = ("SolverFeatherstone", "SolverMuJoCo", "SolverSemiImplicit"),
    capture_graph: bool = True,
) -> dict[str, Any]:
    """Integrate ``asset`` under an injected joint torque in Newton.

    ``SolverFeatherstone`` is preferred deliberately: it is an independent
    reduced-coordinate implementation, whereas ``SolverMuJoCo`` would run
    MuJoCo's own engine under Newton's API and could not serve as a
    cross-check.
    """
    from .generic_newton_runner import _as_numpy, _build_rigid_model, _new_solver

    if tuple(trajectory.joint_names) != tuple(asset.joint_names):
        raise ValueError("Trajectory joint_names must match asset active_joints exactly")
    friction = FrictionModel.from_asset(asset) if friction is None else friction
    built, newton = _build_rigid_model(asset, gravity=tuple((config or {}).get("gravity", asset.gravity)))
    model = built.model
    neutralize_newton_passive(model)
    state_in, state_out = model.state(), model.state()
    control = model.control()
    if not hasattr(control, "joint_f"):
        raise RuntimeError("This Newton build does not expose Control.joint_f for torque control")
    solver = _new_solver(model, newton, order=solver_order)
    step = _NewtonStep(model, solver, enabled=capture_graph and disable_contacts)
    contacts = model.contacts()
    direct = tuple(
        cls for cls in (getattr(newton.solvers, "SolverMuJoCo", None), getattr(newton.solvers, "SolverFeatherstone", None))
        if cls is not None
    )
    needs_ik = not isinstance(solver, direct) if direct else True

    names = tuple(asset.joint_names)
    q0, dq0, _ = trajectory(0.0)
    initial_q = _as_numpy(state_in.joint_q).reshape(-1)
    initial_dq = _as_numpy(state_in.joint_qd).reshape(-1)
    for index, name in enumerate(names):
        initial_q[built.q_index[name]] = q0[index]
        initial_dq[built.qd_index[name]] = dq0[index]
    state_in.joint_q.assign(initial_q.astype("float32"))
    state_in.joint_qd.assign(initial_dq.astype("float32"))
    newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)

    grid = np.arange(0.0, trajectory.duration + 0.5 * time_step, time_step)
    torque_buffer = np.zeros(model.joint_dof_count, dtype=np.float32)
    q_rows, dq_rows, tau_rows, ff_rows, fb_rows = [], [], [], [], []
    viewer = _open_viewer("newton", asset, trajectory, model) if visualize else None
    pacer = _ViewerPacer(viewer, time_step, realtime_scale)
    started = _time.perf_counter()
    for index, sample_time in enumerate(grid):
        if not pacer.running():
            break
        state_q = _as_numpy(state_in.joint_q).reshape(-1)
        state_dq = _as_numpy(state_in.joint_qd).reshape(-1)
        q = np.asarray([state_q[built.q_index[name]] for name in names], dtype=float)
        dq = np.asarray([state_dq[built.qd_index[name]] for name in names], dtype=float)
        if not (np.isfinite(q).all() and np.isfinite(dq).all()):
            raise RuntimeError(f"Newton torque rollout became non-finite at t={sample_time:.6f}s")
        command = controller(float(sample_time), q, dq)
        q_rows.append(q)
        dq_rows.append(dq)
        tau_rows.append(command.total)
        ff_rows.append(command.feedforward)
        fb_rows.append(command.feedback)
        if index + 1 == len(grid):
            break
        solver_torque = command.total - friction.torque(dq)
        torque_buffer.fill(0.0)
        for joint_index, name in enumerate(names):
            torque_buffer[built.qd_index[name]] = solver_torque[joint_index]
        control.joint_f.assign(torque_buffer)
        state_in.clear_forces()
        if not disable_contacts:
            model.collide(state_in, contacts)
        step(state_in, state_out, control, contacts, float(grid[index + 1] - sample_time))
        if needs_ik:
            newton.eval_ik(model, state_out, state_out.joint_q, state_out.joint_qd)
        if pacer.should_render(index):
            viewer.render(float(sample_time), state_out, q)
            pacer.pace()
        state_in, state_out = state_out, state_in
    _close_viewer(viewer)

    return _result(
        grid[: len(q_rows)], trajectory, names, q_rows, dq_rows, tau_rows, ff_rows, fb_rows,
        extra={"backend": "newton", "mode": "rigid", "solver": type(solver).__name__,
               "independent_of_mujoco": type(solver).__name__ != "SolverMuJoCo",
               "wall_time": _time.perf_counter() - started},
    )


# ---------------------------------------------------------------------------
# Elastic transmissions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TransmissionSpec:
    """Series-elastic transmission parameters for every active joint.

    ``rotor_inertia`` is the motor-side inertia reflected through the gear
    ratio, which for an industrial arm joint is of order 0.1 kg m^2.  It is
    declared explicitly rather than inferred from the fictional transmission
    body's mass: the builders would otherwise derive a rotor inertia around
    1e-6 kg m^2, five orders of magnitude too small, and the motor side would
    be violently stiffer than anything the integrator can follow.

    It is applied as joint *armature*, not as body inertia.  A geared rotor
    adds inertia about its own joint axis only; writing it into the fictional
    body's inertia tensor would instead load every downstream joint with it
    and change the arm being identified.

    The body's *mass* is kept negligible so that inserting a transmission does
    not add weight the real robot does not carry.
    """

    stiffness: np.ndarray
    damping: np.ndarray
    rotor_inertia: np.ndarray
    intermediate_mass: float = 1.0e-3

    def __post_init__(self) -> None:
        stiffness = np.asarray(self.stiffness, dtype=float).reshape(-1)
        damping = np.asarray(self.damping, dtype=float).reshape(-1)
        rotor = np.broadcast_to(np.asarray(self.rotor_inertia, dtype=float).reshape(-1), stiffness.shape).copy()
        if stiffness.shape != damping.shape:
            raise ValueError("stiffness and damping must have the same length")
        if np.any(stiffness <= 0.0) or np.any(damping < 0.0) or np.any(rotor <= 0.0):
            raise ValueError("stiffness and rotor_inertia must be positive, damping non-negative")
        if self.intermediate_mass <= 0.0:
            raise ValueError("intermediate_mass must be positive")
        object.__setattr__(self, "stiffness", stiffness)
        object.__setattr__(self, "damping", damping)
        object.__setattr__(self, "rotor_inertia", rotor)

    @classmethod
    def uniform(
        cls, n_dof: int, stiffness: float, *, damping_ratio: float = 0.05,
        rotor_inertia: float = 0.1, **kwargs,
    ) -> "TransmissionSpec":
        """Build a uniform transmission from a stiffness and a damping ratio."""
        stiffness_array = np.full(n_dof, float(stiffness))
        rotor = np.full(n_dof, float(rotor_inertia))
        damping = 2.0 * float(damping_ratio) * np.sqrt(stiffness_array * rotor)
        return cls(stiffness_array, damping, rotor, **kwargs)

    def natural_frequency(self) -> np.ndarray:
        """Transmission resonance in Hz, used to bound the integration step."""
        return np.sqrt(self.stiffness / self.rotor_inertia) / (2.0 * np.pi)

    def required_time_step(self, samples_per_period: float = 20.0) -> float:
        return float(1.0 / (samples_per_period * float(np.max(self.natural_frequency()))))

    def require_stable_step(self, time_step: float, *, samples_per_period: float = 20.0) -> None:
        """Fail loudly when ``time_step`` cannot resolve the transmission mode."""
        required = self.required_time_step(samples_per_period)
        if float(time_step) > required:
            peak = float(np.max(self.natural_frequency()))
            raise ValueError(
                f"time_step={time_step:g}s cannot resolve a {peak:.1f} Hz transmission mode; "
                f"use time_step <= {required:.2e}s or reduce stiffness"
            )

    def as_mapping(self, joint_names: tuple[str, ...], *, motor_stiffness: float = 0.0,
                   motor_damping: float = 0.0) -> dict[str, dict[str, Any]]:
        return {
            name: {
                "stiffness": float(self.stiffness[index]),
                "damping": float(self.damping[index]),
                "motor_stiffness": float(motor_stiffness),
                "motor_damping": float(motor_damping),
                "intermediate_mass": float(self.intermediate_mass),
                "link_mass": None,
                # Kept negligible on purpose: the rotor inertia is applied as
                # joint armature so it loads only its own motor axis.
                "intermediate_inertia_x": None,
                "intermediate_inertia_y": None,
                "intermediate_inertia_z": None,
            }
            for index, name in enumerate(joint_names)
        }


class SeaMotorController:
    """Motor-side computed torque for a series-elastic joint.

    ``tau_m = rnea(q_m, dq_m, a_cmd) + J_rotor a_cmd + friction(dq_m)`` with
    ``a_cmd = ddq_ref + kd e_dot + kp e`` on the motor coordinate.

    The rigid ``rnea`` term is essential and easy to get wrong: the motor
    joint of a serial chain carries the inertia of every downstream body, not
    just its own rotor.  Sizing the controller by the rotor inertia alone
    under-estimates the true motor-side inertia by a factor of roughly fifty
    on this arm, which leaves the loop far too soft and drives the motor into
    a large-amplitude oscillation against its torque limit.

    Control is deliberately collocated (motor position feeds back on motor
    torque).  Closing the loop on the link instead is non-collocated and goes
    unstable as the transmission softens, which is exactly the regime the
    soft tiers of the dataset are meant to explore.
    """

    def __init__(
        self,
        asset: AssetSpec,
        trajectory: MaterializedTrajectory,
        transmission: TransmissionSpec,
        *,
        natural_frequency: float = 25.0,
        damping_ratio: float = 1.0,
        friction: FrictionModel | None = None,
        effort_limit: np.ndarray | None = None,
        epsilon: float = COULOMB_EPSILON,
    ) -> None:
        from . import identification as idn

        if natural_frequency <= 0.0 or damping_ratio <= 0.0:
            raise ValueError("natural_frequency and damping_ratio must be positive")
        self.asset = asset
        self.trajectory = trajectory
        self.transmission = transmission
        self.friction = FrictionModel.from_asset(asset) if friction is None else friction
        self.epsilon = float(epsilon)
        self.natural_frequency = float(natural_frequency)
        self.damping_ratio = float(damping_ratio)
        self.kp = self.natural_frequency**2
        self.kd = 2.0 * self.damping_ratio * self.natural_frequency
        self._pin, self._model, self._data = idn.build_model(asset)
        self._idn = idn
        if effort_limit is None:
            effort_limit = np.asarray([
                joint.effort if joint.effort is not None else np.inf
                for joint in asset.resolve_active_joints()
            ], dtype=float)
        self.effort_limit = np.asarray(effort_limit, dtype=float)

    def stability_margin(self, time_step: float) -> float:
        return float(self.kd * float(time_step))

    def __call__(
        self, t: float, q_motor: np.ndarray, dq_motor: np.ndarray, tau_spring: np.ndarray
    ) -> TorqueCommand:
        q_ref, dq_ref, ddq_ref = self.trajectory(float(t))
        q_motor = np.asarray(q_motor, dtype=float)
        dq_motor = np.asarray(dq_motor, dtype=float)
        # What the motor would have to supply with no tracking error at all.
        feedforward = self._idn.inverse_dynamics(
            self._pin, self._model, self._data, q_ref, dq_ref, ddq_ref,
            friction=self.friction, epsilon=self.epsilon,
        )
        commanded = ddq_ref + self.kd * (dq_ref - dq_motor) + self.kp * (q_ref - q_motor)
        total = self._idn.inverse_dynamics(
            self._pin, self._model, self._data, q_motor, dq_motor, commanded,
            friction=self.friction, epsilon=self.epsilon,
        ) + self.transmission.rotor_inertia * commanded
        total = np.clip(total, -self.effort_limit, self.effort_limit)
        return TorqueCommand(total=total, feedforward=feedforward, feedback=total - feedforward)


def run_mujoco_elastic_torque(
    asset: AssetSpec,
    trajectory: MaterializedTrajectory,
    controller: TorqueLaw,
    transmission: TransmissionSpec,
    *,
    time_step: float = 0.0002,
    friction: FrictionModel | None = None,
    config: Mapping[str, Any] | None = None,
    check_step: bool = True,
    disable_contacts: bool = True,
    visualize: bool = False,
    realtime_scale: float = 1.0,
) -> dict[str, Any]:
    """Torque-driven rollout of a series-elastic chain in MuJoCo.

    The motor torque is injected on the motor joint; the transmission spring
    is MuJoCo's own zero-reference joint stiffness on the elastic joint.
    Feedback is taken from the *motor* side because motor-side control of a
    flexible joint is collocated and therefore stable for soft transmissions.
    """
    import mujoco

    from .generic_mujoco_runner import _build_model, _elastic_addresses

    if tuple(trajectory.joint_names) != tuple(asset.joint_names):
        raise ValueError("Trajectory joint_names must match asset active_joints exactly")
    if check_step:
        transmission.require_stable_step(time_step)
    names = tuple(asset.joint_names)
    friction = FrictionModel.from_asset(asset) if friction is None else friction
    parameters = transmission.as_mapping(names)
    model, _ = _build_model(asset, mujoco, time_step, elastic_transmissions=parameters,
                            body_overrides=dict(config or {}).get("body_overrides", {}))
    neutralize_mujoco_passive(model)
    if disable_contacts:
        disable_mujoco_contacts(model)
    addresses = _elastic_addresses(model, mujoco, names)
    for index, name in enumerate(names):
        model.jnt_stiffness[addresses[name]["elastic_joint_id"]] = transmission.stiffness[index]
        model.dof_damping[addresses[name]["elastic_dof"]] = transmission.damping[index]
        model.dof_armature[addresses[name]["motor_dof"]] = transmission.rotor_inertia[index]
    data = mujoco.MjData(model)
    mujoco.mj_setConst(model, data)

    q0, dq0, _ = trajectory(0.0)
    for index, name in enumerate(names):
        data.qpos[addresses[name]["motor_qpos"]] = q0[index]
        data.qvel[addresses[name]["motor_dof"]] = dq0[index]
    mujoco.mj_forward(model, data)

    grid = np.arange(0.0, trajectory.duration + 0.5 * time_step, time_step)
    rows: dict[str, list] = {key: [] for key in
                             ("q", "dq", "qm", "dqm", "tau", "ff", "fb", "tau_link", "ddq")}
    viewer = _open_viewer("mujoco", asset, trajectory, model, data) if visualize else None
    pacer = _ViewerPacer(viewer, time_step, realtime_scale)
    started = _time.perf_counter()
    for index, sample_time in enumerate(grid):
        if not pacer.running():
            break
        motor_q = np.asarray([data.qpos[addresses[n]["motor_qpos"]] for n in names])
        motor_dq = np.asarray([data.qvel[addresses[n]["motor_dof"]] for n in names])
        elastic_q = np.asarray([data.qpos[addresses[n]["elastic_qpos"]] for n in names])
        elastic_dq = np.asarray([data.qvel[addresses[n]["elastic_dof"]] for n in names])
        link_q, link_dq = motor_q + elastic_q, motor_dq + elastic_dq
        if not (np.isfinite(link_q).all() and np.isfinite(link_dq).all()):
            raise RuntimeError(f"MuJoCo elastic rollout became non-finite at t={sample_time:.6f}s")
        tau_spring = -(transmission.stiffness * elastic_q + transmission.damping * elastic_dq)
        command = controller(float(sample_time), motor_q, motor_dq, tau_spring)
        data.qfrc_applied[:] = 0.0
        solver_torque = command.total - friction.torque(motor_dq)
        for joint_index, name in enumerate(names):
            data.qfrc_applied[addresses[name]["motor_dof"]] = solver_torque[joint_index]
        mujoco.mj_forward(model, data)
        rows["q"].append(link_q)
        rows["dq"].append(link_dq)
        rows["qm"].append(motor_q)
        rows["dqm"].append(motor_dq)
        rows["tau"].append(command.total)
        rows["ff"].append(command.feedforward)
        rows["fb"].append(command.feedback)
        rows["tau_link"].append(tau_spring)
        rows["ddq"].append(np.asarray([data.qacc[addresses[n]["motor_dof"]] + data.qacc[addresses[n]["elastic_dof"]]
                                       for n in names]))
        if pacer.should_render(index):
            viewer.render(float(sample_time), link_q)
            pacer.pace()
        if index + 1 == len(grid):
            break
        mujoco.mj_step(model, data)
    _close_viewer(viewer)

    return _result(
        grid[: len(rows["q"])], trajectory, names,
        rows["q"], rows["dq"], rows["tau"], rows["ff"], rows["fb"],
        q_motor=rows["qm"], dq_motor=rows["dqm"], tau_link=rows["tau_link"],
        ddq=np.asarray(rows["ddq"], dtype=float),
        extra={"backend": "mujoco", "mode": "elastic",
               "transmission_frequency_hz": float(np.max(transmission.natural_frequency())),
               "wall_time": _time.perf_counter() - started},
    )


def run_newton_elastic_torque(
    asset: AssetSpec,
    trajectory: MaterializedTrajectory,
    controller: TorqueLaw,
    transmission: TransmissionSpec,
    *,
    time_step: float = 0.0002,
    friction: FrictionModel | None = None,
    config: Mapping[str, Any] | None = None,
    check_step: bool = True,
    disable_contacts: bool = True,
    visualize: bool = False,
    realtime_scale: float = 1.0,
    solver_order: tuple[str, ...] = ("SolverMuJoCo", "SolverFeatherstone", "SolverSemiImplicit"),
    capture_graph: bool = True,
) -> dict[str, Any]:
    """Torque-driven rollout of a series-elastic chain in Newton.

    Unlike the rigid path, this one defaults to ``SolverMuJoCo``.  Newton's
    reduced-coordinate and semi-implicit solvers both diverge within a few
    milliseconds on the doubled-degree-of-freedom elastic chain, at every
    transmission-body size and time step tried, so they are not usable here.

    ``SolverMuJoCo`` runs MuJoCo's engine under Newton's API, so an elastic
    Newton rollout is **not** an independent check on the MuJoCo one.  The
    result reports this as ``independent_of_mujoco``; only the rigid path
    gives a genuinely independent second opinion.
    """
    from .generic_newton_runner import (
        ElasticTransmissionParams, _as_numpy, _new_solver, build_elastic_model,
    )

    if tuple(trajectory.joint_names) != tuple(asset.joint_names):
        raise ValueError("Trajectory joint_names must match asset active_joints exactly")
    if check_step:
        transmission.require_stable_step(time_step)
    names = tuple(asset.joint_names)
    friction = FrictionModel.from_asset(asset) if friction is None else friction
    transmissions = {
        name: ElasticTransmissionParams(
            stiffness=float(transmission.stiffness[index]),
            damping=float(transmission.damping[index]),
            # The motor joint is driven purely by the injected torque.
            motor_stiffness=0.0, motor_damping=0.0,
            intermediate_mass=transmission.intermediate_mass,
        )
        for index, name in enumerate(names)
    }
    built = build_elastic_model(
        asset, transmissions,
        gravity=tuple((config or {}).get("gravity", asset.gravity)),
        intermediate_mass=transmission.intermediate_mass,
        # The transmission body is a kinematic placeholder only; its rotor
        # inertia is applied as joint armature, so keep its extent negligible.
        intermediate_size=float((config or {}).get("intermediate_size", 1.0e-3)),
        body_overrides=dict(config or {}).get("body_overrides", {}),
    )
    from .generic_newton_runner import _require_newton

    _wp, newton = _require_newton()
    model = built.model
    motor_dofs = [built.dof_index[name].motor_qd for name in names]
    _neutralize_newton_dofs(model, motor_dofs)
    _set_newton_armature(model, motor_dofs, transmission.rotor_inertia)
    state_in, state_out = model.state(), model.state()
    control = model.control()
    if not hasattr(control, "joint_f"):
        raise RuntimeError("This Newton build does not expose Control.joint_f for torque control")
    solver = _new_solver(model, newton, order=solver_order)
    step = _NewtonStep(model, solver, enabled=capture_graph and disable_contacts)
    contacts = model.contacts()
    direct = tuple(cls for cls in (getattr(newton.solvers, "SolverMuJoCo", None),
                                   getattr(newton.solvers, "SolverFeatherstone", None)) if cls is not None)
    needs_ik = not isinstance(solver, direct) if direct else True

    q0, dq0, _ = trajectory(0.0)
    initial_q = _as_numpy(state_in.joint_q).reshape(-1)
    initial_dq = _as_numpy(state_in.joint_qd).reshape(-1)
    for index, name in enumerate(names):
        mapping = built.dof_index[name]
        initial_q[mapping.motor_q] = q0[index]
        initial_q[mapping.elastic_q] = 0.0
        initial_dq[mapping.motor_qd] = dq0[index]
        initial_dq[mapping.elastic_qd] = 0.0
    state_in.joint_q.assign(initial_q.astype("float32"))
    state_in.joint_qd.assign(initial_dq.astype("float32"))
    newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)

    grid = np.arange(0.0, trajectory.duration + 0.5 * time_step, time_step)
    buffer = np.zeros(model.joint_dof_count, dtype=np.float32)
    rows: dict[str, list] = {key: [] for key in ("q", "dq", "qm", "dqm", "tau", "ff", "fb", "tau_link")}
    viewer = _open_viewer("newton", asset, trajectory, model) if visualize else None
    pacer = _ViewerPacer(viewer, time_step, realtime_scale)
    started = _time.perf_counter()
    for index, sample_time in enumerate(grid):
        if not pacer.running():
            break
        state_q = _as_numpy(state_in.joint_q).reshape(-1)
        state_dq = _as_numpy(state_in.joint_qd).reshape(-1)
        motor_q = np.asarray([state_q[built.dof_index[n].motor_q] for n in names])
        motor_dq = np.asarray([state_dq[built.dof_index[n].motor_qd] for n in names])
        elastic_q = np.asarray([state_q[built.dof_index[n].elastic_q] for n in names])
        elastic_dq = np.asarray([state_dq[built.dof_index[n].elastic_qd] for n in names])
        link_q, link_dq = motor_q + elastic_q, motor_dq + elastic_dq
        if not (np.isfinite(link_q).all() and np.isfinite(link_dq).all()):
            raise RuntimeError(f"Newton elastic rollout became non-finite at t={sample_time:.6f}s")
        tau_spring = -(transmission.stiffness * elastic_q + transmission.damping * elastic_dq)
        command = controller(float(sample_time), motor_q, motor_dq, tau_spring)
        rows["q"].append(link_q)
        rows["dq"].append(link_dq)
        rows["qm"].append(motor_q)
        rows["dqm"].append(motor_dq)
        rows["tau"].append(command.total)
        rows["ff"].append(command.feedforward)
        rows["fb"].append(command.feedback)
        rows["tau_link"].append(tau_spring)
        if index + 1 == len(grid):
            break
        solver_torque = command.total - friction.torque(motor_dq)
        buffer.fill(0.0)
        for joint_index, name in enumerate(names):
            buffer[built.dof_index[name].motor_qd] = solver_torque[joint_index]
        control.joint_f.assign(buffer)
        state_in.clear_forces()
        if not disable_contacts:
            model.collide(state_in, contacts)
        step(state_in, state_out, control, contacts, float(grid[index + 1] - sample_time))
        if needs_ik:
            newton.eval_ik(model, state_out, state_out.joint_q, state_out.joint_qd)
        if pacer.should_render(index):
            viewer.render(float(sample_time), state_out, link_q)
            pacer.pace()
        state_in, state_out = state_out, state_in
    _close_viewer(viewer)

    return _result(
        grid[: len(rows["q"])], trajectory, names,
        rows["q"], rows["dq"], rows["tau"], rows["ff"], rows["fb"],
        q_motor=rows["qm"], dq_motor=rows["dqm"], tau_link=rows["tau_link"],
        extra={"backend": "newton", "mode": "elastic", "solver": type(solver).__name__,
               "independent_of_mujoco": type(solver).__name__ != "SolverMuJoCo",
               "transmission_frequency_hz": float(np.max(transmission.natural_frequency())),
               "wall_time": _time.perf_counter() - started},
    )


def _set_newton_armature(model: Any, dofs: list[int], inertia: np.ndarray) -> None:
    """Add the reflected rotor inertia to the motor degrees of freedom."""
    array = getattr(model, "joint_armature", None)
    if array is None:
        raise RuntimeError("This Newton build does not expose Model.joint_armature")
    values = (array.numpy() if hasattr(array, "numpy") else np.asarray(array)).copy()
    for index, dof in enumerate(dofs):
        values[dof] = float(inertia[index])
    array.assign(values)


def _neutralize_newton_dofs(model: Any, dofs: list[int]) -> None:
    """Zero friction/armature everywhere and the PD gains on ``dofs`` only.

    The elastic joints keep their ``target_ke``/``target_kd``: that is how the
    Newton builder represents the transmission spring.  Only the motor joints
    must be freed, so the injected torque is the sole motor-side input.
    """
    import numpy as _np

    for field in ("joint_friction", "joint_armature"):
        array = getattr(model, field, None)
        if array is None:
            continue
        values = array.numpy() if hasattr(array, "numpy") else _np.asarray(array)
        array.assign(_np.zeros_like(values))
    for field in ("joint_target_ke", "joint_target_kd"):
        array = getattr(model, field, None)
        if array is None:
            continue
        values = (array.numpy() if hasattr(array, "numpy") else _np.asarray(array)).copy()
        for dof in dofs:
            values[dof] = 0.0
        array.assign(values)


def neutralize_newton_passive(model: Any) -> None:
    """Zero Newton's joint friction, armature and built-in PD gains.

    Newton's URDF importer puts ``<dynamics damping>`` into ``joint_target_kd``
    — a controller gain, not a passive force — so leaving it in place would
    silently add a PD loop on top of the injected torque.
    """
    import numpy as _np

    for field in ("joint_friction", "joint_armature", "joint_target_ke", "joint_target_kd"):
        array = getattr(model, field, None)
        if array is None:
            continue
        values = array.numpy() if hasattr(array, "numpy") else _np.asarray(array)
        array.assign(_np.zeros_like(values))
