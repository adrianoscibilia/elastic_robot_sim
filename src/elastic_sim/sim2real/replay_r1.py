"""R1: motor-forced elastic replay (S2R_01 §4.2, T-3.3).

The measured motor trajectory is imposed on the elastic chain (MuJoCo,
:func:`~elastic_sim.torque_runners.run_mujoco_elastic_torque`, the production
runner) and the link side is integrated; the spring torque is observed
through the iiwa JTS model, ``ft_sim = g tau_s + o[run]``.

* **Reference.** A smoothing cubic spline of the measured motor ``q``
  (``scipy.interpolate.make_smoothing_spline``), evaluated once on the
  physics grid; the trajectory's ``analytic`` evaluator is a table lookup
  there. Consistent q, q', q'' and no interpolation aliasing. The smoothing
  is set per joint by the **discrepancy principle**: the fit residual's RMS
  equals the joint's measured hold sigma(q) (FR_05 §6, floor: quantum /
  sqrt(12)). GCV is not used: on these bags it reads the real 10-50 Hz motor
  motion (~300 quanta on A2, against ~1 quantum of hold noise) as noise and
  smooths it away, so R1 would not impose the measured motion.
* **Controller** (:class:`MotorForcing`): the exact motor-side law. In the
  runner's ``(theta, e)`` coordinates the motor row is
  ``J_r theta'' = tau_m - f_m(theta') - tau_s`` (link friction loads both
  rows and cancels; the intermediate body sits on the axis), so
  ``tau_m = J_r a + tau_s + f_m(theta')``, ``a = q_ref'' + kd e' + kp e``,
  with ``tau_s`` the spring torque the runner hands every controller call,
  at ``replay.r1.track_bandwidth`` (default 300 rad/s), zeta = 1. It only
  forces the measured motor motion: it is not a model of KUKA's
  controller. S2R_01 §4.2 names ``SeaMotorController`` on the plant theta;
  its rigid ``rnea`` feedforward leaves the transmission's own dynamics as a
  1-3 N m disturbance, i.e. 95-225 encoder quanta of tracking error at 300
  rad/s on B's bags (T-3.3 asks <= 10), so R1 uses the exact law.
* **Plant.** theta's inertials in memory (``model_setup`` hook), K and zeta
  from theta (frozen, A-1), the reflected rotor inertia as armature, motor
  friction, link friction (S2R_04 A-8: epsilon 1e-3). The round-6 nonlinear
  spring and transmission error are off by default (``extras:
  link_friction``): their per-bag phase is unknown and S2R_06 F-9 decides
  first whether the real residual carries them (``extras: plant`` turns them
  on at their round-6 values).
* **Initial state.** The motor at the reference's ``(q, dq)`` at the start
  (at rest to the sensor noise), the link moving with it, the deflection
  from the static equilibrium (:mod:`.equilibrium`, ``InitialState`` hook,
  T-3.1); ``K delta`` against the measured ``(ft(0) - o) / g`` is recorded.
* **Windows** (``replay.r1.windows``): each window starts quasi-statically,
  ``delta_0 = (ft(t0) - o) / (g K)``, link velocity = motor velocity; the
  first ``burn_in_s`` of every window is not scored. Windows tile a bag on
  their post-burn-in parts.
* **Rows.** The 1 kHz rows of R0's grid (``replay.r0.decimate`` /
  ``hold_decimate``), so R0 and R1 results compare row for row (G-3).
* **Physics step.** As production: the largest step that divides 1 ms and
  resolves the transmission mode (``elastic_time_step``'s rule).

Every bag (window) is one task; ``jobs`` runs them in a process pool.
"""

from __future__ import annotations

import dataclasses
import os
import time as _time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from ..materialized import MaterializedTrajectory
from .config import R0Settings, R1Settings
from .plant import BodyMap, PlantBase, PlantParameters
from .real_runs import ReplayBag
from .result import ReplayResult

MODE = "R1"
#: The iiwa's motor encoder step (FR_05; the smallest q step in A and B), rad.
ENCODER_QUANTUM = 5.989e-8
#: T-3.3 done-when: motor tracking RMS <= this many quanta per joint.
TRACKING_QUANTA = 10.0
CONTROL_PERIOD = 1.0e-3


@dataclass(frozen=True)
class R1Bag:
    """A bag's theta-independent R1 inputs: the 1 kHz record and the rows scored."""

    run_id: str
    bag: str
    kind: str
    split: str
    t: np.ndarray      # (N,) 1 kHz
    q: np.ndarray      # (N, n) measured motor
    ft: np.ndarray     # (N, n)
    rows: np.ndarray   # kept rows (R0's grid)
    sigma_q: np.ndarray | None = None   # (n,) measured q noise: the reference's smoothing

    @property
    def n_rows(self) -> int:
        return len(self.rows)


def prepare_bags(bags: Iterable[ReplayBag], r0: R0Settings, *, split_of: Callable[[str, str], str | None] | None
                 = None, sigma_q: Mapping[str, np.ndarray] | None = None) -> list[R1Bag]:
    """``sigma_q``: per run, the hold sigma(q) (``RealRun.sidecar``); the
    reference's smoothing level (default: the quantisation noise)."""
    out = []
    for bag in bags:
        split = (split_of(bag.run_id, bag.bag) if split_of is not None else None) or ""
        step = int(r0.hold_decimate if bag.kind == "hold" else r0.decimate)
        out.append(R1Bag(bag.run_id, bag.bag, bag.kind, split, np.asarray(bag.t, dtype=float),
                         np.asarray(bag.q, dtype=float), np.asarray(bag.ft, dtype=float),
                         np.arange(0, bag.n_rows, step),
                         None if sigma_q is None or bag.run_id not in sigma_q
                         else np.asarray(sigma_q[bag.run_id], dtype=float)))
    return out


def run_sigma_q(run: Any) -> np.ndarray | None:
    """The run's hold sigma(q) per joint (sidecar), or ``None``."""
    holds = ((run.sidecar or {}).get("summary") or {}).get("holds") or {}
    return None if "sigma_q" not in holds else np.asarray(holds["sigma_q"], dtype=float)


# ---------------------------------------------------------------------------
# Reference
# ---------------------------------------------------------------------------


def discrepancy_spline(t: np.ndarray, y: np.ndarray, sigma: float, *, iterations: int = 40):
    """The smoothing spline whose fit residual RMS is ``sigma`` (bisection on
    log lambda; the residual grows with lambda). Returns ``(spline, lam)``."""
    from scipy.interpolate import make_smoothing_spline

    def rms(lam):
        spline = make_smoothing_spline(t, y, lam=lam)
        return spline, float(np.sqrt(np.mean((spline(t) - y) ** 2)))

    low, high = -30.0, 5.0   # log10 lambda (t in s)
    spline, value = rms(10.0 ** low)
    if value >= sigma:
        return spline, 10.0 ** low
    best = (spline, 10.0 ** low)
    for _ in range(iterations):
        middle = 0.5 * (low + high)
        spline, value = rms(10.0 ** middle)
        if value > sigma:
            high = middle
        else:
            low, best = middle, (spline, 10.0 ** middle)
        if high - low < 0.02:
            break
    return best


def spline_reference(t: np.ndarray, q: np.ndarray, joint_names: Sequence[str], *, time_step: float,
                     sigma: Sequence[float] | None = None) -> tuple[MaterializedTrajectory, dict[str, Any]]:
    """The smoothing-spline reference on ``[t0, t_end]``, time from 0, its
    ``(q, dq, ddq)`` tabulated on the physics grid ``k * time_step``; joint
    j's fit residual RMS is ``sigma[j]`` (default: quantum / sqrt(12))."""
    t = np.asarray(t, dtype=float) - float(t[0])
    q = np.asarray(q, dtype=float)
    floor = ENCODER_QUANTUM / np.sqrt(12.0)
    sigma = np.full(q.shape[1], floor) if sigma is None else np.maximum(np.asarray(sigma, dtype=float), floor)
    splines, lams = [], []
    for j in range(q.shape[1]):
        spline, lam = discrepancy_spline(t, q[:, j], float(sigma[j]))
        splines.append(spline)
        lams.append(lam)
    grid = np.arange(0.0, t[-1] + 0.5 * time_step, time_step)
    table = [np.column_stack([s(grid, nu) for s in splines]) for nu in (0, 1, 2)]
    fit = np.column_stack([s(t) for s in splines])

    def analytic(time_s: float, _grid=grid, _table=table, _step=time_step):
        k = min(int(round(time_s / _step)), len(_grid) - 1)
        return _table[0][k].copy(), _table[1][k].copy(), _table[2][k].copy()

    trajectory = MaterializedTrajectory(time=t, position=fit, velocity=np.column_stack([s(t, 1) for s in splines]),
                                        acceleration=np.column_stack([s(t, 2) for s in splines]),
                                        joint_names=tuple(joint_names), metadata={"source": "R1 spline reference"},
                                        analytic=analytic)
    return trajectory, {"lam": lams, "sigma": sigma.tolist(),
                        "fit_rms": np.sqrt(np.mean((fit - q) ** 2, axis=0)).tolist()}


# ---------------------------------------------------------------------------
# One rollout
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class R1Task:
    """One rollout: a bag, or a window of it ([start, stop) rows, scored from ``score_from``)."""

    bag: R1Bag
    start: int
    stop: int
    score_from: int
    score_to: int
    quasi_static: bool   # a window after the first: delta_0 from the measured ft


class MotorForcing:
    """``tau_m = J_r (q_ref'' + kd e' + kp e) + tau_s + f_m(theta')``: imposes
    the reference on the motor coordinate of the elastic chain exactly (the
    motor row is ``J_r theta'' = tau_m - f_m - tau_s``); the PD only rejects
    the integration and initial errors. ``compensates_friction``: the runner
    subtracts ``f_m`` at the same velocity, as for ``SeaMotorController``."""

    compensates_friction = True

    def __init__(self, trajectory: MaterializedTrajectory, rotor_inertia: np.ndarray, friction: Any, *,
                 natural_frequency: float = 300.0, damping_ratio: float = 1.0,
                 effort_limit: np.ndarray | None = None):
        if natural_frequency <= 0.0 or damping_ratio <= 0.0:
            raise ValueError("natural_frequency and damping_ratio must be positive")
        self.trajectory = trajectory
        self.rotor_inertia = np.asarray(rotor_inertia, dtype=float)
        self.friction = friction
        self.natural_frequency = float(natural_frequency)
        self.kp = self.natural_frequency ** 2
        self.kd = 2.0 * float(damping_ratio) * self.natural_frequency
        self.effort_limit = np.full_like(self.rotor_inertia, np.inf) if effort_limit is None \
            else np.asarray(effort_limit, dtype=float)

    def stability_margin(self, time_step: float) -> float:
        return float(self.kd * float(time_step))

    def __call__(self, t: float, q_motor: np.ndarray, dq_motor: np.ndarray, tau_spring: np.ndarray):
        from ..torque_runners import TorqueCommand

        q_ref, dq_ref, ddq_ref = self.trajectory(float(t))
        dq_motor = np.asarray(dq_motor, dtype=float)
        commanded = ddq_ref + self.kd * (dq_ref - dq_motor) + self.kp * (q_ref - np.asarray(q_motor, dtype=float))
        friction = self.friction.torque(dq_motor)
        feedforward = self.rotor_inertia * ddq_ref + np.asarray(tau_spring, dtype=float) + friction
        total = np.clip(self.rotor_inertia * commanded + np.asarray(tau_spring, dtype=float) + friction,
                        -self.effort_limit, self.effort_limit)
        return TorqueCommand(total=total, feedforward=feedforward, feedback=total - feedforward)


def rollout(task: R1Task, plant: PlantParameters, settings: R1Settings, *, stiffness_scale: float = 1.0,
            extras: str | None = None, link_inertia: tuple[np.ndarray, np.ndarray] | None = None,
            time_step: float | None = None, sg: tuple[int, int] = (31, 3), visualize: bool = False
            ) -> dict[str, Any]:
    """Simulate one task; returns the sampled channels on its scored rows
    and the diagnostics (equilibrium, tracking, wall time)."""
    import mujoco

    from ..dataset_config import physics_step
    from ..plant_extras import PlantExtras
    from ..torque_runners import InitialState, TransmissionSpec, link_inertia_envelope
    from ..torque_runners import run_mujoco_elastic_torque
    from .equilibrium import spring_at_rest, static_deflection

    started = _time.perf_counter()
    base = plant.base
    bag = task.bag
    names = tuple(base.joint_names)
    if link_inertia is None:
        link_inertia = link_inertia_envelope(base.asset)
    extras = settings.extras if extras is None else extras
    stiffness = np.asarray(plant.transmission.stiffness, dtype=float) * float(stiffness_scale)
    transmission = TransmissionSpec.from_damping_ratio(stiffness, plant.transmission.damping_ratio,
                                                       plant.transmission.rotor_inertia, link_inertia[0],
                                                       link_inertia_floor=link_inertia[1])
    if extras == "link_friction":
        plant_extras = PlantExtras(link_friction=plant.link_friction, epsilon=plant.link_friction.epsilon)
    elif extras == "plant":
        plant_extras = plant.extras
    else:
        raise ValueError(f"R1 extras {extras!r}: link_friction or plant")
    if time_step is None:
        factor = getattr(plant_extras, "max_stiffness_factor", 1.0) if extras == "plant" else 1.0
        time_step = physics_step(transmission.required_time_step() / np.sqrt(factor), CONTROL_PERIOD)
    ratio = int(round(CONTROL_PERIOD / time_step))
    if abs(ratio * time_step - CONTROL_PERIOD) > 1e-12:
        raise ValueError(f"R1 time step {time_step} does not divide {CONTROL_PERIOD}")
    rows = np.arange(task.start, task.stop)
    if settings.reference_filter == "savgol":
        from scipy.signal import savgol_filter

        source = savgol_filter(bag.q, sg[0], sg[1], axis=0, mode="interp")[rows]
        reference, spline_info = spline_reference(bag.t[rows], source, names, time_step=time_step)
    else:
        reference, spline_info = spline_reference(bag.t[rows], bag.q[rows], names, time_step=time_step,
                                                  sigma=bag.sigma_q)
    spline_info["filter"] = settings.reference_filter
    effort = np.asarray([j.effort if j.effort is not None else np.inf for j in base.asset.resolve_active_joints()],
                        dtype=float)
    controller = MotorForcing(reference, transmission.rotor_inertia, plant.motor_friction,
                              natural_frequency=settings.track_bandwidth, effort_limit=effort)
    margin = controller.stability_margin(time_step)
    if margin > 0.5:
        raise ValueError(f"R1 controller stability margin {margin:.3f} > 0.5 at time step {time_step}")

    def model_setup(mujoco_, model):
        plant.apply_to_mujoco(mujoco_, model, BodyMap.build(mujoco_, model, base))

    offset = plant.sensor.offset(bag.run_id)
    equilibrium: dict[str, Any] = {}
    if task.quasi_static:
        deflection: Any = (bag.ft[task.start] - offset) / (plant.sensor.gain * stiffness)
        equilibrium["kind"] = "quasi_static"
    else:
        spring = spring_at_rest(plant_extras, transmission.stiffness, transmission.damping)

        def deflection(mujoco_, model, data, indices):
            delta, record = static_deflection(mujoco_, model, data, indices, spring, stiffness=transmission.stiffness)
            equilibrium.update(record, kind="static")
            return delta

    q0, dq0, _ = reference(0.0)
    result = run_mujoco_elastic_torque(
        base.asset, reference, controller, transmission, time_step=time_step, friction=plant.motor_friction,
        extras=plant_extras, initial_state=InitialState(q_motor=q0, dq_motor=dq0, deflection=deflection, dq_link=dq0),
        model_setup=model_setup, implicit_friction=True, visualize=visualize)
    # The 1 kHz rows of this task on the physics grid.
    local = np.arange(task.stop - task.start) * ratio
    tau = np.asarray(result["tau_link"], dtype=float)[local]
    q_motor = np.asarray(result["q_motor"], dtype=float)[local]
    q_link = np.asarray(result["q_link"], dtype=float)[local]
    ref = np.asarray(reference.position)
    scored = slice(task.score_from - task.start, task.score_to - task.start)
    measured = bag.q[task.start:task.stop]
    if "spring_torque" in equilibrium:
        measured_spring = (bag.ft[task.start] - offset) / plant.sensor.gain
        equilibrium["measured_spring_torque"] = measured_spring.tolist()
    return {
        "rows": np.arange(task.score_from, task.score_to), "tau_s": tau[scored], "q_motor": q_motor[scored],
        "q_link": q_link[scored], "q_ref": ref[scored], "dq_ref": np.asarray(reference.velocity)[scored],
        "ddq_ref": np.asarray(reference.acceleration)[scored],
        "diagnostics": {
            "bag": f"{bag.run_id}:{bag.bag}", "start": task.start, "stop": task.stop, "time_step": time_step,
            "steps": len(result["time"]), "stability_margin": margin, "spline": spline_info,
            "equilibrium": equilibrium,
            "tracking_sq_sum_measured": np.sum((q_motor[scored] - measured[scored]) ** 2, axis=0).tolist(),
            "tracking_sq_sum_reference": np.sum((q_motor[scored] - ref[scored]) ** 2, axis=0).tolist(),
            "tracking_max_measured": np.max(np.abs(q_motor[scored] - measured[scored]), axis=0).tolist(),
            "n_scored": int(scored.stop - scored.start),
            "transmission_frequency_hz": transmission.natural_frequency().tolist(),
            "wall_s": _time.perf_counter() - started,
        },
    }


def _run_task(arguments):
    task, plant, settings, options = arguments
    return rollout(task, plant, settings, **options)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class R1Engine:
    """R1 over a fixed set of bags; one MuJoCo rollout per bag (window) and theta."""

    mode = MODE

    def __init__(self, base: PlantBase, bags: Sequence[R1Bag], settings: R1Settings, *, jobs: int = 1,
                 stiffness_scale: float = 1.0, extras: str | None = None, time_step: float | None = None,
                 sg: tuple[int, int] = (31, 3), visualize: bool = False):
        from ..torque_runners import link_inertia_envelope

        if not bags:
            raise ValueError("R1Engine: no bags")
        self.base = base
        self.bags = tuple(bags)
        self.settings = settings
        self.jobs = max(1, int(jobs))
        self.options = {"stiffness_scale": float(stiffness_scale), "extras": extras, "time_step": time_step,
                        "sg": (int(sg[0]), int(sg[1])), "visualize": bool(visualize),
                        "link_inertia": link_inertia_envelope(base.asset)}
        self.tasks = [task for bag in self.bags for task in self.windows(bag)]
        if visualize:
            self.jobs = 1
        self.run_ids = tuple(sorted({b.run_id for b in self.bags}))
        self.n_rows = int(sum(b.n_rows for b in self.bags))
        self.last_diagnostics: list[dict[str, Any]] = []

    def windows(self, bag: R1Bag) -> list[R1Task]:
        n = len(bag.t)
        spec = self.settings.windows
        burn = int(round(self.settings.burn_in_s / CONTROL_PERIOD))
        if spec is None:
            return [R1Task(bag, 0, n, min(burn, n), n, False)]
        length = int(round(spec.length_s / CONTROL_PERIOD))
        stride = int(round(spec.stride_s / CONTROL_PERIOD))
        burn = int(round(spec.burn_in_s / CONTROL_PERIOD))
        if stride > length - burn or stride < 1:
            raise ValueError("replay.r1.windows: need 1 <= stride <= length - burn_in (windows tile the bag)")
        tasks, start = [], 0
        while True:
            stop = min(start + length, n)
            score_from = 0 if start == 0 else start + burn
            score_to = n if stop == n else start + burn + stride
            tasks.append(R1Task(bag, start, stop, score_from, min(score_to, stop), start > 0))
            if stop == n:
                return tasks
            start += stride

    def simulate(self, plant: PlantParameters) -> list[dict[str, Any]]:
        arguments = [(task, plant, self.settings, self.options) for task in self.tasks]
        if self.jobs == 1 or len(arguments) == 1:
            return [_run_task(a) for a in arguments]
        import multiprocessing

        with ProcessPoolExecutor(max_workers=min(self.jobs, len(arguments)),
                                 mp_context=multiprocessing.get_context("fork")) as pool:
            return list(pool.map(_run_task, arguments))

    def replay(self, plant: PlantParameters) -> list[ReplayResult]:
        outputs = self.simulate(plant)
        self.last_diagnostics = [o["diagnostics"] for o in outputs]
        gain = np.array(plant.sensor.gain, dtype=float)
        by_bag: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for task, output in zip(self.tasks, outputs):
            by_bag.setdefault((task.bag.run_id, task.bag.bag), []).append(output)
        results = []
        for bag in self.bags:
            parts = by_bag[(bag.run_id, bag.bag)]
            rows_all = np.concatenate([p["rows"] for p in parts])
            pick = np.isin(rows_all, bag.rows)
            stack = {k: np.concatenate([p[k] for p in parts])[pick]
                     for k in ("tau_s", "q_link", "q_ref", "dq_ref", "ddq_ref", "q_motor")}
            rows = rows_all[pick]
            offset = np.array(plant.sensor.offset(bag.run_id), dtype=float)
            measured_sq = np.sum([p["diagnostics"]["tracking_sq_sum_measured"] for p in parts], axis=0)
            reference_sq = np.sum([p["diagnostics"]["tracking_sq_sum_reference"] for p in parts], axis=0)
            count = sum(p["diagnostics"]["n_scored"] for p in parts)
            extra = {"tracking_rms_measured": np.sqrt(measured_sq / count).tolist(),
                     "tracking_rms_reference": np.sqrt(reference_sq / count).tolist(),
                     "tracking_max_measured": np.max([p["diagnostics"]["tracking_max_measured"] for p in parts],
                                                     axis=0).tolist(),
                     "wall_s": sum(p["diagnostics"]["wall_s"] for p in parts),
                     "equilibrium": parts[0]["diagnostics"]["equilibrium"],
                     "time_step": parts[0]["diagnostics"]["time_step"],
                     "spline": parts[0]["diagnostics"]["spline"],
                     "q_ref": stack["q_ref"], "q_motor": stack["q_motor"]}
            results.append(ReplayResult(run_id=bag.run_id, bag=bag.bag, kind=bag.kind, split=bag.split, mode=MODE,
                                        rows=rows, t=bag.t[rows], q=bag.q[rows], dq=stack["dq_ref"],
                                        ddq=stack["ddq_ref"], q_link=stack["q_link"], ft=bag.ft[rows],
                                        tau_s=stack["tau_s"], ft_sim=gain * stack["tau_s"] + offset, gain=gain,
                                        offset=offset, extra=extra))
        return results


def tracking_check(results: Sequence[ReplayResult], *, quanta: float = TRACKING_QUANTA) -> dict[str, Any]:
    """T-3.3 done-when: motor tracking RMS <= ``quanta`` encoder steps per
    joint on every bag, the simulated motor against the reference it is
    forced along (the controller's job). Against the measured q it is also
    reported: that adds the reference's own fit residual, the measured hold
    sigma(q) by construction (A6: ~20 quanta of standstill motion)."""
    per_bag = {f"{r.run_id}:{r.bag}": (np.asarray(r.extra["tracking_rms_reference"]) / ENCODER_QUANTUM).tolist()
               for r in results}
    measured = {f"{r.run_id}:{r.bag}": (np.asarray(r.extra["tracking_rms_measured"]) / ENCODER_QUANTUM).tolist()
                for r in results}
    worst = max((max(v) for v in per_bag.values()), default=0.0)
    return {"quantum_rad": ENCODER_QUANTUM, "limit_quanta": quanta, "per_bag_quanta": per_bag,
            "per_bag_quanta_measured": measured, "worst_quanta": worst, "pass": bool(worst <= quanta)}


#: Gate G-3 (S2R_01 T-3.3): R1 with K x 100 against R0, per joint RMS, N m.
G3_TOLERANCE = 0.05
G3_STIFFNESS_SCALE = 100.0
#: The band where K x 100 is the stiff limit on the iiwa (S2R_09 part B).
G3_LOWPASS_HZ = 10.0


def gate_g3(run: Any, base: PlantBase, plant: PlantParameters, *, bags: Sequence[str], r0: R0Settings,
            r1: R1Settings, jobs: int = 1, stiffness_scale: float = G3_STIFFNESS_SCALE,
            lowpass_hz: float = G3_LOWPASS_HZ) -> dict[str, Any]:
    """G-3: R1 at ``stiffness_scale`` x K (reference pre-filtered by R0's SG,
    so both engines see the same motion) against R0 at the same theta, every
    1 kHz row of ``bags``. Per bag and joint, the RMS of ``tau_s(R1) - tau_s``:

    * ``spec``: R0 as configured (W = 31 SG derivatives of the measured q) --
      S2R_01's criterion, <= 0.05 N m;
    * ``same_kinematics``: R0 on R1's own reference ``(q, dq, ddq)``: the
      engines alone (elastic MuJoCo + equilibrium + forcing vs rigid
      Pinocchio + intermediate bodies);
    * ``lowpass``: ``same_kinematics`` after a zero-phase 4th-order
      Butterworth at ``lowpass_hz`` (the first and last 0.2 s dropped).

    Plus the motor tracking check (T-3.3) and the wall time per bag."""
    from scipy.signal import butter, sosfiltfilt

    from .replay_r0 import R0Bag, R0Engine
    from .replay_r0 import prepare_bags as prepare_r0

    chosen = [b for b in run.bags if b.bag in set(bags)]
    full = dataclasses.replace(r0, decimate=1, hold_decimate=1)
    stiff = dataclasses.replace(plant, transmission=dataclasses.replace(
        plant.transmission, stiffness=np.asarray(plant.transmission.stiffness, dtype=float) * stiffness_scale))
    engine = R1Engine(base, prepare_bags(chosen, full, sigma_q={run.run_id: run_sigma_q(run)}),
                      dataclasses.replace(r1, reference_filter="savgol", windows=None, burn_in_s=0.0),
                      jobs=jobs, stiffness_scale=stiffness_scale, sg=(r0.sg_window, r0.sg_poly))
    results = engine.replay(plant)
    spec_engine = R0Engine(base, prepare_r0(chosen, full, time_step=1.0 / run.rate_hz))
    _, tau_spec, _ = spec_engine.simulate(stiff)
    sos = butter(4, lowpass_hz, fs=1.0 / CONTROL_PERIOD, output="sos")
    edge = int(round(0.2 / CONTROL_PERIOD))
    per_bag = {}
    for result, rows in zip(results, spec_engine.slices):
        kin = R0Bag(result.run_id, result.bag, result.kind, "", result.rows, result.t, result.extra["q_ref"],
                    result.dq, result.ddq, result.ft)
        _, tau_kin, _ = R0Engine(base, [kin], intermediate=spec_engine.intermediate).simulate(stiff)
        spec = result.tau_s - tau_spec[rows]
        same = result.tau_s - tau_kin
        low = sosfiltfilt(sos, result.tau_s, axis=0) - sosfiltfilt(sos, tau_kin, axis=0)
        per_bag[result.bag] = {
            "spec": np.sqrt(np.mean(spec ** 2, axis=0)).tolist(),
            "same_kinematics": np.sqrt(np.mean(same ** 2, axis=0)).tolist(),
            "lowpass": np.sqrt(np.mean(low[edge:-edge] ** 2, axis=0)).tolist(),
            "tracking_rms_reference_quanta": (np.asarray(result.extra["tracking_rms_reference"])
                                              / ENCODER_QUANTUM).tolist(),
            "wall_s": result.extra["wall_s"], "time_step": result.extra["time_step"],
            "equilibrium_residual_nm": result.extra["equilibrium"].get("residual_max_nm")}

    def worst(key):
        return float(max(max(v[key]) for v in per_bag.values()))

    return {"gate": "G-3", "definition": "S2R_01 T-3.3", "run": run.run_id, "bags": list(per_bag),
            "stiffness_scale": stiffness_scale, "tolerance_nm": G3_TOLERANCE, "lowpass_hz": lowpass_hz,
            "theta": dict(plant.theta), "per_bag": per_bag,
            "worst": {k: worst(k) for k in ("spec", "same_kinematics", "lowpass")},
            "pass_spec": worst("spec") <= G3_TOLERANCE, "pass_lowpass": worst("lowpass") <= G3_TOLERANCE,
            "tracking_worst_quanta": float(max(max(v["tracking_rms_reference_quanta"]) for v in per_bag.values())),
            "joint_names": list(base.joint_names)}


def default_jobs() -> int:
    return max(1, (os.cpu_count() or 2) - 1)


__all__ = ["CONTROL_PERIOD", "ENCODER_QUANTUM", "MODE", "MotorForcing", "R1Bag", "R1Engine", "R1Task", "TRACKING_QUANTA",
           "G3_LOWPASS_HZ", "G3_STIFFNESS_SCALE", "G3_TOLERANCE", "default_jobs", "discrepancy_spline", "gate_g3", "prepare_bags", "rollout", "run_sigma_q", "spline_reference",
           "tracking_check"]
