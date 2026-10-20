"""R2: closed-loop replay, the digital twin (S2R_01 §4.3, T-3.4).

The robot's own command (the FRI command echo at 1 kHz, S2R_04 A-10; one
fresh command per cycle, S2R_06 F-8: no hold model) drives the KUKA
position-loop emulation (:mod:`.controllers`) on the elastic plant
(MuJoCo, the production runner): rotor inertia as armature, motor friction,
link friction (velocity-implicit), K and zeta from theta. The plant starts
at the measured motor pose, at rest, in static equilibrium (T-3.1 hook).

Observed: ``ft`` (``g tau_s + o``) and the **tracking error** ``q - cmd``,
real against simulated (A-2 (e)), per joint. ``controller.delay_cycles``
shifts the command; ``controller.drive_rate`` sets the loop's rate (the
runner's control decimation on a physics step that divides its period).
"""

from __future__ import annotations

import dataclasses
import time as _time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from .config import R0Settings, R2Settings
from .plant import BodyMap, IntermediateBodies, PlantBase, PlantParameters
from .real_runs import ReplayBag
from .result import ReplayResult

MODE = "R2"
COMMAND_PERIOD = 1.0e-3


@dataclass(frozen=True)
class R2Bag:
    run_id: str
    bag: str
    kind: str
    split: str
    t: np.ndarray        # (N,) 1 kHz
    q: np.ndarray        # (N, n) measured motor
    ft: np.ndarray       # (N, n)
    command: np.ndarray  # (N, n) FRI command echo
    rows: np.ndarray     # scored rows (R0's grid)

    @property
    def n_rows(self) -> int:
        return len(self.rows)


def prepare_bags(bags: Iterable[ReplayBag], r0: R0Settings, *, split_of: Callable[[str, str], str | None] | None
                 = None) -> list[R2Bag]:
    out = []
    for bag in bags:
        if bag.cmd_echo is None:
            raise ValueError(f"{bag.run_id}:{bag.bag} has no command echo (R2 needs it, S2R_04 A-10)")
        split = (split_of(bag.run_id, bag.bag) if split_of is not None else None) or ""
        step = int(r0.hold_decimate if bag.kind == "hold" else r0.decimate)
        out.append(R2Bag(bag.run_id, bag.bag, bag.kind, split, np.asarray(bag.t, dtype=float),
                         np.asarray(bag.q, dtype=float), np.asarray(bag.ft, dtype=float),
                         np.asarray(bag.cmd_echo, dtype=float), np.arange(0, bag.n_rows, step)))
    return out


def rollout(bag: R2Bag, plant: PlantParameters, settings: R2Settings, *,
            link_inertia: tuple[np.ndarray, np.ndarray] | None = None,
            intermediate: IntermediateBodies | None = None, time_step: float | None = None,
            visualize: bool = False) -> dict[str, Any]:
    from ..dataset_config import physics_step
    from ..plant_extras import PlantExtras
    from ..torque_runners import InitialState, TransmissionSpec, link_inertia_envelope, run_mujoco_elastic_torque
    from .controllers import command_reference, kuka_position_emulation
    from .equilibrium import spring_at_rest, static_deflection

    started = _time.perf_counter()
    base = plant.base
    emulation = plant.controller
    if emulation is None:
        raise ValueError("R2 needs the controller.* parameters")
    if link_inertia is None:
        link_inertia = link_inertia_envelope(base.asset)
    transmission = TransmissionSpec.from_damping_ratio(plant.transmission.stiffness, plant.transmission.damping_ratio,
                                                       plant.transmission.rotor_inertia, link_inertia[0],
                                                       link_inertia_floor=link_inertia[1])
    extras = PlantExtras(link_friction=plant.link_friction, epsilon=plant.link_friction.epsilon)
    drive_period = 1.0 / float(emulation.drive_rate)
    if time_step is None:
        time_step = physics_step(transmission.required_time_step(), drive_period)
    decimation = int(round(drive_period / time_step))
    ratio = int(round(COMMAND_PERIOD / time_step))
    if abs(ratio * time_step - COMMAND_PERIOD) > 1e-12 or abs(decimation * time_step - drive_period) > 1e-12:
        raise ValueError(f"R2 time step {time_step} must divide the drive period {drive_period} and 1 ms")
    reference = command_reference(bag.t, bag.command, base.joint_names, delay_cycles=emulation.delay_cycles,
                                  period=COMMAND_PERIOD, time_step=time_step)
    controller = kuka_position_emulation(plant, reference, transmission, intermediate=intermediate,
                                         emulation=settings.emulation)
    margin = float(np.max(controller.kd) * drive_period)
    if margin > 0.5:
        raise ValueError(f"R2 controller margin kd * T_drive = {margin:.3f} > 0.5")

    def model_setup(mujoco_, model):
        plant.apply_to_mujoco(mujoco_, model, BodyMap.build(mujoco_, model, base))

    spring = spring_at_rest(extras, transmission.stiffness, transmission.damping)
    equilibrium: dict[str, Any] = {}

    def deflection(mujoco_, model, data, indices):
        delta, record = static_deflection(mujoco_, model, data, indices, spring, stiffness=transmission.stiffness)
        equilibrium.update(record)
        return delta

    q0 = bag.q[0]
    zeros = np.zeros_like(q0)
    result = run_mujoco_elastic_torque(
        base.asset, reference, controller, transmission, time_step=time_step, friction=plant.motor_friction,
        control_decimation=decimation, extras=extras, implicit_friction=True,
        initial_state=InitialState(q_motor=q0, dq_motor=zeros, deflection=deflection, dq_link=zeros),
        model_setup=model_setup, visualize=visualize)
    local = np.arange(len(bag.t)) * ratio
    q_motor = np.asarray(result["q_motor"], dtype=float)[local]
    tau = np.asarray(result["tau_link"], dtype=float)[local]
    error_sim = q_motor - bag.command
    error_real = bag.q - bag.command
    return {"tau_s": tau, "q_motor": q_motor, "q_link": np.asarray(result["q_link"], dtype=float)[local],
            "tau_motor": np.asarray(result["tau_motor"], dtype=float)[local],
            "tracking": {"real_rms": np.sqrt(np.mean(error_real ** 2, axis=0)).tolist(),
                         "sim_rms": np.sqrt(np.mean(error_sim ** 2, axis=0)).tolist(),
                         "real_max": np.max(np.abs(error_real), axis=0).tolist(),
                         "sim_max": np.max(np.abs(error_sim), axis=0).tolist(),
                         "difference_rms": np.sqrt(np.mean((error_sim - error_real) ** 2, axis=0)).tolist(),
                         "correlation": [float(np.corrcoef(error_sim[:, j], error_real[:, j])[0, 1])
                                         if np.std(error_sim[:, j]) > 0 and np.std(error_real[:, j]) > 0 else None
                                         for j in range(q0.size)]},
            "diagnostics": {"bag": f"{bag.run_id}:{bag.bag}", "time_step": time_step, "decimation": decimation,
                            "steps": len(result["time"]), "margin": margin, "equilibrium": equilibrium,
                            "wall_s": _time.perf_counter() - started}}


def _run(arguments):
    bag, plant, settings, options = arguments
    try:
        return rollout(bag, plant, settings, **options)
    except Exception as error:  # a diverged rollout is a result (T-3.4 counts them)
        from ..torque_runners import RolloutDiverged

        return {"error": f"{type(error).__name__}: {error}", "diverged": isinstance(error, RolloutDiverged),
                "diagnostics": {"bag": f"{bag.run_id}:{bag.bag}"}}


class R2Engine:
    """R2 over a fixed set of bags; one closed-loop rollout per bag and theta."""

    mode = MODE

    def __init__(self, base: PlantBase, bags: Sequence[R2Bag], settings: R2Settings, *, jobs: int = 1,
                 time_step: float | None = None, visualize: bool = False):
        from ..torque_runners import link_inertia_envelope

        if not bags:
            raise ValueError("R2Engine: no bags")
        self.base = base
        self.bags = tuple(bags)
        self.settings = settings
        self.jobs = max(1, int(jobs))
        self.options = {"link_inertia": link_inertia_envelope(base.asset), "time_step": time_step,
                        "intermediate": IntermediateBodies.from_elastic_build(base), "visualize": bool(visualize)}
        if visualize:
            self.jobs = 1
        self.last_outputs: list[dict[str, Any]] = []

    def simulate(self, plant: PlantParameters) -> list[dict[str, Any]]:
        arguments = [(bag, plant, self.settings, self.options) for bag in self.bags]
        if self.jobs == 1 or len(arguments) == 1:
            return [_run(a) for a in arguments]
        import multiprocessing

        with ProcessPoolExecutor(max_workers=min(self.jobs, len(arguments)),
                                 mp_context=multiprocessing.get_context("fork")) as pool:
            return list(pool.map(_run, arguments))

    def replay(self, plant: PlantParameters) -> list[ReplayResult]:
        """Results for the bags that ran (a diverged bag is in ``last_outputs``)."""
        outputs = self.simulate(plant)
        self.last_outputs = outputs
        gain = np.array(plant.sensor.gain, dtype=float)
        results = []
        for bag, output in zip(self.bags, outputs):
            if "error" in output:
                continue
            rows = bag.rows
            offset = np.array(plant.sensor.offset(bag.run_id), dtype=float)
            tau = output["tau_s"][rows]
            zeros = np.zeros_like(tau)
            results.append(ReplayResult(run_id=bag.run_id, bag=bag.bag, kind=bag.kind, split=bag.split, mode=MODE,
                                        rows=rows, t=bag.t[rows], q=bag.q[rows], dq=zeros, ddq=zeros,
                                        q_link=output["q_link"][rows], ft=bag.ft[rows], tau_s=tau,
                                        ft_sim=gain * tau + offset, gain=gain, offset=offset,
                                        extra={"tracking": output["tracking"], **output["diagnostics"],
                                               "q_motor_sim": output["q_motor"][rows],
                                               "command": bag.command[rows]}))
        return results


def nominal_controller_theta(base: PlantBase, *, bandwidth_hz: float = 10.0, damping_ratio: float = 1.0,
                             model_scale: float = 1.0, delay_cycles: int = 1, drive_rate: float = 4000.0
                             ) -> dict[str, float]:
    """The hand-set R2 controller (S2R_01 T-3.4: nominal theta, a hand-set
    bandwidth): the ``controller.*`` names :meth:`PlantParameters.from_theta`
    reads."""
    theta = {f"controller.{j}.bandwidth": 2.0 * np.pi * bandwidth_hz for j in base.joint_names}
    theta.update({f"controller.{j}.damping_ratio": damping_ratio for j in base.joint_names})
    theta.update({"controller.model_scale": model_scale, "controller.delay_cycles": float(delay_cycles),
                  "controller.drive_rate": drive_rate})
    return theta


__all__ = ["COMMAND_PERIOD", "MODE", "R2Bag", "R2Engine", "nominal_controller_theta", "prepare_bags", "rollout"]
