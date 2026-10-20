"""R0: inverse dynamics on the recorded state (S2R_01 §4.1, T-3.2).

Nothing is integrated. Per kept row k,

    q_link = q - (ft - o) / (g K)                     (deflection: measured ft)
    tau_s  = RNEA_theta(q_link, dq, ddq) + f_link(dq) (rigid chain + link friction)
    ft_sim = g * tau_s + o[run]                       (iiwa JTS observation model)

with ``dq``, ``ddq`` a zero-phase Savitzky-Golay derivative of the measured
motor ``q`` (default W = 31, poly 3), rows decimated (default 5x, 200 Hz),
holds at ``dq = ddq = 0`` (first 0.5 s already dropped by the sidecar).

* **Plant.** The Pinocchio model carries theta's inertials (in place,
  :meth:`PlantParameters.apply_to_pinocchio`) **plus the elastic build's
  intermediate bodies** (S2R_04 A-9), so R0 and R1 calibrate the same plant.
* **Jacobian** (:meth:`R0Engine.jacobian`), d ft_sim / d theta in physical
  units, analytic for every group the R0 stages free:

  * inertials through the joint-torque regressor ``Y`` (``tau = Y pi``) and
    the chain rule of the physical parameterisation (mass, COM, inertia
    scale -> the 10 standard parameters of the link);
  * link friction: ``dq`` and ``tanh(dq / eps)``;
  * offset, gain and stiffness: the direct term plus the deflection's,
    ``d tau / d q_link`` (RNEA derivatives) times ``d q_link / d theta``.

  Anything else falls back to central differences.
* **Gate G-1** (:func:`gate_g1`): the contract's ``baseline_rms_rigid``
  recipe, reproduced exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from scipy.signal import savgol_filter

from .config import R0Settings
from .plant import IntermediateBodies, PlantBase, PlantParameters
from .real_runs import RealRun, ReplayBag
from .result import ReplayResult

MODE = "R0"
_INERTIAL_FIELDS = ("mass", "com_x", "com_y", "com_z", "inertia_scale")
# Pinocchio's standard-parameter order of the rotational inertia about the frame origin.
_INERTIA_ORDER = ((0, 0), (0, 1), (1, 1), (0, 2), (1, 2), (2, 2))


@dataclass(frozen=True)
class R0Bag:
    """The theta-independent part of a bag's R0 replay: kept rows and the
    measured state and derivatives on them."""

    run_id: str
    bag: str
    kind: str
    split: str
    rows: np.ndarray
    t: np.ndarray
    q: np.ndarray
    dq: np.ndarray
    ddq: np.ndarray
    ft: np.ndarray

    @property
    def n_rows(self) -> int:
        return len(self.t)


def prepare_bags(bags: Iterable[ReplayBag], settings: R0Settings, *, time_step: float,
                 split_of: Callable[[str, str], str | None] | None = None, keep_unsplit: bool = True,
                 recipe: str = "replay", contract_sg: tuple[int, int] = (5, 3)) -> list[R0Bag]:
    """Kept rows and derivatives per bag.

    ``recipe="replay"`` is §4.1 (SG on ``q``, decimation, holds at rest,
    optional quasi-static row filter). ``recipe="contract"`` is the contract
    baselines' state (``erd_recording.contract._per_bag_baselines``): the
    dataset's ``dq``, ``ddq`` = SG(``contract_sg``) of ``dq``, every row.
    """
    prepared = []
    for bag in bags:
        split = (split_of(bag.run_id, bag.bag) if split_of is not None else None) or ""
        if not split and not keep_unsplit:
            continue
        n = bag.q.shape[1]
        if recipe == "contract":
            window, poly = contract_sg
            dq = np.asarray(bag.dq, dtype=float)
            ddq = savgol_filter(dq, window, poly, deriv=1, delta=time_step, axis=0, mode="interp")
            rows = np.arange(bag.n_rows)
        elif recipe == "replay":
            if bag.kind == "hold":
                dq = ddq = np.zeros_like(bag.q)
                rows = np.arange(0, bag.n_rows, int(settings.hold_decimate))
            else:
                dq = savgol_filter(bag.q, settings.sg_window, settings.sg_poly, deriv=1, delta=time_step, axis=0,
                                   mode="interp")
                ddq = savgol_filter(bag.q, settings.sg_window, settings.sg_poly, deriv=2, delta=time_step, axis=0,
                                    mode="interp")
                rows = np.arange(0, bag.n_rows, int(settings.decimate))
                if settings.rows == "quasi_static":
                    calm = np.all(np.abs(dq) < settings.quasi_static_velocity, axis=1) & \
                        np.all(np.abs(ddq) < settings.quasi_static_acceleration, axis=1)
                    rows = rows[calm[rows]]
        else:
            raise ValueError(f"unknown R0 recipe {recipe!r}")
        if len(rows) == 0:
            continue
        prepared.append(R0Bag(run_id=bag.run_id, bag=bag.bag, kind=bag.kind, split=split, rows=rows,
                              t=np.asarray(bag.t, dtype=float)[rows], q=np.asarray(bag.q, dtype=float)[rows],
                              dq=np.asarray(dq, dtype=float)[rows].reshape(-1, n),
                              ddq=np.asarray(ddq, dtype=float)[rows].reshape(-1, n),
                              ft=np.asarray(bag.ft, dtype=float)[rows]))
    return prepared


class R0Engine:
    """R0 over a fixed set of bags; theta changes per call, the model is
    built once and updated in place."""

    mode = MODE

    def __init__(self, base: PlantBase, bags: Sequence[R0Bag], *, deflection: bool = True,
                 intermediate: IntermediateBodies | None | str = "auto", friction: bool = True):
        from .. import identification as idn

        if not bags:
            raise ValueError("R0Engine: no bags")
        self.base = base
        self.bags = tuple(bags)
        self.deflection = bool(deflection)
        self.friction = bool(friction)
        self.intermediate = IntermediateBodies.from_elastic_build(base) if intermediate == "auto" else intermediate
        self.pin, self.model, self.data = idn.build_model(base.asset)
        self.n = len(base.joint_names)
        self.q = np.vstack([b.q for b in self.bags])
        self.dq = np.vstack([b.dq for b in self.bags])
        self.ddq = np.vstack([b.ddq for b in self.bags])
        self.ft = np.vstack([b.ft for b in self.bags])
        bounds = np.cumsum([0] + [b.n_rows for b in self.bags])
        self.slices = [slice(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:])]
        self.run_ids = tuple(sorted({b.run_id for b in self.bags}))
        self.row_run = np.concatenate([np.full(b.n_rows, self.run_ids.index(b.run_id)) for b in self.bags])
        self.n_rows = int(bounds[-1])

    # -- forward ---------------------------------------------------------------

    def _offsets(self, plant: PlantParameters) -> np.ndarray:
        per_run = np.vstack([plant.sensor.offset(run) for run in self.run_ids])
        return per_run[self.row_run]

    def _stiffness(self, plant: PlantParameters) -> np.ndarray | None:
        k = np.asarray(plant.transmission.stiffness, dtype=float)
        if not self.deflection or not np.all(np.isfinite(k)):
            return None
        return k

    def set_plant(self, plant: PlantParameters) -> None:
        """Theta's inertials (+ the intermediate bodies) into the model."""
        plant.apply_to_pinocchio(self.pin, self.model)
        if self.intermediate is not None:
            self.intermediate.add_to(self.model)

    def q_link(self, plant: PlantParameters, offsets: np.ndarray | None = None, *,
               deflection_ft: np.ndarray | None = None, deflection: bool | None = None) -> np.ndarray:
        """q - (ft - o) / (g K); ``deflection_ft`` replaces the measured ft
        (the twin's fixed point), ``deflection=False`` gives q."""
        k = self._stiffness(plant) if deflection is None or deflection else None
        if k is None:
            return self.q
        offsets = self._offsets(plant) if offsets is None else offsets
        ft = self.ft if deflection_ft is None else deflection_ft
        return self.q - (ft - offsets) / (plant.sensor.gain * k)

    def tau_s(self, plant: PlantParameters, q_link: np.ndarray | None = None) -> np.ndarray:
        """The spring torque on every row (rigid chain at q_link + link friction)."""
        self.set_plant(plant)
        q_link = self.q_link(plant) if q_link is None else q_link
        pin, model, data = self.pin, self.model, self.data
        tau = np.empty_like(q_link)
        for k in range(len(q_link)):
            tau[k] = pin.rnea(model, data, q_link[k], self.dq[k], self.ddq[k])
        if self.friction:
            tau += plant.link_friction.torque(self.dq)
        return tau

    def simulate(self, plant: PlantParameters, *, deflection_ft: np.ndarray | None = None,
                 deflection: bool | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(q_link, tau_s, ft_sim)`` stacked over every row."""
        offsets = self._offsets(plant)
        q_link = self.q_link(plant, offsets, deflection_ft=deflection_ft, deflection=deflection)
        tau = self.tau_s(plant, q_link)
        return q_link, tau, plant.sensor.gain * tau + offsets

    def replay(self, plant: PlantParameters) -> list[ReplayResult]:
        q_link, tau, ft_sim = self.simulate(plant)
        return self.results(plant, q_link, tau, ft_sim)

    def results(self, plant: PlantParameters, q_link: np.ndarray, tau: np.ndarray, ft_sim: np.ndarray
                ) -> list[ReplayResult]:
        out = []
        gain = np.array(plant.sensor.gain, dtype=float)
        for bag, rows in zip(self.bags, self.slices):
            out.append(ReplayResult(run_id=bag.run_id, bag=bag.bag, kind=bag.kind, split=bag.split, mode=MODE,
                                    rows=bag.rows, t=bag.t, q=bag.q, dq=bag.dq, ddq=bag.ddq, q_link=q_link[rows],
                                    ft=bag.ft, tau_s=tau[rows], ft_sim=ft_sim[rows], gain=gain,
                                    offset=np.array(plant.sensor.offset(bag.run_id), dtype=float)))
        return out

    # -- Jacobian ----------------------------------------------------------------

    def jacobian(self, plant: PlantParameters, names: Sequence[str], *,
                 theta: Mapping[str, float] | None = None) -> np.ndarray:
        """d ft_sim / d theta[name] (physical units), rows flattened row-major
        ((row, joint) -> row * n + joint), one column per name.

        ``theta`` (default ``plant.theta``) is needed only for the
        finite-difference fallback."""
        n, pin, model, data = self.n, self.pin, self.model, self.data
        parsed = [_parse(name, self.base) for name in names]
        offsets = self._offsets(plant)
        q_link = self.q_link(plant, offsets)
        self.set_plant(plant)
        stiffness = self._stiffness(plant)
        gain = np.asarray(plant.sensor.gain, dtype=float)
        inertial = [i for i, p in enumerate(parsed) if p[0] == "inertial"]
        need_dq = stiffness is not None and any(p[0] in ("offset", "gain", "stiffness") for p in parsed)
        d_pi = self._dpi_dtheta(plant, [parsed[i] for i in inertial]) if inertial else None
        jac = np.zeros((self.n_rows, n, len(names)))
        tau = np.empty((self.n_rows, n))
        dtau_dq = np.empty((self.n_rows, n, n)) if need_dq else None
        for k in range(self.n_rows):
            tau[k] = pin.rnea(model, data, q_link[k], self.dq[k], self.ddq[k])
            if inertial:
                y = pin.computeJointTorqueRegressor(model, data, q_link[k], self.dq[k], self.ddq[k])
                jac[k][:, inertial] = gain[:, None] * (y @ d_pi)
            if need_dq:
                pin.computeRNEADerivatives(model, data, q_link[k], self.dq[k], self.ddq[k])
                dtau_dq[k] = data.dtau_dq
        if self.friction:
            tau += plant.link_friction.torque(self.dq)
        epsilon = plant.link_friction.epsilon
        signal = self.ft - offsets
        fallback = []
        for column, (group, j, extra) in enumerate(parsed):
            if group == "inertial":
                continue
            if group == "viscous":
                jac[:, j, column] = gain[j] * self.dq[:, j] if self.friction else 0.0
            elif group == "coulomb":
                jac[:, j, column] = gain[j] * np.tanh(self.dq[:, j] / epsilon) if self.friction else 0.0
            elif group == "offset":
                rows = self.row_run == self.run_ids.index(extra) if extra in self.run_ids else \
                    np.zeros(self.n_rows, dtype=bool)
                jac[rows, j, column] = 1.0
                if need_dq:
                    jac[rows, :, column] += gain[None, :] * dtau_dq[rows][:, :, j] / (gain[j] * stiffness[j])
            elif group == "gain":
                jac[:, j, column] = tau[:, j]
                if need_dq:
                    dql = signal[:, j] / (gain[j] ** 2 * stiffness[j])
                    jac[:, :, column] += gain[None, :] * dtau_dq[:, :, j] * dql[:, None]
            elif group == "stiffness":
                if need_dq:
                    dql = signal[:, j] / (gain[j] * stiffness[j] ** 2)
                    jac[:, :, column] += gain[None, :] * dtau_dq[:, :, j] * dql[:, None]
            else:
                fallback.append(column)
        jac = jac.reshape(self.n_rows * n, len(names))
        if fallback:
            theta = dict(plant.theta if theta is None else theta)
            for column in fallback:
                jac[:, column] = self._central_difference(theta, names[column])
        return jac

    def _central_difference(self, theta: Mapping[str, float], name: str, *, relative_step: float = 1e-6
                            ) -> np.ndarray:
        x = float(theta[name])
        h = relative_step * max(abs(x), 1e-3)
        sims = []
        for sign in (1.0, -1.0):
            probe = dict(theta)
            probe[name] = x + sign * h
            sims.append(self.simulate(PlantParameters.from_theta(self.base, probe, runs=self.run_ids))[2])
        return ((sims[0] - sims[1]) / (2.0 * h)).reshape(-1)

    def _dpi_dtheta(self, plant: PlantParameters, parsed: Sequence[tuple[str, int, str]]) -> np.ndarray:
        """d pi / d theta (10 n x p): pi per link = [m, m c, I_o] with
        I_o = s I_c0 + m (|c|^2 E - c c^T) (S2R_01 §5 parameterisation)."""
        out = np.zeros((10 * self.n, len(parsed)))
        for column, (_, index, field) in enumerate(parsed):
            link = self.base.link_names[index]
            inertial = plant.inertials[link]
            m, c = inertial.mass, inertial.com
            block = np.zeros(10)
            if field == "mass":
                block[0] = 1.0
                block[1:4] = c
                block[4:] = _vech(float(c @ c) * np.eye(3) - np.outer(c, c))
            elif field.startswith("com_"):
                axis = "xyz".index(field[-1])
                e = np.zeros(3)
                e[axis] = 1.0
                block[1 + axis] = m
                block[4:] = m * _vech(2.0 * c[axis] * np.eye(3) - np.outer(e, c) - np.outer(c, e))
            elif field == "inertia_scale":
                # PlantParameters.from_theta: I_c = inertia_scale * the base tensor.
                block[4:] = _vech(self.base.inertials[link].inertia)
            out[10 * index:10 * index + 10, column] = block
        return out


def _vech(matrix: np.ndarray) -> np.ndarray:
    return np.asarray([matrix[r, c] for r, c in _INERTIA_ORDER])


def _parse(name: str, base: PlantBase) -> tuple[str, int, str]:
    """``(group, index, extra)``: index = link (inertial) or joint index."""
    parts = name.split(".")
    if parts[0] == "inertial" and len(parts) == 3 and parts[2] in _INERTIAL_FIELDS and parts[1] in base.link_names:
        return "inertial", base.link_names.index(parts[1]), parts[2]
    if parts[:2] == ["friction", "link"] and len(parts) == 4 and parts[2] in base.joint_names:
        return parts[3], base.joint_names.index(parts[2]), ""
    if parts[:2] == ["sensor", "ft"] and len(parts) == 4 and parts[2] in base.joint_names:
        field, _, run = parts[3].partition("@")
        if field in ("offset", "gain"):
            return field, base.joint_names.index(parts[2]), run
    if parts[0] == "transmission" and len(parts) == 3 and parts[2] == "stiffness" and parts[1] in base.joint_names:
        return "stiffness", base.joint_names.index(parts[1]), ""
    return "other", -1, name


# ---------------------------------------------------------------------------
# Gate G-1
# ---------------------------------------------------------------------------


def gate_g1(run: RealRun, base: PlantBase, *, intermediate: IntermediateBodies | None = None) -> dict[str, Any]:
    """S2R_01 T-3.2 G-1: ``erd_recording.contract._per_bag_baselines``'s
    rigid baseline, through the R0 engine: the dataset's ``dq``, ``ddq`` =
    SG(5, 3) of ``dq``, no friction, K -> inf, o = 0, g = 1, nominal
    inertials (no intermediate bodies unless given: they are not in the
    contract's model). Median over the excitation bags of RMS(ft - rigid)."""
    differentiation = run.contract.get("differentiation") or {}
    sg = (int(differentiation.get("sg_window", 5)), int(differentiation.get("sg_poly", 3)))
    time_step = float(run.dataset_manifest["sample_time_step"])
    bags = prepare_bags([b for b in run.bags if b.kind == "excitation"], R0Settings(), time_step=time_step,
                        recipe="contract", contract_sg=sg)
    engine = R0Engine(base, bags, deflection=False, intermediate=intermediate, friction=False)
    plant = PlantParameters.nominal(base)
    tau = engine.tau_s(plant)
    per_bag = [float(np.sqrt(np.mean((engine.ft[s] - tau[s]) ** 2))) for s in engine.slices]
    value = float(np.median(per_bag))
    contract = float(run.contract["baselines"]["splits"]["all"]["baseline_rms_rigid"])
    return {"run": run.run_id, "baseline_rms_rigid": value, "contract": contract,
            "relative_error": abs(value - contract) / contract, "per_bag": per_bag,
            "intermediate_bodies": intermediate is not None, "sg": list(sg)}


def r0_settings_label(settings: R0Settings) -> str:
    return (f"SG({settings.sg_window}, {settings.sg_poly}) on q, decimate {settings.decimate} "
            f"(holds {settings.hold_decimate}), deflection {settings.deflection}, rows {settings.rows}")


__all__ = ["MODE", "R0Bag", "R0Engine", "gate_g1", "prepare_bags", "r0_settings_label"]
