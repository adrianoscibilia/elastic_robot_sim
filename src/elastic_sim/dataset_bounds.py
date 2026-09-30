"""Load-time explicit-integration bound for every explicitly applied term (`R6_00` Sec 3).

The runners apply friction and the spring corrections as generalized forces,
which MuJoCo's Euler step integrates explicitly.  An explicit velocity-
dependent force of slope ``s`` on a degree of freedom of effective inertia
``M_eff`` is stable only while ``s h / M_eff`` stays below about two; `R6_00`
asks for ``<= 1`` on every DOF:

    (b + c / epsilon + d_spring,explicit) * h / M_eff <= 1

``M_eff`` is the DOF's smallest effective inertia over the inertia envelope,
``1 / max_q (M(q)^-1)_jj`` with the armature included.  On the elastic chain,
whose coordinates are the rotor angle ``theta`` and the deflection
``e = q - theta``, the kinetic energy is ``theta_dot' J theta_dot / 2 +
q_dot' M q_dot / 2``, so the inverse mass matrix in ``(theta, e)`` is
``[[J^-1, -J^-1], [-J^-1, J^-1 + M^-1]]``: the rotor coordinate's effective
inertia is exactly the rotor's ``J``, and a force along the *link* coordinate
``theta + e`` meets the link's own ``1 / (M^-1)_jj``.

The round-5 rigid divergence (`R5_11` Sec 3.2) is this bound at 180 on the
iiwa's A7.  A violation is a config error that names the joint and the term.

Terms and where they act:

* **motor friction** (`simulation.motor_friction`, explicit): the rotor DOF
  of an elastic robot, the joint DOF of the rigid tier;
* **link friction** (`plant_extras.link_friction`): the link coordinate.
  Explicit in rounds 4-5; round 6 integrates it velocity-implicitly
  (`torque_runners._ImplicitLinkFriction`), which takes it out of this bound
  -- it is still tabulated, with the value it *would* have, so the reason is
  visible;
* **spring damping, explicit part**: the transmission damper itself is
  MuJoCo's implicit ``dof_damping``; what is explicit is the transmission
  error's velocity coupling, ``d (A n)^2`` on the rotor DOF.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .dataset_config import DatasetConfig, _per_joint, physics_step


@dataclass(frozen=True)
class BoundRow:
    tier: str
    joint: str
    dof: str
    term: str
    slope: float
    time_step: float
    effective_inertia: float
    integration: str

    @property
    def value(self) -> float:
        return float(self.slope * self.time_step / self.effective_inertia)

    @property
    def enforced(self) -> bool:
        return self.integration == "explicit"

    def as_dict(self) -> dict[str, Any]:
        return {"tier": self.tier, "joint": self.joint, "dof": self.dof, "term": self.term,
                "slope": self.slope, "time_step": self.time_step,
                "effective_inertia": self.effective_inertia, "value": self.value,
                "integration": self.integration, "enforced": self.enforced}


def _inverse_mass_envelope(asset: Any, config: DatasetConfig, armature: np.ndarray | None,
                           n_samples: int, seed: int) -> np.ndarray:
    """``max_q (M(q) + diag(armature))^-1_jj`` over the excitation window."""
    from . import identification as idn
    from .dataset_bag import _inertia_envelope_bounds

    pin, model, data = idn.build_model(asset)
    joints = asset.resolve_active_joints()
    bounds = _inertia_envelope_bounds(asset, config)
    if bounds is None:
        lower = np.asarray([-np.pi if j.lower is None else j.lower for j in joints], dtype=float)
        upper = np.asarray([np.pi if j.upper is None else j.upper for j in joints], dtype=float)
    else:
        lower = np.asarray([b[0] for b in bounds], dtype=float)
        upper = np.asarray([b[1] for b in bounds], dtype=float)
    rng = np.random.default_rng(seed)
    worst = np.zeros(len(joints))
    extra = np.zeros(len(joints)) if armature is None else np.asarray(armature, dtype=float)
    for q in lower + (upper - lower) * rng.random((n_samples, len(joints))):
        mass = np.asarray(pin.crba(model, data, q), dtype=float)
        mass = np.triu(mass) + np.triu(mass, 1).T
        worst = np.maximum(worst, np.diag(np.linalg.inv(mass + np.diag(extra))))
    return worst


def explicit_term_table(config: DatasetConfig, asset: Any, *, n_samples: int = 256,
                        seed: int = 0) -> list[BoundRow]:
    """Every explicitly applied velocity term's ``slope h / M_eff``, at the worst corner.

    Worst corner: the largest friction the prior can draw (``friction_scale``'s
    top, the link-friction factor's top on the ``per_robot`` tier), the
    smallest rotor inertia (``rotor_inertia_factor``'s bottom), the largest
    physics step the build can use (``max_time_step`` / ``rigid_time_step``
    rounded to divide the sample period) and the bare, payload-free arm (a
    payload only adds inertia).
    """
    names = tuple(asset.joint_names)
    n = len(names)
    period = config.control_period
    h_elastic = physics_step(float(config.max_time_step), period)
    h_rigid = physics_step(float(config.rigid_time_step), period)
    scale_high = float(config.friction_scale_range[1])
    rows: list[BoundRow] = []

    if config.motor_friction is not None:
        motor = config.motor_friction.nominal(asset)
        # The rigid reference keeps the nominal friction (`motor_friction_for`);
        # each elastic robot draws up to friction_scale's top.
        rigid_slope = motor.max_slope()
        motor_slope = rigid_slope * (scale_high if config.motor_friction.per_robot else 1.0)
    else:
        from .identification import FrictionModel

        # Rounds 4-5: the URDF's own friction, scaled only when friction is sampled.
        scale = scale_high if config.n_friction_samples > 1 else 1.0
        rigid_slope = motor_slope = FrictionModel.from_asset(asset).max_slope() * scale
    extras = config.plant_extras
    link_slope = np.zeros(n)
    if extras.has_link_friction:
        from .identification import COULOMB_EPSILON

        viscous = _per_joint(extras.link_friction_viscous or (0.0,), n, "link_friction.viscous")
        coulomb = _per_joint(extras.link_friction_coulomb or (0.0,), n, "link_friction.coulomb")
        if extras.link_friction_tier == "per_robot":
            viscous = viscous * extras.link_friction_viscous_factor[1]
            coulomb = coulomb * extras.link_friction_coulomb_factor[1]
        link_slope = viscous + coulomb / COULOMB_EPSILON
    link_integration = extras.link_friction_integration
    if "newton" in config.backends and link_integration == "implicit":
        # The implicit step is MuJoCo's; a Newton row would integrate it explicitly.
        link_integration = "explicit"

    nominal_rotor = config.nominal_rotor_inertia(n)
    if config.rigid_reference:
        armature = nominal_rotor if config.is_round6 else None
        rigid_inverse = _inverse_mass_envelope(asset, config, armature, n_samples, seed)
        rigid_inertia = 1.0 / rigid_inverse
        for j, name in enumerate(names):
            rows.append(BoundRow("rigid", name, "joint", "motor friction", float(rigid_slope[j]), h_rigid,
                                 float(rigid_inertia[j]), "explicit"))
            if extras.has_link_friction and config.is_round6:
                rows.append(BoundRow("rigid", name, "joint", "link friction", float(link_slope[j]), h_rigid,
                                     float(rigid_inertia[j]), link_integration))

    if config.transmission.robots:
        link_inverse = _inverse_mass_envelope(asset, config, None, n_samples, seed)
        sampling = config.transmission
        rotor_low = nominal_rotor * (sampling.rotor_inertia_factor[0] if sampling.rotor_inertia_nominal else 1.0)
        spring_damping = np.zeros(n)
        if extras.has_transmission_error:
            amplitude = _per_joint(extras.transmission_error_amplitude, n, "transmission_error.amplitude")
            order = _per_joint(extras.transmission_error_order, n, "transmission_error.order")
            if sampling.stiffness_nominal:
                k_high = np.asarray(sampling.stiffness_nominal, dtype=float) * sampling.stiffness_factor[1]
            else:
                k_high = np.asarray([hi for _, hi in sampling.stiffness], dtype=float)
            rotor_high = nominal_rotor * (sampling.rotor_inertia_factor[1] if sampling.rotor_inertia_nominal else 1.0)
            # d = 2 zeta sqrt(k J_eff) <= 2 zeta_max sqrt(k_max J_rotor,max).
            d_high = 2.0 * sampling.damping_ratio[1] * np.sqrt(k_high * rotor_high)
            spring_damping = d_high * (amplitude * order) ** 2
        for j, name in enumerate(names):
            rows.append(BoundRow("elastic", name, "rotor", "motor friction", float(motor_slope[j]), h_elastic,
                                 float(rotor_low[j]), "explicit"))
            if extras.has_transmission_error:
                rows.append(BoundRow("elastic", name, "rotor", "spring damping (explicit part)",
                                     float(spring_damping[j]), h_elastic, float(rotor_low[j]), "explicit"))
            if extras.has_link_friction:
                rows.append(BoundRow("elastic", name, "link", "link friction", float(link_slope[j]), h_elastic,
                                     float(1.0 / link_inverse[j]), link_integration))
    return rows


def dof_totals(rows: list[BoundRow]) -> dict[tuple[str, str, str], float]:
    """The bound per DOF: the enforced terms that share a DOF add up."""
    totals: dict[tuple[str, str, str], float] = {}
    for row in rows:
        if row.enforced:
            key = (row.tier, row.joint, row.dof)
            totals[key] = totals.get(key, 0.0) + row.value
    return totals


def require_explicit_term_bound(config: DatasetConfig, *, source: Any = "config", asset: Any = None) -> list[BoundRow]:
    """Raise when any DOF's explicit terms exceed the bound; return the table otherwise."""
    if asset is None:
        from pathlib import Path

        from .assets import AssetRegistry

        asset = AssetRegistry.for_repository(Path(__file__).resolve().parents[2]).load(config.asset)
    rows = explicit_term_table(config, asset)
    totals = dof_totals(rows)
    offenders = [(key, value) for key, value in totals.items() if value > 1.0]
    if offenders:
        detail = []
        for (tier, joint, dof), value in offenders:
            terms = ", ".join(f"{r.term} {r.value:.3g}" for r in rows
                              if (r.tier, r.joint, r.dof) == (tier, joint, dof) and r.enforced)
            detail.append(f"{tier} {joint} ({dof} DOF): (b + c/eps + d) h / M_eff = {value:.3g} > 1 [{terms}]")
        raise ValueError(
            f"{source}: explicit-integration bound violated (R6_00 Sec 3):\n  - " + "\n  - ".join(detail)
        )
    return rows


def bound_limited_steps(config: DatasetConfig, asset: Any, *,
                        ceiling: float | None = None) -> dict[str, dict[str, Any]]:
    """The largest physics step per tier that keeps every DOF's explicit terms ``<= 1``.

    Each row's value is linear in ``h``, so the step at which a DOF's enforced
    terms sum to exactly one is ``h / total``; the tier's step is the smallest
    of those, capped at ``ceiling`` (the schema default) and rounded down to
    divide the sample period (:func:`physics_step`).  This is how a config's
    ``auto`` step is derived (`R6_02` Sec 6: re-derive, don't hand-set).
    """
    from .dataset_config import AUTO_TIME_STEP_CEILING

    ceiling = AUTO_TIME_STEP_CEILING if ceiling is None else float(ceiling)
    period = config.control_period
    rows = explicit_term_table(config, asset)
    out: dict[str, dict[str, Any]] = {}
    for tier in ("elastic", "rigid"):
        rates: dict[tuple[str, str], float] = {}
        for row in rows:
            if row.tier == tier and row.enforced:
                key = (row.joint, row.dof)
                rates[key] = rates.get(key, 0.0) + row.slope / row.effective_inertia
        if not rates:
            continue
        (joint, dof), rate = max(rates.items(), key=lambda item: item[1])
        limit = 1.0 / rate if rate > 0.0 else float("inf")
        step = physics_step(min(limit, ceiling), period)
        out[tier] = {"time_step": step, "bound_step": limit, "binding_joint": joint, "binding_dof": dof,
                     "value": step * rate, "ceiling": ceiling}
    return out


def resolve_auto_time_steps(config: DatasetConfig, *, source: Any = "config", asset: Any = None) -> DatasetConfig:
    """Replace ``auto`` physics steps by :func:`bound_limited_steps`."""
    from dataclasses import replace

    if asset is None:
        from pathlib import Path

        from .assets import AssetRegistry

        asset = AssetRegistry.for_repository(Path(__file__).resolve().parents[2]).load(config.asset)
    steps = bound_limited_steps(config, asset)
    update: dict[str, float] = {}
    if "max_time_step" in config.auto_time_steps and "elastic" in steps:
        update["max_time_step"] = steps["elastic"]["time_step"]
    if "rigid_time_step" in config.auto_time_steps and "rigid" in steps:
        update["rigid_time_step"] = steps["rigid"]["time_step"]
    return replace(config, **update) if update else config


def format_table(rows: list[BoundRow]) -> str:
    lines = [f"{'tier':8s} {'joint':22s} {'dof':6s} {'term':32s} {'slope':>10s} {'h':>9s} {'M_eff':>10s} "
             f"{'value':>8s}  integration"]
    for row in rows:
        lines.append(f"{row.tier:8s} {row.joint:22s} {row.dof:6s} {row.term:32s} {row.slope:10.4g} "
                     f"{row.time_step:9.3g} {row.effective_inertia:10.4g} {row.value:8.3g}  {row.integration}"
                     + ("" if row.enforced else " (exempt)"))
    return "\n".join(lines)
