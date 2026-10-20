"""Plant parameters from theta, applied in memory to Pinocchio and MuJoCo (S2R_01 T-2.2, T-2.5).

One :class:`PlantParameters` feeds both engines (S2R_01 P-2): the inertials
go into Pinocchio's ``model.inertias`` and into MuJoCo's ``body_mass``,
``body_ipos``, ``body_inertia`` and ``body_iquat`` through a
:class:`BodyMap` (URDF link name -> body id), which works on the rigid build
and on the elastic build with its ``elastic_link__*`` intermediate bodies.
Nothing is written to ``assets/``: no temp URDF, no rebuild per evaluation.

**Inertial parameterisation** (S2R_01 §5): per link, ``mass``, the COM in
the link frame ``com_{x,y,z}`` and ``inertia_scale``, a uniform scale of the
URDF's rotational inertia about the COM. The tensor is independent of the
mass (a heavier link with the URDF tensor is still physical: the triangle
inequalities don't involve the mass). A uniform scale keeps the principal
axes, so MuJoCo's ``body_iquat`` is kept and only ``body_inertia`` scales;
θ₀ therefore reproduces the untouched asset bit for bit in both engines. A
full tensor (future parameters) takes the eigen-decomposition path.

**Physical consistency** [S-37]: :meth:`PlantParameters.check_physical`.
"""

from __future__ import annotations

import dataclasses
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..identification import COULOMB_EPSILON, FrictionModel
from ..plant_extras import PlantExtras, StiffnessNonlinearity, TransmissionError
from .config import REPO_ROOT

#: S2R_04 A-8: link Coulomb friction is smoothed at the production plant's
#: fixed epsilon (``identification.COULOMB_EPSILON`` = 1e-3, what
#: ``PlantExtras.epsilon`` uses), so a calibrated coefficient means the same
#: law in the simulation stack. Still 20x the standstill sigma(dq) of 5e-5
#: rad/s (FR_05 §6).
LINK_FRICTION_EPSILON = COULOMB_EPSILON
#: S2R_01 §4.2 / legacy: the finite loss a non-physical theta gets.
NONPHYSICAL_LOSS = 1.0e12
#: Tolerance of the physical-consistency checks, relative to the tensor's scale.
_PHYSICAL_RTOL = 1.0e-12


@dataclass(frozen=True)
class LinkInertial:
    """A link's inertial in its own (= Pinocchio joint / MuJoCo body) frame.

    ``inertia`` is the 3x3 rotational inertia about the COM. ``urdf_scale``,
    when set, says the tensor is exactly ``urdf_scale`` times the URDF's
    (principal axes unchanged), which the MuJoCo setter uses to keep
    ``body_iquat`` bit-exact.
    """

    mass: float
    com: np.ndarray
    inertia: np.ndarray
    urdf_scale: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "mass", float(self.mass))
        object.__setattr__(self, "com", np.asarray(self.com, dtype=float).reshape(3))
        object.__setattr__(self, "inertia", np.asarray(self.inertia, dtype=float).reshape(3, 3))

    def inertia_about_origin(self) -> np.ndarray:
        c = self.com
        return self.inertia + self.mass * (float(c @ c) * np.eye(3) - np.outer(c, c))

    def pseudo_inertia(self) -> np.ndarray:
        """4x4 pseudo-inertia J = [[0.5 tr(I_o) 1 - I_o, m c], [m c^T, m]] [S-37]."""
        inertia_o = self.inertia_about_origin()
        sigma = 0.5 * np.trace(inertia_o) * np.eye(3) - inertia_o
        h = self.mass * self.com
        return np.block([[sigma, h[:, None]], [h[None, :], np.array([[self.mass]])]])

    def check_physical(self, label: str) -> list[str]:
        problems = []
        if not (math.isfinite(self.mass) and self.mass > 0.0):
            problems.append(f"{label}: mass {self.mass!r} is not positive")
        if not np.all(np.isfinite(self.inertia)) or not np.allclose(self.inertia, self.inertia.T, rtol=0, atol=1e-15):
            problems.append(f"{label}: inertia tensor is not a finite symmetric matrix")
            return problems
        moments = np.linalg.eigvalsh(self.inertia)
        tol = _PHYSICAL_RTOL * max(1e-300, float(np.max(np.abs(moments))))
        if np.any(moments <= tol):
            problems.append(f"{label}: inertia tensor is not positive definite (principal moments {moments.tolist()})")
        a, b, c = np.sort(moments)
        if not a + b > c + tol:
            problems.append(f"{label}: principal moments {[a, b, c]} violate the triangle inequality")
        if self.mass > 0.0 and np.min(np.linalg.eigvalsh(self.pseudo_inertia())) <= tol:
            problems.append(f"{label}: pseudo-inertia matrix is not positive definite")
        return problems


@dataclass(frozen=True)
class TransmissionParameters:
    """K, zeta and the reflected rotor inertia per joint. The damping needs a
    link inertia (``d = 2 zeta sqrt(k J_eff)``), so the replay engines build
    the :class:`TransmissionSpec` with :meth:`spec`."""

    stiffness: np.ndarray
    damping_ratio: np.ndarray
    rotor_inertia: np.ndarray

    def spec(self, link_inertia: np.ndarray, **kwargs: Any):
        from ..torque_runners import TransmissionSpec

        return TransmissionSpec.from_damping_ratio(self.stiffness, self.damping_ratio, self.rotor_inertia,
                                                   link_inertia, **kwargs)


@dataclass(frozen=True)
class SensorCalibration:
    """Joint torque sensor: ft = gain * tau_s + offset[run] per joint."""

    gain: np.ndarray
    offsets: Mapping[str, np.ndarray]

    def offset(self, run_id: str) -> np.ndarray:
        if run_id not in self.offsets:
            return np.zeros_like(self.gain)
        return self.offsets[run_id]


@dataclass(frozen=True)
class ControllerEmulation:
    """KUKA position-loop emulation parameters (R2, S2R_01 §4.3)."""

    bandwidth: np.ndarray
    damping_ratio: np.ndarray
    model_scale: float
    delay_cycles: int
    drive_rate: float


# ---------------------------------------------------------------------------
# Plant base: the nominal values theta is measured against
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlantBase:
    """Nominal plant: the asset's URDF inertials and the round-6 priors
    (``plant_base``) for friction, transmission and extras."""

    asset: Any
    joint_names: tuple[str, ...]
    link_names: tuple[str, ...]          # child link of each active joint
    inertials: Mapping[str, LinkInertial]
    link_friction: FrictionModel
    motor_friction: FrictionModel
    stiffness: np.ndarray
    rotor_inertia: np.ndarray
    extras_defaults: Mapping[str, float]
    plant_base_source: str
    extras_template: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: Any) -> "PlantBase":
        from ..assets import AssetRegistry

        return cls.load(AssetRegistry.for_repository().load(config.asset), config.plant_base)

    @classmethod
    def load(cls, asset: Any, plant_base: str | Path | None) -> "PlantBase":
        from .. import identification as idn

        pin, model, _ = idn.build_model(asset)
        joints = asset.resolve_active_joints()
        links = tuple(j.child for j in joints)
        _refuse_composite_bodies(asset, links)
        inertials = {}
        for index, link in enumerate(links, start=1):
            inertia = model.inertias[index]
            inertials[link] = LinkInertial(inertia.mass, np.array(inertia.lever), np.array(inertia.inertia),
                                           urdf_scale=1.0)
        n = len(links)
        if plant_base is None:
            zeros = np.zeros(n)
            return cls(asset, tuple(asset.joint_names), links, inertials, FrictionModel(zeros, zeros,
                       LINK_FRICTION_EPSILON), FrictionModel.from_asset(asset), np.full(n, np.inf), np.ones(n), {},
                       "none")
        from ..dataset_config import _per_joint, load_config

        base = load_config(plant_base, check_bounds=False)
        source = f"round-6 {Path(plant_base).name}"
        extras = base.plant_extras
        viscous = _per_joint(extras.link_friction_viscous or (0.0,), n, "link_friction.viscous")
        coulomb = _per_joint(extras.link_friction_coulomb or (0.0,), n, "link_friction.coulomb")
        motor = (base.motor_friction.nominal(asset) if base.motor_friction is not None
                 else FrictionModel.from_asset(asset))
        transmission = base.transmission
        stiffness = _per_joint(transmission.stiffness_nominal, n, "stiffness_nominal")
        rotor = _per_joint(transmission.rotor_inertia_nominal or transmission.rotor_inertia, n, "rotor_inertia")
        defaults: dict[str, float] = {}
        template: dict[str, Any] = {}
        if extras.has_transmission_error:
            defaults["transmission_error.amplitude"] = float(np.max(extras.transmission_error_amplitude))
            template["transmission_error_order"] = tuple(_per_joint(extras.transmission_error_order, n, "order"))
        if extras.has_nonlinear_spring and len(extras.stiffness_factors) == 3 \
                and extras.stiffness_breakpoints_effort_fraction:
            factors = [float(v) for v in extras.stiffness_factors]
            defaults["stiffness_nonlinearity.factor_low"] = factors[0]
            defaults["stiffness_nonlinearity.factor_high"] = factors[2]
            template["stiffness_factor_mid"] = factors[1]
            template["stiffness_knees_effort_fraction"] = tuple(extras.stiffness_breakpoints_effort_fraction)
        return cls(asset, tuple(asset.joint_names), links, inertials,
                   FrictionModel(viscous, coulomb, LINK_FRICTION_EPSILON), motor, stiffness, rotor, defaults, source,
                   template)


def _refuse_composite_bodies(asset: Any, links: Sequence[str]) -> None:
    """A link fixed-jointed to an active joint's child, with mass, would be
    merged into that body by both engines; the setters don't handle that
    yet (none of the iiwa's links is one)."""
    root = ET.parse(asset.urdf_path).getroot()
    masses = {}
    for link in root.findall("link"):
        mass = link.find("inertial/mass")
        masses[link.get("name")] = 0.0 if mass is None else float(mass.get("value", "0"))
    children: dict[str, list[str]] = {}
    for joint in root.findall("joint"):
        if joint.get("type") == "fixed":
            children.setdefault(joint.find("parent").get("link"), []).append(joint.find("child").get("link"))
    for link in links:
        stack = list(children.get(link, []))
        while stack:
            child = stack.pop()
            if masses.get(child, 0.0) > 0.0:
                raise NotImplementedError(f"{asset.name}: link {child!r} (mass {masses[child]}) is fixed to "
                                          f"{link!r}; composite bodies aren't supported by the sim2real setters yet")
            stack.extend(children.get(child, []))


# ---------------------------------------------------------------------------
# Plant parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlantParameters:
    base: PlantBase
    inertials: Mapping[str, LinkInertial]
    link_friction: FrictionModel
    motor_friction: FrictionModel
    transmission: TransmissionParameters
    extras: PlantExtras
    sensor: SensorCalibration
    controller: ControllerEmulation | None = None
    theta: Mapping[str, float] = field(default_factory=dict)

    @classmethod
    def from_theta(cls, base: PlantBase, theta: Mapping[str, float], *, runs: Sequence[str] = ()) -> "PlantParameters":
        """Build the plant; a name missing from ``theta`` keeps the base value."""
        known = ("inertial.", "friction.", "sensor.", "transmission.", "extras.", "controller.")
        unknown = sorted(n for n in theta if not n.startswith(known))
        if unknown:
            raise KeyError(f"unknown parameter groups: {unknown}")
        get = theta.get
        inertials = {}
        for link, nominal in base.inertials.items():
            scale = float(get(f"inertial.{link}.inertia_scale", 1.0))
            com = np.array([get(f"inertial.{link}.com_{a}", nominal.com[i]) for i, a in enumerate("xyz")], dtype=float)
            urdf_scale = None if nominal.urdf_scale is None else scale * nominal.urdf_scale
            inertials[link] = LinkInertial(get(f"inertial.{link}.mass", nominal.mass), com, scale * nominal.inertia,
                                           urdf_scale=urdf_scale)
        joints = base.joint_names

        def per_joint(template: str, default: np.ndarray | float) -> np.ndarray:
            default = np.broadcast_to(np.asarray(default, dtype=float), (len(joints),))
            return np.asarray([get(template.format(j), default[i]) for i, j in enumerate(joints)], dtype=float)

        link_friction = FrictionModel(per_joint("friction.link.{}.viscous", base.link_friction.viscous),
                                      per_joint("friction.link.{}.coulomb", base.link_friction.coulomb),
                                      base.link_friction.epsilon)
        motor_friction = FrictionModel(per_joint("friction.motor.{}.viscous", base.motor_friction.viscous),
                                       per_joint("friction.motor.{}.coulomb", base.motor_friction.coulomb),
                                       base.motor_friction.epsilon)
        transmission = TransmissionParameters(per_joint("transmission.{}.stiffness", base.stiffness),
                                              per_joint("transmission.{}.damping_ratio", 0.1),
                                              per_joint("transmission.{}.rotor_inertia", base.rotor_inertia))
        run_ids = set(runs) | {n.split("@", 1)[1] for n in theta if n.startswith("sensor.ft.") and "@" in n}
        sensor = SensorCalibration(per_joint("sensor.ft.{}.gain", 1.0),
                                   {run: per_joint("sensor.ft.{}.offset@" + run, 0.0) for run in sorted(run_ids)})
        controller = None
        if any(n.startswith("controller.") for n in theta):
            controller = ControllerEmulation(per_joint("controller.{}.bandwidth", 2.0 * math.pi * 10.0),
                                             per_joint("controller.{}.damping_ratio", 1.0),
                                             float(get("controller.model_scale", 1.0)),
                                             int(round(get("controller.delay_cycles", 1))),
                                             float(get("controller.drive_rate", 4000.0)))
        extras = _extras(base, transmission.stiffness, link_friction, get)
        return cls(base, inertials, link_friction, motor_friction, transmission, extras, sensor, controller,
                   dict(theta))

    @classmethod
    def nominal(cls, base: PlantBase) -> "PlantParameters":
        return cls.from_theta(base, {})

    # -- physical consistency (T-2.5) ---------------------------------------------

    def check_physical(self) -> list[str]:
        """Empty when every link is physically consistent [S-37]: mass > 0,
        a positive-definite tensor satisfying the triangle inequalities, a
        positive-definite pseudo-inertia."""
        problems = []
        for link, inertial in self.inertials.items():
            problems.extend(inertial.check_physical(link))
        return problems

    def physical_failure(self) -> dict[str, Any] | None:
        """``None``, or the record the problem layer logs for a non-physical
        theta: a large finite loss and the reasons (S2R_01 §5)."""
        problems = self.check_physical()
        if not problems:
            return None
        return {"loss": NONPHYSICAL_LOSS, "error": "non-physical inertial parameters", "reasons": problems}

    # -- engines ---------------------------------------------------------------------

    def apply_to_pinocchio(self, pin: Any, model: Any) -> None:
        """Write the inertials into ``model.inertias`` in place (joint j's body
        is link j's inertial: no composite bodies, see PlantBase)."""
        for index, link in enumerate(self.base.link_names, start=1):
            inertial = self.inertials[link]
            model.inertias[index] = pin.Inertia(inertial.mass, inertial.com, inertial.inertia)

    def apply_to_mujoco(self, mujoco: Any, model: Any, body_map: "BodyMap", data: Any = None) -> None:
        """Write the inertials into the MuJoCo model in place; with ``data``,
        refresh the constants (``mj_setConst``)."""
        for link in self.base.link_names:
            body = body_map.body_id[link]
            inertial = self.inertials[link]
            model.body_mass[body] = inertial.mass
            # As a change on MuJoCo's own COM: its importer stores the URDF
            # origin within an ulp of its frame arithmetic, so theta0 stays
            # bit-identical to the untouched model.
            model.body_ipos[body] = body_map.ipos0[link] + (inertial.com - self.base.inertials[link].com)
            if inertial.urdf_scale is not None:
                model.body_inertia[body] = inertial.urdf_scale * body_map.inertia0[link]
                model.body_iquat[body] = body_map.iquat0[link]
            else:
                moments, quat = principal_inertia(mujoco, inertial.inertia)
                model.body_inertia[body] = moments
                model.body_iquat[body] = quat
        if data is not None:
            mujoco.mj_setConst(model, data)

    # -- export (optional tool, A-3) ------------------------------------------------

    def export_urdf(self, target: str | Path) -> Path:
        """Write a copy of the asset URDF with these inertials to ``target``
        (a file, or a directory that gets the asset's file name). Refuses
        any target under ``assets/``; writes nothing else.

        Each inertial is written as ``rpy="0 0 0"`` with the full tensor in
        the link frame. That is exact for the iiwa (no inertial rpy). For an
        asset whose URDF has inertial rpy (the UR10, Phase 7) it relies on
        ``scene.normalize_inertial_frames`` having been applied to the asset,
        which round 4 did (S2R_04 R-5)."""
        source = Path(self.base.asset.urdf_path)
        target = Path(target).expanduser().resolve()
        if target.is_dir() or target.suffix != ".urdf":
            target = target / source.name
        if target.is_relative_to((REPO_ROOT / "assets").resolve()):
            raise ValueError(f"refusing to write {target}: export_urdf never writes under assets/ (S2R_00 §8)")
        tree = ET.parse(source)
        links = {link.get("name"): link for link in tree.getroot().findall("link")}
        for name, inertial in self.inertials.items():
            node = links[name].find("inertial")
            origin = node.find("origin")
            if origin is None:
                origin = ET.SubElement(node, "origin")
            origin.set("xyz", " ".join(repr(float(v)) for v in inertial.com))
            origin.set("rpy", "0 0 0")
            node.find("mass").set("value", repr(float(inertial.mass)))
            tensor = node.find("inertia")
            for key, (r, c) in {"ixx": (0, 0), "ixy": (0, 1), "ixz": (0, 2), "iyy": (1, 1), "iyz": (1, 2),
                                "izz": (2, 2)}.items():
                tensor.set(key, repr(float(inertial.inertia[r, c])))
        target.parent.mkdir(parents=True, exist_ok=True)
        tree.write(target, encoding="utf-8", xml_declaration=True)
        return target


def _extras(base: PlantBase, stiffness: np.ndarray, link_friction: FrictionModel, get) -> PlantExtras:
    """Round-6 extras with the link friction from theta (epsilon 1e-3, A-8); the
    transmission error's phase is per bag and left at zero here (the R1/R2
    engines draw or fix it)."""
    n = len(base.joint_names)
    nonlinearity = None
    template = base.extras_template
    if "stiffness_knees_effort_fraction" in template:
        effort = np.asarray([j.effort for j in base.asset.resolve_active_joints()], dtype=float)
        knees = np.outer(effort, np.asarray(template["stiffness_knees_effort_fraction"], dtype=float))
        factors = np.asarray([get("extras.stiffness_nonlinearity.factor_low",
                                  base.extras_defaults["stiffness_nonlinearity.factor_low"]),
                              template["stiffness_factor_mid"],
                              get("extras.stiffness_nonlinearity.factor_high",
                                  base.extras_defaults["stiffness_nonlinearity.factor_high"])], dtype=float)
        if np.all(np.isfinite(stiffness)):
            nonlinearity = StiffnessNonlinearity.from_torque_knees(factors, knees, stiffness)
    transmission_error = None
    if "transmission_error_order" in template:
        amplitude = float(get("extras.transmission_error.amplitude",
                              base.extras_defaults["transmission_error.amplitude"]))
        transmission_error = TransmissionError(amplitude=(amplitude,) * n, order=template["transmission_error_order"],
                                               phase=(0.0,) * n)
    return PlantExtras(link_friction=link_friction, stiffness_nonlinearity=nonlinearity,
                       transmission_error=transmission_error, epsilon=link_friction.epsilon)


# ---------------------------------------------------------------------------
# MuJoCo body map, intermediate bodies
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BodyMap:
    """URDF link name -> MuJoCo body id, plus each body's untouched COM,
    principal inertia and orientation (captured at build time, before any
    override)."""

    body_id: Mapping[str, int]
    ipos0: Mapping[str, np.ndarray]
    inertia0: Mapping[str, np.ndarray]
    iquat0: Mapping[str, np.ndarray]
    intermediate: Mapping[str, int]  # joint -> elastic_link__<joint> body id (elastic build)

    @classmethod
    def build(cls, mujoco: Any, model: Any, base: PlantBase) -> "BodyMap":
        body_id, ipos0, inertia0, iquat0, intermediate = {}, {}, {}, {}, {}
        for joint, link in zip(base.joint_names, base.link_names):
            index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, link)
            if index < 0:
                raise ValueError(f"MuJoCo model has no body {link!r} (fused by the URDF import?)")
            body_id[link] = int(index)
            ipos0[link] = np.array(model.body_ipos[index])
            inertia0[link] = np.array(model.body_inertia[index])
            iquat0[link] = np.array(model.body_iquat[index])
            mid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"elastic_link__{joint}")
            if mid >= 0:
                intermediate[joint] = int(mid)
        return cls(body_id, ipos0, inertia0, iquat0, intermediate)


def principal_inertia(mujoco: Any, inertia: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(body_inertia, body_iquat)`` of a full tensor about the COM: its
    eigenvalues and the rotation of its eigenvectors (det +1) as a quaternion."""
    moments, axes = np.linalg.eigh(np.asarray(inertia, dtype=float))
    if np.linalg.det(axes) < 0.0:
        axes[:, 2] *= -1.0
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.ascontiguousarray(axes).reshape(-1))
    return moments, quat


@dataclass(frozen=True)
class IntermediateBodies:
    """The elastic build's ``elastic_link__<joint>`` bodies as Pinocchio
    inertias in each motor joint's frame (pinocchio joint index ->
    ``pin.Inertia``). Each rotates with its motor joint at the joint origin,
    so at zero deflection the elastic chain is the rigid chain with these
    added to the links (S2R_04 A-9: R0 always includes them, so that R0 and
    R1 calibrate the same plant)."""

    inertias: Mapping[int, Any]

    @classmethod
    def from_mujoco(cls, pin: Any, model: Any, mujoco: Any, mujoco_model: Any, body_map: BodyMap
                    ) -> "IntermediateBodies":
        inertias = {}
        for index, joint in enumerate(model.names[1:], start=1):
            if joint not in body_map.intermediate:
                continue
            body = body_map.intermediate[joint]
            moments = np.array(mujoco_model.body_inertia[body])
            rotation = np.zeros(9)
            mujoco.mju_quat2Mat(rotation, np.array(mujoco_model.body_iquat[body]))
            rotation = rotation.reshape(3, 3)
            tensor = rotation @ np.diag(moments) @ rotation.T
            inertias[index] = pin.Inertia(float(mujoco_model.body_mass[body]), np.array(mujoco_model.body_ipos[body]),
                                          tensor)
        return cls(inertias)

    @classmethod
    def from_elastic_build(cls, base: PlantBase) -> "IntermediateBodies":
        """Build the elastic MuJoCo model once (default intermediate mass) and
        read the bodies off it. Their inertia doesn't depend on K."""
        import mujoco

        from .. import identification as idn
        from ..generic_mujoco_runner import _build_model
        from ..torque_runners import TransmissionSpec

        names = tuple(base.joint_names)
        transmission = TransmissionSpec.uniform(len(names), 1.0e4)
        mj_model, _ = _build_model(base.asset, mujoco, 1.0e-4, elastic_transmissions=transmission.as_mapping(names))
        pin, model, _ = idn.build_model(base.asset)
        return cls.from_mujoco(pin, model, mujoco, mj_model, BodyMap.build(mujoco, mj_model, base))

    def add_to(self, model: Any) -> None:
        """``model.inertias[i] += body`` in place."""
        for index, extra in self.inertias.items():
            model.inertias[index] = model.inertias[index] + extra

    def dynamic_parameters(self, n_links: int) -> np.ndarray:
        """The standard-parameter vector (10 per link) the bodies add."""
        out = np.zeros(10 * n_links)
        for index, extra in self.inertias.items():
            out[10 * (index - 1):10 * index] = extra.toDynamicParameters()
        return out


def add_intermediate_bodies(pin: Any, model: Any, mujoco: Any, mujoco_model: Any, body_map: BodyMap) -> None:
    """Add the elastic build's ``elastic_link__<joint>`` bodies to a
    Pinocchio model, so it is the elastic chain at zero deflection: each
    rotates with its motor joint, at the joint origin, in the joint frame.
    For G-5 and any rigid-vs-elastic comparison."""
    IntermediateBodies.from_mujoco(pin, model, mujoco, mujoco_model, body_map).add_to(model)


__all__ = ["BodyMap", "ControllerEmulation", "IntermediateBodies", "LINK_FRICTION_EPSILON", "LinkInertial", "NONPHYSICAL_LOSS", "PlantBase",
           "PlantParameters", "SensorCalibration", "TransmissionParameters", "add_intermediate_bodies",
           "principal_inertia"]
