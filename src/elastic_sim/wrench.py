"""End-effector force/torque sensor model (round 5, `R5_03` T-1).

Two of the three round-5 platforms measure their link-side quantity with an
**end-effector wrench**, not a transmission torque:

* **FMRR** carries an ATI Axia90 at ``ft_link``, after the flange and the yaw
  joint — i.e. at the end effector. The legacy simulator mapped the *spring*
  force to that channel, which is not what the sensor measures: in free motion
  the cell sees only the bodies mounted past it, which on this platform is the
  handle (``R5_01`` Amendment 1 consequence 3, ``R5_04`` Sec 3).
* **UR10** has no link-side sensor at all; with an external cell at the flange
  it would be in the same situation (``R5_Q`` Q-5).
* **iiwa** is the exception: its joint torque sensors measure the link-side
  joint torque directly, so it keeps ``target: link_torque``.

So the three platforms genuinely have three different targets, and a dataset
has to say which one it carries. This module computes the wrench a simulated
cell would read and maps it to joint space.

**What the sensor measures.** The bodies rigidly mounted past the sensor frame
form one rigid body with spatial inertia ``I_s`` expressed at that frame. The
wrench transmitted across the sensor is then the Newton-Euler force that body
requires::

    f = I_s a + v x* (I_s v),     a including gravity

with ``v`` and ``a`` the sensor frame's own spatial motion. The sensor body's
*own* inertia is deliberately excluded: a cell measures what is beyond its
measurement plane, not its own mass.

**Why the inertia is read from the URDF and not from the Pinocchio model.**
Everything past ``ft_link`` on FMRR is attached through fixed joints (the yaw
joint is locked, ``R5_01`` Sec 1.2), and Pinocchio merges a fixed-joint subtree
into its parent body — so ``model.inertias`` no longer knows which part of that
body sits past the sensor, and ``computeSupportedForceByFrame`` has no subtree
to sum. The link inertials are therefore parsed from the URDF, where the
distinction still exists, and transformed into the sensor frame with the frame
placements Pinocchio still provides.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np

from .assets import AssetSpec, expand_simple_xacro_text


@dataclass(frozen=True)
class LinkInertial:
    """One URDF ``<link><inertial>``, in that link's own frame."""

    name: str
    mass: float
    com: tuple[float, float, float]
    inertia: tuple[float, float, float, float, float, float]  # ixx, iyy, izz, ixy, ixz, iyz

    def as_pinocchio(self, pin: Any) -> Any:
        ixx, iyy, izz, ixy, ixz, iyz = self.inertia
        tensor = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]], dtype=float)
        return pin.Inertia(float(self.mass), np.asarray(self.com, dtype=float), tensor)


def _text_of(asset: AssetSpec) -> str:
    text = Path(asset.urdf_path).read_text(encoding="utf-8")
    if "${" in text or "<?xacro" in text:
        text = expand_simple_xacro_text(text)
    return text


def read_link_inertials(asset: AssetSpec) -> dict[str, LinkInertial]:
    """Parse every link's own inertial from the asset's URDF.

    Links with no ``<inertial>`` are omitted rather than defaulted to zero, so a
    caller can tell "massless by declaration" from "not declared".
    """
    root = ET.fromstring(_text_of(asset))
    inertials: dict[str, LinkInertial] = {}
    for link in root.findall("link"):
        name = link.get("name")
        inertial = link.find("inertial")
        if not name or inertial is None:
            continue
        mass_element = inertial.find("mass")
        tensor = inertial.find("inertia")
        if mass_element is None:
            continue
        origin = inertial.find("origin")
        com = (0.0, 0.0, 0.0)
        if origin is not None and origin.get("xyz"):
            values = [float(v) for v in origin.get("xyz").replace(",", " ").split()]
            com = (values[0], values[1], values[2])
        if origin is not None and origin.get("rpy"):
            rpy = [float(v) for v in origin.get("rpy").replace(",", " ").split()]
            if any(abs(value) > 1e-12 for value in rpy):
                raise ValueError(
                    f"link {name!r} declares an inertial <origin rpy>; rotate the tensor into the "
                    "link frame first (scene.normalize_inertial_frames does this for the simulators)"
                )

        def _value(attribute: str) -> float:
            return 0.0 if tensor is None else float(tensor.get(attribute, 0.0))

        inertials[name] = LinkInertial(
            name=name, mass=float(mass_element.get("value", 0.0)), com=com,
            inertia=(_value("ixx"), _value("iyy"), _value("izz"),
                     _value("ixy"), _value("ixz"), _value("iyz")),
        )
    return inertials


def links_beyond(asset: AssetSpec, sensor_link: str) -> tuple[str, ...]:
    """Every link mounted past ``sensor_link``, in declaration order.

    ``sensor_link`` itself is **not** included: a force/torque cell measures
    what is beyond its measurement plane, and its own housing is on the
    mounting side. Descent follows every joint type — a cell mounted before an
    articulated tool measures the tool whatever its joints do — so a caller
    that meant only the rigid part should say so by locking those joints, which
    is what the asset layer already does for non-active 1-DoF joints.
    """
    root = ET.fromstring(_text_of(asset))
    children: dict[str, list[str]] = {}
    for joint in root.findall("joint"):
        parent, child = joint.find("parent"), joint.find("child")
        if parent is None or child is None:
            continue
        children.setdefault(parent.get("link", ""), []).append(child.get("link", ""))
    if sensor_link not in {link.get("name") for link in root.findall("link")}:
        available = ", ".join(sorted(str(link.get("name")) for link in root.findall("link")))
        raise ValueError(f"Asset {asset.name!r} has no link {sensor_link!r}; available: {available}")
    beyond: list[str] = []
    stack = list(children.get(sensor_link, []))
    while stack:
        name = stack.pop(0)
        if name in beyond:
            continue
        beyond.append(name)
        stack.extend(children.get(name, []))
    return tuple(beyond)


@dataclass(frozen=True)
class SensorSpec:
    """Where an asset's force/torque cell is and what it carries."""

    frame: str
    #: Links whose mass the cell measures; derived from ``frame`` when empty.
    beyond: tuple[str, ...] = ()

    @classmethod
    def from_asset(cls, asset: AssetSpec) -> "SensorSpec | None":
        """Read ``force_torque_sensor: {frame: ...}`` from the asset metadata."""
        raw = asset.metadata.get("force_torque_sensor")
        if not raw:
            return None
        if not isinstance(raw, dict) or not raw.get("frame"):
            raise ValueError(
                f"Asset {asset.name!r}: force_torque_sensor must be a mapping with a `frame` key"
            )
        return cls(frame=str(raw["frame"]), beyond=tuple(str(v) for v in raw.get("beyond", []) or []))


class ForceTorqueSensor:
    """A simulated force/torque cell at one frame of an asset.

    Built once per asset and reused over a rollout: the sensor-frame inertia of
    the bodies beyond it is constant (they are one rigid body), so only the
    frame's own motion changes per sample.
    """

    def __init__(self, asset: AssetSpec, spec: SensorSpec) -> None:
        from . import identification as idn

        self.asset = asset
        self.spec = spec
        self._pin, self._model, self._data = idn.build_model(asset)
        pin = self._pin
        if not self._model.existFrame(spec.frame):
            raise ValueError(f"Asset {asset.name!r} has no Pinocchio frame {spec.frame!r}")
        self.frame_id = self._model.getFrameId(spec.frame)
        self.beyond = spec.beyond or links_beyond(asset, spec.frame)
        if not self.beyond:
            raise ValueError(
                f"Asset {asset.name!r}: nothing is mounted past {spec.frame!r}, so a sensor there "
                "would read zero in free motion; check the frame name"
            )
        inertials = read_link_inertials(asset)
        missing = [name for name in self.beyond if name not in inertials]
        if missing:
            raise ValueError(f"Asset {asset.name!r}: links past the sensor without an <inertial>: {missing}")

        # One rigid body, expressed at the sensor frame.  Frame placements are
        # configuration-independent *between* rigidly-connected links, so the
        # zero configuration is as good as any for this sum -- but assert it
        # rather than assume it, since an articulated tool would break it.
        zero = np.zeros(self._model.nv)
        pin.forwardKinematics(self._model, self._data, zero)
        pin.updateFramePlacements(self._model, self._data)
        sensor_placement = self._data.oMf[self.frame_id]
        total = pin.Inertia.Zero()
        for name in self.beyond:
            body = self._data.oMf[self._model.getFrameId(name)]
            total = total + (sensor_placement.inverse() * body).act(inertials[name].as_pinocchio(pin))
        self.inertia = total
        self.mass = float(total.mass)
        self.gravity = pin.Motion(np.asarray(asset.gravity, dtype=float), np.zeros(3))

    def describe(self) -> dict[str, Any]:
        return {
            "frame": self.spec.frame,
            "measures": list(self.beyond),
            "mass_kg": self.mass,
            "com_in_sensor_frame_m": [float(v) for v in self.inertia.lever],
        }

    def wrench(self, q: np.ndarray, dq: np.ndarray, ddq: np.ndarray) -> np.ndarray:
        """The 6-vector ``[fx, fy, fz, tx, ty, tz]`` read at the sensor frame.

        Expressed in the sensor frame, gravity included, so a robot at rest
        reads the static weight of whatever is mounted past the cell. Sign
        convention: the wrench the bodies beyond the sensor require, i.e. what
        the mounting side must transmit through the cell. A real cell's own
        sign and axis convention is a calibration matter and belongs in the
        recorder, not here.
        """
        pin = self._pin
        pin.forwardKinematics(self._model, self._data,
                              np.asarray(q, dtype=float), np.asarray(dq, dtype=float),
                              np.asarray(ddq, dtype=float))
        pin.updateFramePlacements(self._model, self._data)
        placement = self._data.oMf[self.frame_id]
        velocity = pin.getFrameVelocity(self._model, self._data, self.frame_id, pin.ReferenceFrame.LOCAL)
        acceleration = pin.getFrameAcceleration(self._model, self._data, self.frame_id, pin.ReferenceFrame.LOCAL)
        # Gravity enters as an acceleration of the frame, rotated into it.
        total = acceleration - placement.actInv(self.gravity)
        force = self.inertia * total + velocity.cross(self.inertia * velocity)
        return np.asarray(force.vector, dtype=float)

    def joint_torque(self, q: np.ndarray, wrench: np.ndarray) -> np.ndarray:
        """Map a sensor-frame wrench to joint space: ``tau = J(q)^T w``.

        ``q`` is the *selected position side* (`R5_03` T-1): on a real robot the
        only configuration available for this mapping is the one the encoders
        report, and on the motor side that differs from the link side by the
        deflection. The resulting error belongs in the error budget, not in the
        target, which is why the target is computed from the true link-side
        motion and only the mapping uses this ``q``.
        """
        pin = self._pin
        jacobian = pin.computeFrameJacobian(
            self._model, self._data, np.asarray(q, dtype=float), self.frame_id,
            pin.ReferenceFrame.LOCAL,
        )
        return np.asarray(jacobian.T @ np.asarray(wrench, dtype=float), dtype=float)

    def rollout(
        self, q_link: np.ndarray, dq_link: np.ndarray, ddq_link: np.ndarray,
        *, q_mapping: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """``(wrench, joint_torque)`` over a whole rollout, one row per sample.

        The wrench is computed from the *link-side* motion, which is what the
        cell physically experiences; ``q_mapping`` (defaulting to the link side)
        is the configuration the Jacobian is evaluated at.
        """
        q_link = np.atleast_2d(np.asarray(q_link, dtype=float))
        dq_link = np.atleast_2d(np.asarray(dq_link, dtype=float))
        ddq_link = np.atleast_2d(np.asarray(ddq_link, dtype=float))
        mapping = q_link if q_mapping is None else np.atleast_2d(np.asarray(q_mapping, dtype=float))
        if not (len(q_link) == len(dq_link) == len(ddq_link) == len(mapping)):
            raise ValueError("q_link, dq_link, ddq_link and q_mapping must have the same length")
        wrenches = np.empty((len(q_link), 6))
        torques = np.empty((len(q_link), q_link.shape[1]))
        for index in range(len(q_link)):
            wrenches[index] = self.wrench(q_link[index], dq_link[index], ddq_link[index])
            torques[index] = self.joint_torque(mapping[index], wrenches[index])
        return wrenches, torques


#: Column names of the raw wrench, written beside the mapped target.
WRENCH_COLUMNS = ("w_fx", "w_fy", "w_fz", "w_tx", "w_ty", "w_tz")
