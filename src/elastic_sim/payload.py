"""Rigid tool attached to the flange, injected into the URDF.

A payload must be visible to *every* consumer of an asset at once: the
Pinocchio model behind the computed-torque controller, both simulator
importers, and the collision checker.  ``generic_mujoco_runner``'s
``body_overrides`` reaches only the simulator, so a payload applied that way
would silently de-tune the controller.  Injecting a link into the URDF text
and handing back an ``AssetSpec`` that points at it keeps them in step by
construction.

See ``REFACTOR_SPECS/round3/R3_01_TARGET_OBSERVABILITY_AND_PAYLOAD.md``.
"""

from __future__ import annotations

import contextlib
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

from .assets import AssetSpec


@dataclass(frozen=True)
class Payload:
    """A uniform box rigidly attached to the last link.

    ``offset`` is in the frame of the last *active* joint's child link (see
    ``_last_link_name``) -- not necessarily a URDF's dedicated flange/ee
    frame, if it has one, since that frame is reached through a fixed joint
    this module deliberately does not walk (not every asset has one). If a
    caller's URDF has a fixed end-effector joint offset from that link (the
    KUKA iiwa assets in this repo do: +0.035 m along z from ``iiwa_link_7``
    to ``iiwa_link_ee``), fold it into ``offset`` explicitly -- see the
    ``payload.offset_z`` comment in
    ``config/identification/kuka_lbr_iiwa_14_r820_table.yaml``.
    """

    mass: float = 0.0
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    size: float = 0.1

    @property
    def is_empty(self) -> bool:
        return self.mass <= 0.0

    def principal_inertia(self) -> tuple[float, float, float]:
        """Box inertia about its own centre of mass, principal axes."""
        value = self.mass * (2.0 * self.size**2) / 12.0
        return (value, value, value)

    def as_dict(self) -> dict:
        return {"mass": float(self.mass), "offset": [float(v) for v in self.offset], "size": float(self.size)}


def _last_link_name(asset: AssetSpec) -> str:
    """The child link of the last active joint -- not the URDF's last
    declared link, and not a trailing fixed end-effector frame."""
    joints = asset.resolve_active_joints()
    if len(joints) == 0:
        raise ValueError(f"asset {asset.name!r} has no active joints; payload injection needs a single chain")
    return joints[-1].child


@contextlib.contextmanager
def payload_asset(asset: AssetSpec, payload: Payload | None):
    """Yield ``asset`` with ``payload`` welded to its last link.

    Yields the asset unchanged when the payload is empty (identity, not a
    copy), so the bare-flange path costs nothing.  Otherwise, the modified
    URDF is written as a sibling of the original file, in the same
    directory, so every relative mesh reference in it keeps resolving
    without copying or symlinking a single mesh.
    """
    if payload is None or payload.is_empty:
        yield asset
        return

    source = Path(asset.urdf_path)
    text = source.read_text(encoding="utf-8")
    parent = _last_link_name(asset)
    if f'"{parent}"' not in text and f"'{parent}'" not in text:
        raise ValueError(f"asset {asset.name!r}: last active joint's child link {parent!r} not found in its URDF")

    ixx, iyy, izz = payload.principal_inertia()
    x, y, z = payload.offset
    injected = (
        '<link name="identification_payload">'
        '<inertial><origin rpy="0 0 0" xyz="0 0 0"/>'
        f'<mass value="{payload.mass:.9g}"/>'
        f'<inertia ixx="{ixx:.9g}" ixy="0" ixz="0" iyy="{iyy:.9g}" iyz="0" izz="{izz:.9g}"/>'
        "</inertial>"
        # A box matching the inertia's own geometry, so a fitted payload is
        # visible to the collision checker (R3_10 Sec 3.2): the original
        # design left this out and validated trajectories payload-free
        # entirely, which is fine for conditioning (a property of the state
        # trajectory, not the inertias -- R3_01 Sec 2.6) but not for
        # collision, where an unmodelled 5-25 cm box can pass straight
        # through the table or the arm.
        '<visual><origin rpy="0 0 0" xyz="0 0 0"/>'
        f'<geometry><box size="{payload.size:.9g} {payload.size:.9g} {payload.size:.9g}"/></geometry>'
        "</visual>"
        '<collision><origin rpy="0 0 0" xyz="0 0 0"/>'
        f'<geometry><box size="{payload.size:.9g} {payload.size:.9g} {payload.size:.9g}"/></geometry>'
        "</collision>"
        "</link>"
        '<joint name="identification_payload_joint" type="fixed">'
        f'<parent link="{parent}"/><child link="identification_payload"/>'
        f'<origin rpy="0 0 0" xyz="{x:.9g} {y:.9g} {z:.9g}"/>'
        "</joint>"
    )
    # A str.replace could match "</robot>" inside a comment; this finds only
    # the document's actual closing tag.
    index = text.rindex("</robot>")
    text = text[:index] + injected + text[index:]

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".urdf", prefix="identification_payload_",
        dir=source.resolve().parent, delete=False, encoding="utf-8",
    ) as handle:
        handle.write(text)
        target = Path(handle.name)
    try:
        yield replace(asset, urdf_path=target)
    finally:
        target.unlink(missing_ok=True)
