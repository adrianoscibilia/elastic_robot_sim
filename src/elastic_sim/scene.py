"""Compose a portable robot description into a larger static scene.

Dataset generation needs the arm mounted on a table rather than floating at
the world origin.  Rather than teach every backend about scene objects, the
table is folded into a derived URDF: MuJoCo, Newton and the Pinocchio
collision checker then all see the same geometry through the paths they
already use, and no runner changes at all.

Meshes are referenced back into the source asset instead of being copied, so
a scene stays cheap and the robot description remains the single source of
truth for inertias and joint limits.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from .assets import AssetSpec


def robot_root_link(root: ET.Element) -> str:
    """Return the one link that is never a joint child."""
    links = [link.get("name") for link in root.findall("link")]
    children = {
        child.get("link")
        for joint in root.findall("joint")
        for child in joint.findall("child")
    }
    roots = [name for name in links if name and name not in children]
    if len(roots) != 1:
        raise ValueError(f"expected exactly one root link, found {roots}")
    return str(roots[0])


def strip_world_root(root: ET.Element) -> tuple[ET.Element, tuple[str, str]]:
    """Remove a placeholder root link named ``world`` and its single fixed joint.

    Returns the modified tree and the removed joint's ``(xyz, rpy)`` origin so
    the caller can fold it into the table mount.  A no-op (returning
    ``("0 0 0", "0 0 0")``) when there is no link named ``world`` -- most
    portable URDFs (the iiwa's included) have none.

    A link named ``world`` that *is* present is a placeholder iff **all**
    hold: it has no ``inertial``/``visual``/``collision`` child; exactly one
    joint has it as parent; that joint is ``type="fixed"``.  Anything else
    raises ``ValueError`` naming the violated condition rather than guessing
    -- a link named ``world`` gets special-cased treatment by MuJoCo's and
    Newton's URDF importers, so silently keeping or dropping it on an
    assumption is not safe.
    """
    world_links = [link for link in root.findall("link") if link.get("name") == "world"]
    if not world_links:
        return root, ("0 0 0", "0 0 0")
    if len(world_links) > 1:
        raise ValueError(f"found {len(world_links)} links named 'world', expected at most one")
    world_link = world_links[0]
    extra = [child.tag for child in world_link if child.tag in ("inertial", "visual", "collision")]
    if extra:
        raise ValueError(f"link 'world' has {extra} children, so it is not a placeholder root")
    joints_from_world = [
        joint for joint in root.findall("joint")
        if (parent := joint.find("parent")) is not None and parent.get("link") == "world"
    ]
    if len(joints_from_world) != 1:
        raise ValueError(f"link 'world' is the parent of {len(joints_from_world)} joint(s), expected exactly 1")
    joint = joints_from_world[0]
    if joint.get("type") != "fixed":
        raise ValueError(f"joint {joint.get('name')!r} out of 'world' has type {joint.get('type')!r}, expected 'fixed'")
    origin = joint.find("origin")
    xyz = origin.get("xyz", "0 0 0") if origin is not None else "0 0 0"
    rpy = origin.get("rpy", "0 0 0") if origin is not None else "0 0 0"
    root.remove(world_link)
    root.remove(joint)
    return root, (xyz, rpy)


def _rotation_matrix_from_rpy(rpy: str) -> np.ndarray:
    """URDF convention: ``R = Rz(yaw) . Ry(pitch) . Rx(roll)``."""
    roll, pitch, yaw = (float(v) for v in rpy.split())
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return rz @ ry @ rx


def normalize_inertial_frames(root: ET.Element) -> ET.Element:
    """Fold every link's ``<inertial><origin rpy=...>`` rotation into the tensor.

    R4_Q (round 4, UR10 port): MuJoCo 3.6.0's rigid-body composition
    (``mj_crb``, feeding ``mj_inverse``) disagrees with Pinocchio 4.1.0's
    RNEA/CRBA by up to ~0.3% relative in specific mass-matrix entries for a
    link whose ``<inertial><origin>`` carries a non-identity ``rpy`` --
    reproduced on the bare ``ur10`` asset (so it predates and is independent
    of this round's table composition), even though each engine's own
    per-body inertia tensor, reconstructed from its native representation
    (Pinocchio's ``model.inertias[i]``; MuJoCo's ``body_inertia`` +
    ``body_iquat``), is bit-identical to the other's and to the tensor
    obtained by hand-rotating the URDF's raw coefficients by ``rpy``. The
    discrepancy is present at rest (``q=0``) already, is independent of the
    robot's configuration, and vanishes when the same physical tensor is
    instead expressed directly in the link frame (``rpy="0 0 0"``,
    coefficients pre-rotated) -- i.e. it is specific to how one of the two
    engines *composes* a rotated inertial frame through the kinematic chain,
    not to the tensor itself. This function performs that pre-rotation
    generically (any link, any asset), so it is a no-op -- verified,
    `R4_06 Sec A2` -- on every URDF in this repository that already uses
    ``rpy="0 0 0"`` everywhere (the iiwa's), and only changes links that
    actually have a rotated inertial frame (three on the UR10: shoulder,
    wrist_1, wrist_2).
    """
    for link in root.findall("link"):
        inertial = link.find("inertial")
        if inertial is None:
            continue
        origin = inertial.find("origin")
        rpy = origin.get("rpy", "0 0 0") if origin is not None else "0 0 0"
        if tuple(float(v) for v in rpy.split()) == (0.0, 0.0, 0.0):
            continue
        inertia = inertial.find("inertia")
        tensor = np.array([
            [float(inertia.get("ixx", 0.0)), float(inertia.get("ixy", 0.0)), float(inertia.get("ixz", 0.0))],
            [float(inertia.get("ixy", 0.0)), float(inertia.get("iyy", 0.0)), float(inertia.get("iyz", 0.0))],
            [float(inertia.get("ixz", 0.0)), float(inertia.get("iyz", 0.0)), float(inertia.get("izz", 0.0))],
        ])
        rotation = _rotation_matrix_from_rpy(rpy)
        rotated = rotation @ tensor @ rotation.T
        inertia.set("ixx", f"{rotated[0, 0]:.17g}")
        inertia.set("ixy", f"{rotated[0, 1]:.17g}")
        inertia.set("ixz", f"{rotated[0, 2]:.17g}")
        inertia.set("iyy", f"{rotated[1, 1]:.17g}")
        inertia.set("iyz", f"{rotated[1, 2]:.17g}")
        inertia.set("izz", f"{rotated[2, 2]:.17g}")
        origin.set("rpy", "0 0 0")
    return root


def _compose_translation_then_transform(translation: tuple[float, float, float], xyz: str, rpy: str) -> tuple[str, str]:
    """Fold ``T(translation) . T(xyz, rpy)`` into one origin, ``T(translation)`` having identity rotation.

    With the first transform's rotation the identity, the composed rotation
    is just ``rpy`` unchanged and the composed translation is a plain vector
    sum (no rotation of ``xyz`` is needed).  A non-identity first rotation
    would need real rotation-matrix composition; ``compose_table_scene``'s
    mount is always axis-aligned, so that case does not arise here.
    """
    dx, dy, dz = (float(v) for v in xyz.split())
    tx, ty, tz = translation
    return f"{tx + dx:.12g} {ty + dy:.12g} {tz + dz:.12g}", rpy


def _box_inertia(mass: float, size: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = size
    factor = mass / 12.0
    return factor * (y * y + z * z), factor * (x * x + z * z), factor * (x * x + y * y)


def compose_table_scene(
    asset: AssetSpec,
    *,
    output_path: str | Path,
    table_height: float = 0.75,
    table_size: tuple[float, float] = (1.2, 0.8),
    table_mass: float = 60.0,
    table_link: str = "table",
    scene_name: str | None = None,
) -> str:
    """Return URDF text mounting ``asset`` on a box table of ``table_height``.

    The table link frame sits on the floor, its box spans ``[0, table_height]``
    in z, and the robot's root link is attached by a fixed joint at the table
    top.  Mesh filenames are rewritten relative to ``output_path`` so the
    generated file can be committed next to a small ``asset.yaml`` without
    duplicating any mesh.
    """
    if table_height <= 0.0:
        raise ValueError("table_height must be positive")
    if table_mass <= 0.0 or any(value <= 0.0 for value in table_size):
        raise ValueError("table_size and table_mass must be positive")

    source = ET.parse(asset.urdf_path).getroot()
    source = normalize_inertial_frames(source)
    source, removed_origin = strip_world_root(source)
    root_link = robot_root_link(source)
    mount_xyz, mount_rpy = _compose_translation_then_transform(
        (0.0, 0.0, table_height), *removed_origin
    )
    name = scene_name or f"{asset.name}_table"
    scene = ET.Element("robot", {"name": name})

    size = (float(table_size[0]), float(table_size[1]), float(table_height))
    ixx, iyy, izz = _box_inertia(table_mass, size)
    link = ET.SubElement(scene, "link", {"name": table_link})
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", {"rpy": "0 0 0", "xyz": f"0 0 {table_height / 2.0:.12g}"})
    ET.SubElement(inertial, "mass", {"value": f"{table_mass:.12g}"})
    ET.SubElement(inertial, "inertia", {
        "ixx": f"{ixx:.12g}", "ixy": "0", "ixz": "0",
        "iyy": f"{iyy:.12g}", "iyz": "0", "izz": f"{izz:.12g}",
    })
    for tag in ("visual", "collision"):
        element = ET.SubElement(link, tag)
        ET.SubElement(element, "origin", {"rpy": "0 0 0", "xyz": f"0 0 {table_height / 2.0:.12g}"})
        geometry = ET.SubElement(element, "geometry")
        ET.SubElement(geometry, "box", {"size": f"{size[0]:.12g} {size[1]:.12g} {size[2]:.12g}"})

    mount = ET.SubElement(scene, "joint", {"name": f"{table_link}_to_{root_link}", "type": "fixed"})
    ET.SubElement(mount, "parent", {"link": table_link})
    ET.SubElement(mount, "child", {"link": root_link})
    ET.SubElement(mount, "origin", {"rpy": mount_rpy, "xyz": mount_xyz})

    target_dir = Path(output_path).expanduser().resolve().parent
    for element in source:
        if element.tag in {"link", "joint"}:
            scene.append(_rewrite_meshes(element, asset.urdf_path.parent, target_dir))
        elif element.tag == "material":
            scene.append(element)
    return ET.tostring(scene, encoding="unicode")


def _rewrite_meshes(element: ET.Element, source_dir: Path, target_dir: Path) -> ET.Element:
    """Repoint relative mesh filenames from ``source_dir`` to ``target_dir``."""
    for mesh in element.iter("mesh"):
        filename = mesh.get("filename")
        if not filename or filename.startswith(("package://", "file://", "/")):
            continue
        absolute = (source_dir / filename).resolve()
        mesh.set("filename", os.path.relpath(absolute, target_dir).replace(os.sep, "/"))
    return element


def write_table_scene(asset: AssetSpec, output_path: str | Path, **kwargs) -> Path:
    """Write :func:`compose_table_scene` output, creating parent directories."""
    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    text = compose_table_scene(asset, output_path=target, **kwargs)
    target.write_text('<?xml version="1.0" ?>\n' + text + "\n", encoding="utf-8")
    return target
