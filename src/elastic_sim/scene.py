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

import os
from pathlib import Path
from xml.etree import ElementTree as ET

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
    root_link = robot_root_link(source)
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
    ET.SubElement(mount, "origin", {"rpy": "0 0 0", "xyz": f"0 0 {table_height:.12g}"})

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
