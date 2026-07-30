"""Robot-asset specifications shared by simulation, collection and calibration.

This module deliberately has no simulator dependency.  It is the single place
where a robot-specific configuration names a URDF and (optionally) constrains
which URDF joints are actuated.  All consumers can therefore use the same
joint order without embedding robot names in their implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree as ET


_ONE_DOF_TYPES = frozenset(("revolute", "continuous", "prismatic"))


@dataclass(frozen=True)
class UrdfJoint:
    """A one-DoF joint described by a URDF, in declaration order."""

    name: str
    joint_type: str
    parent: str
    child: str
    axis: tuple[float, float, float]
    lower: float | None = None
    upper: float | None = None
    effort: float | None = None
    velocity: float | None = None
    mimic: str | None = None

    @property
    def is_one_dof(self) -> bool:
        return self.joint_type in _ONE_DOF_TYPES


@dataclass(frozen=True)
class AssetSpec:
    """Configuration for a robot asset.

    ``active_joints`` is intentionally optional.  Empty means discover every
    non-mimic revolute/continuous/prismatic joint from the URDF.  When it is
    specified it is also the public state/input order used by trajectories and
    datasets; this is required whenever a source dataset uses a distinct order.
    """

    name: str
    urdf_path: Path
    active_joints: tuple[str, ...] = ()
    base_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    base_quaternion: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    self_collisions: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def resolve_active_joints(self) -> tuple[UrdfJoint, ...]:
        """Return configured active joints, validating names and joint types."""
        discovered = discover_urdf_joints(self.urdf_path)
        selectable = [joint for joint in discovered if joint.is_one_dof and joint.mimic is None]
        by_name = {joint.name: joint for joint in selectable}
        names = self.active_joints or tuple(joint.name for joint in selectable)
        if not names:
            raise ValueError(f"Asset {self.name!r} has no non-mimic 1-DoF URDF joints")
        missing = [name for name in names if name not in by_name]
        if missing:
            available = ", ".join(by_name) or "<none>"
            raise ValueError(
                f"Asset {self.name!r} selects unsupported joints {missing}; available: {available}"
            )
        if len(set(names)) != len(names):
            raise ValueError(f"Asset {self.name!r} contains duplicate active joint names")
        return tuple(by_name[name] for name in names)

    @property
    def joint_names(self) -> tuple[str, ...]:
        return tuple(joint.name for joint in self.resolve_active_joints())


def discover_urdf_joints(urdf_path: str | Path) -> tuple[UrdfJoint, ...]:
    """Parse the joint declarations of a URDF without requiring a simulator.

    This supports ordinary URDF files.  Xacro must be expanded before use so
    that discovery and Newton receive exactly the same robot description.
    """
    path = Path(urdf_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"URDF does not exist: {path}")
    root = ET.parse(path).getroot()
    if root.tag != "robot":
        raise ValueError(f"Expected a <robot> root in {path}, got <{root.tag}>")

    joints: list[UrdfJoint] = []
    for element in root.findall("joint"):
        name = element.get("name")
        joint_type = element.get("type")
        parent = element.find("parent")
        child = element.find("child")
        if not name or not joint_type or parent is None or child is None:
            raise ValueError(f"Malformed joint declaration in {path}")
        parent_link = parent.get("link")
        child_link = child.get("link")
        if not parent_link or not child_link:
            raise ValueError(f"Joint {name!r} in {path} has an invalid parent/child")
        axis_element = element.find("axis")
        axis_text = "1 0 0" if axis_element is None else axis_element.get("xyz", "1 0 0")
        try:
            axis_values = tuple(float(value) for value in axis_text.replace(",", " ").split())
        except ValueError as exc:
            raise ValueError(f"Joint {name!r} has invalid axis {axis_text!r}") from exc
        if len(axis_values) != 3:
            raise ValueError(f"Joint {name!r} axis must contain three values")
        limit = element.find("limit")
        mimic = element.find("mimic")
        joints.append(
            UrdfJoint(
                name=name,
                joint_type=joint_type,
                parent=parent_link,
                child=child_link,
                axis=axis_values,  # type: ignore[arg-type]
                lower=_optional_float(limit, "lower"),
                upper=_optional_float(limit, "upper"),
                effort=_optional_float(limit, "effort"),
                velocity=_optional_float(limit, "velocity"),
                mimic=None if mimic is None else mimic.get("joint"),
            )
        )
    return tuple(joints)


def load_asset_spec(path: str | Path) -> AssetSpec:
    """Load one portable asset YAML file.

    Relative ``urdf`` paths are resolved relative to the YAML file, making an
    asset folder relocatable.  A minimal file is simply ``name`` plus ``urdf``.
    """
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency is part of normal runtime
        raise ImportError("Loading an asset YAML requires PyYAML (pip install pyyaml)") from exc
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Asset spec must be a mapping: {source}")
    data = raw.get("asset", raw)
    if not isinstance(data, dict):
        raise ValueError(f"Asset section must be a mapping: {source}")
    name = _required_string(data, "name", source)
    urdf_text = _required_string(data, "urdf", source)
    urdf_path = (source.parent / urdf_text).resolve() if not Path(urdf_text).is_absolute() else Path(urdf_text)
    active = data.get("active_joints", ())
    if active is None:
        active = ()
    if not isinstance(active, list) or not all(isinstance(item, str) for item in active):
        raise ValueError(f"active_joints must be a list of joint names: {source}")
    base = data.get("base", {})
    if not isinstance(base, dict):
        raise ValueError(f"base must be a mapping: {source}")
    position = _float_tuple(base.get("position", (0.0, 0.0, 0.0)), 3, "base.position", source)
    quaternion = _float_tuple(base.get("quaternion", (0.0, 0.0, 0.0, 1.0)), 4, "base.quaternion", source)
    known = {"name", "urdf", "active_joints", "base", "self_collisions"}
    return AssetSpec(
        name=name,
        urdf_path=urdf_path,
        active_joints=tuple(active),
        base_position=position,  # type: ignore[arg-type]
        base_quaternion=quaternion,  # type: ignore[arg-type]
        self_collisions=bool(data.get("self_collisions", False)),
        metadata={key: value for key, value in data.items() if key not in known},
    )


class AssetRegistry:
    """A small filesystem registry for config-only robot definitions."""

    def __init__(self, spec_directories: Iterable[str | Path]):
        self.spec_directories = tuple(Path(item).expanduser().resolve() for item in spec_directories)

    @classmethod
    def for_repository(cls, repository_root: str | Path | None = None) -> "AssetRegistry":
        root = Path(repository_root or Path(__file__).resolve().parents[2]).resolve()
        return cls((root / "config" / "assets",))

    def available(self) -> dict[str, Path]:
        found: dict[str, Path] = {}
        for directory in self.spec_directories:
            if not directory.is_dir():
                continue
            for path in sorted((*directory.glob("*.yaml"), *directory.glob("*.yml"))):
                spec = load_asset_spec(path)
                if spec.name in found:
                    raise ValueError(f"Duplicate asset name {spec.name!r}: {found[spec.name]} and {path}")
                found[spec.name] = path
        return found

    def load(self, name: str) -> AssetSpec:
        available = self.available()
        try:
            return load_asset_spec(available[name])
        except KeyError as exc:
            choices = ", ".join(sorted(available)) or "<none>"
            raise KeyError(f"Unknown asset {name!r}; available: {choices}") from exc


def _optional_float(element: ET.Element | None, attr: str) -> float | None:
    if element is None or element.get(attr) is None:
        return None
    return float(element.get(attr))  # type: ignore[arg-type]


def _required_string(data: dict[str, Any], key: str, source: Path) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key!r} must be a non-empty string: {source}")
    return value


def _float_tuple(value: Any, length: int, label: str, source: Path) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{label} must have {length} numeric values: {source}")
    try:
        return tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain numeric values: {source}") from exc
