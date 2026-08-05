"""Portable configuration for explicit motor-to-link elastic transmissions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .assets import AssetSpec


_NUMERIC_FIELDS = frozenset((
    "stiffness", "damping", "motor_stiffness", "motor_damping",
    "effort_limit", "intermediate_mass", "intermediate_size", "link_mass",
))


def load_simulation_settings(path: str | Path, asset: AssetSpec) -> dict[str, Any]:
    """Load one YAML file into the backend-neutral runner configuration.

    ``elastic.defaults`` applies to every automatically discovered active
    joint; ``elastic.joints.<urdf_joint>`` overrides individual entries.
    ``link_mass`` is intentionally optional: omitting it preserves the URDF
    inertial mass, while setting it explicitly overrides that child link.
    """
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Simulation settings require PyYAML") from exc
    source = Path(path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"Simulation settings must be a mapping: {source}")
    configured_asset = raw.get("asset")
    if configured_asset is not None:
        if not isinstance(configured_asset, str) or not configured_asset.strip():
            raise ValueError("asset must be a non-empty asset name when supplied")
        if configured_asset != asset.name:
            raise ValueError(
                f"Simulation settings are for asset {configured_asset!r}, not {asset.name!r}: {source}"
            )
    elastic = raw.get("elastic", {})
    if not isinstance(elastic, Mapping):
        raise ValueError("elastic must be a mapping")
    defaults = _numeric_mapping(elastic.get("defaults", {}), "elastic.defaults")
    configured = elastic.get("joints", {})
    if not isinstance(configured, Mapping):
        raise ValueError("elastic.joints must map URDF joint names to parameter mappings")
    unknown = sorted(set(configured).difference(asset.joint_names))
    if unknown:
        raise ValueError(f"elastic.joints contains non-active joints: {', '.join(unknown)}")
    transmissions: dict[str, dict[str, float | None]] = {}
    for name in asset.joint_names:
        values = dict(defaults)
        if name in configured:
            values.update(_numeric_mapping(configured[name], f"elastic.joints.{name}"))
        transmissions[name] = values
    simulation = raw.get("simulation", {})
    if not isinstance(simulation, Mapping):
        raise ValueError("simulation must be a mapping")
    result: dict[str, Any] = {"transmissions": transmissions}
    if "gravity" in simulation:
        gravity = simulation["gravity"]
        if not isinstance(gravity, (list, tuple)) or len(gravity) != 3:
            raise ValueError("simulation.gravity must be a three-element numeric vector")
        result["gravity"] = tuple(float(value) for value in gravity)
    return result


def merge_runner_settings(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Merge CLI overrides without discarding per-joint settings from YAML."""
    merged = dict(base)
    for key, value in overrides.items():
        if value is not None:
            merged[key] = value
    return merged


def _numeric_mapping(value: Any, label: str) -> dict[str, float | None]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    unknown = sorted(set(value).difference(_NUMERIC_FIELDS))
    if unknown:
        raise ValueError(f"{label} has unsupported fields: {', '.join(unknown)}")
    result: dict[str, float | None] = {}
    for key, raw in value.items():
        if raw is None and key in {"effort_limit", "link_mass"}:
            result[key] = None
            continue
        try:
            parsed = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label}.{key} must be numeric") from exc
        if key != "damping" and parsed <= 0.0:
            raise ValueError(f"{label}.{key} must be positive")
        if key == "damping" and parsed < 0.0:
            raise ValueError(f"{label}.{key} must be non-negative")
        result[key] = parsed
    return result
