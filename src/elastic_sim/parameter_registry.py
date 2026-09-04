"""Asset-defined physical parameter registry for sim-to-real calibration.

The registry deliberately keeps optimizer coordinates separate from physical
values.  Optimizers see a bounded ``[-1, 1]`` vector while model adapters see
named SI-valued parameters.  This makes adding a parameter an asset/config
change rather than a rewrite of every optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    lower: float
    upper: float
    initial: float
    log_scale: bool = True
    enabled: bool = True
    backend: str = "all"
    target: str | None = None

    def __post_init__(self) -> None:
        if self.upper <= self.lower:
            raise ValueError(f"invalid bounds for parameter {self.name!r}")
        if self.log_scale and self.lower <= 0.0:
            raise ValueError(f"log-scaled parameter {self.name!r} needs a positive lower bound")
        if not self.lower <= self.initial <= self.upper:
            raise ValueError(f"initial value for {self.name!r} is outside its bounds")
        if not self.name.strip():
            raise ValueError("parameter names must be non-empty")


class ParameterRegistry:
    """Validated, named calibration parameters and normalized coordinates."""

    def __init__(self, specs: Sequence[ParameterSpec]) -> None:
        active = tuple(spec for spec in specs if spec.enabled)
        if not active:
            raise ValueError("parameter registry has no enabled parameters")
        if len({spec.name for spec in active}) != len(active):
            raise ValueError("parameter names must be unique")
        self.specs = active

    @classmethod
    def from_config(cls, config: Mapping[str, Any], joint_names: Sequence[str]) -> "ParameterRegistry":
        raw = config.get("calibration", {}).get("parameters", ())
        if raw:
            if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
                raise ValueError("calibration.parameters must be a list")
            specs = []
            for item in raw:
                if not isinstance(item, Mapping):
                    raise ValueError("each calibration parameter must be a mapping")
                specs.append(
                    ParameterSpec(
                        name=str(item["name"]),
                        lower=float(item["lower"]),
                        upper=float(item["upper"]),
                        initial=float(item.get("initial", item["lower"])),
                        log_scale=bool(item.get("scale", item.get("log_scale", True)) in (True, "log", "logarithmic")),
                        enabled=bool(item.get("enabled", True)),
                        backend=str(item.get("backend", "all")),
                        target=None if item.get("target") is None else str(item["target"]),
                    )
                )
            return cls(specs)

        # Safe defaults for a new asset.  Payload is intentionally not added.
        model = config.get("model", config)
        transmissions = model.get("transmissions", {}) if isinstance(model, Mapping) else {}
        specs: list[ParameterSpec] = []
        for joint in joint_names:
            values = transmissions.get(joint, {}) if isinstance(transmissions, Mapping) else {}
            for field, lo, hi, log in (
                ("stiffness", 1.0, 1.0e6, True),
                ("damping", 1.0e-3, 1.0e5, True),
                ("motor_stiffness", 1.0, 1.0e6, True),
                ("motor_damping", 1.0e-3, 1.0e5, True),
                ("intermediate_mass", 1.0e-5, 1.0e2, True),
            ):
                initial = float(values.get(field, model.get(f"default_{field}", lo * 10.0)))
                initial = float(np.clip(initial, lo, hi))
                specs.append(ParameterSpec(
                    f"transmission.{joint}.{field}", lo, hi, initial, log_scale=log
                ))
        return cls(specs)

    @property
    def bounds(self) -> list[tuple[float, float]]:
        return [(-1.0, 1.0)] * len(self.specs)

    @property
    def initial(self) -> np.ndarray:
        return self.encode({spec.name: spec.initial for spec in self.specs})

    def decode(self, theta: Sequence[float]) -> dict[str, float]:
        values = np.asarray(theta, dtype=float).reshape(-1)
        if len(values) != len(self.specs):
            raise ValueError(f"theta has {len(values)} values; expected {len(self.specs)}")
        result: dict[str, float] = {}
        for value, spec in zip(values, self.specs):
            value = float(np.clip(value, -1.0, 1.0))
            if spec.log_scale:
                result[spec.name] = float(np.exp(np.interp(value, (-1.0, 1.0), (np.log(spec.lower), np.log(spec.upper)))))
            else:
                result[spec.name] = float(np.interp(value, (-1.0, 1.0), (spec.lower, spec.upper)))
        return result

    def encode(self, params: Mapping[str, float]) -> np.ndarray:
        result = []
        for spec in self.specs:
            value = float(params[spec.name])
            if not spec.lower <= value <= spec.upper:
                raise ValueError(f"value for {spec.name!r} is outside its bounds")
            if spec.log_scale:
                value, lower, upper = np.log(value), np.log(spec.lower), np.log(spec.upper)
            else:
                lower, upper = spec.lower, spec.upper
            result.append(2.0 * (value - lower) / (upper - lower) - 1.0)
        return np.asarray(result, dtype=float)

    def to_dict(self) -> list[dict[str, Any]]:
        return [
            {"name": s.name, "lower": s.lower, "upper": s.upper, "initial": s.initial,
             "scale": "log" if s.log_scale else "linear", "enabled": s.enabled,
             "backend": s.backend, "target": s.target}
            for s in self.specs
        ]


def apply_parameter_overrides(config: Mapping[str, Any], params: Mapping[str, float]) -> dict[str, Any]:
    """Return a deep-copied model config with named registry values applied."""
    import copy

    result = copy.deepcopy(dict(config))
    model = result.setdefault("model", {})
    if not isinstance(model, dict):
        raise ValueError("model configuration must be a mapping")
    transmissions = model.setdefault("transmissions", {})
    if not isinstance(transmissions, dict):
        raise ValueError("model.transmissions must be a mapping")
    for name, value in params.items():
        parts = name.split(".")
        if len(parts) == 3 and parts[0] == "transmission":
            transmissions.setdefault(parts[1], {})[parts[2]] = float(value)
        elif len(parts) == 3 and parts[0] == "body":
            overrides = model.setdefault("body_overrides", {}).setdefault(parts[1], {})
            overrides[parts[2]] = float(value)
        else:
            raise ValueError(f"unsupported parameter name {name!r}")
    return result
