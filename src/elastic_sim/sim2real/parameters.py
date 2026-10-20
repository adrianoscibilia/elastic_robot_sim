"""Parameter registry v2 (S2R_01 §5, T-2.1; working overlay S2R_02 A-3).

Names are ``<group>.<subject>.<field>[@<run>]``. The *universe* of names is
expanded against the asset, the round-6 plant base and the config's runs
(:func:`parameter_universe`); the config's ``parameters.priors`` block gives
each name its prior and bounds through wildcard patterns (the last matching
pattern wins), and ``parameters.free`` selects what the optimizer moves.
Everything else is frozen at its value.

A free parameter lives in the optimizer's box ``z in [-1, 1]`` (log or
linear between its bounds). Its prior is ``(mean, sigma)`` in the
parameter's own scale (``sigma`` is a log-sigma for ``log``), and
:meth:`Registry.prior_penalty` is ``0.5 * sum(((f(x) - f(mean)) / sigma)^2)``.

**Working overlay (A-3).** The values a stage is working with are written to
``data/sim2real/<robot>/<run_id>/working.yaml`` (:meth:`Registry.write_overlay`)
and read back by every next replay (:meth:`Registry.with_overlay`). The final
calibrated file is published once, to ``config/sim2real/calibrated/``, with
``publish=True``. Nothing else in the repository is ever written: the asset,
``PARAMETER_PROVENANCE`` and the round-6 configs stay untouched.
"""

from __future__ import annotations

import dataclasses
import fnmatch
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml

from .config import _PRIOR_KEYS, REPLAY_MODES, REPO_ROOT

REGISTRY_SCHEMA = "elastic_sim.sim2real.registry/1"
OVERLAY_SCHEMA = "elastic_sim.sim2real.overlay/1"
SCALES = ("linear", "log")
#: Where a working overlay may be written, and where the final one is published.
WORKING_ROOT = REPO_ROOT / "data" / "sim2real"
PUBLISH_ROOT = REPO_ROOT / "config" / "sim2real" / "calibrated"

_ENTRY_KEYS = {"value", "mean", "sigma", "scale", "lower", "upper", "values", "frozen", "modes", "provenance"}
#: Report fields an overlay entry may carry (S2R_04 A-11, S2R_06 A-12);
#: read back, ignored. ``calibrated: false`` = "not calibrated: not seen by
#: these data" (frozen by A-12 at ``frozen_d_post``, or excluded by A-13).
ANNOTATION_KEYS = {"data_share", "data_share_posterior", "prior_determined", "calibrated", "frozen_d_post",
                   "not_calibrated_reason", "calibrated_by", "partners"}


class RegistryError(ValueError):
    """A parameter name, prior or value is invalid; the message names it."""


# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParameterSpec:
    """One resolved parameter: its value, prior, bounds and provenance.

    ``values`` makes the parameter discrete (``controller.delay_cycles``):
    it is never free (an outer grid moves it, not the optimizer) and has no
    bounds or prior.
    """

    name: str
    value: float
    mean: float
    sigma: float | None
    scale: str
    lower: float
    upper: float
    frozen: bool = True
    modes: tuple[str, ...] = REPLAY_MODES
    provenance: str = ""
    values: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.scale not in SCALES:
            raise RegistryError(f"{self.name}: scale must be one of {SCALES}, got {self.scale!r}")
        bad_modes = set(self.modes) - set(REPLAY_MODES)
        if bad_modes:
            raise RegistryError(f"{self.name}: unknown modes {sorted(bad_modes)}")
        if self.values is not None:
            if not self.frozen:
                raise RegistryError(f"{self.name}: a discrete parameter (values: ...) can't be free; "
                                    "the outer grid moves it")
            if self.value not in self.values:
                raise RegistryError(f"{self.name}: value {self.value!r} is not one of {list(self.values)}")
            return
        if not self.lower < self.upper:
            raise RegistryError(f"{self.name}: bounds [{self.lower}, {self.upper}] must satisfy lower < upper")
        if self.scale == "log" and self.lower <= 0.0:
            raise RegistryError(f"{self.name}: a log-scale parameter needs lower > 0 (got {self.lower})")
        for label, x in (("value", self.value), ("prior mean", self.mean)):
            if not self.lower <= x <= self.upper:
                raise RegistryError(f"{self.name}: {label} {x!r} is outside its bounds "
                                    f"[{self.lower!r}, {self.upper!r}]")
        if not self.frozen and (self.sigma is None or not self.sigma > 0.0):
            raise RegistryError(f"{self.name}: a free parameter needs a prior sigma > 0 (MAP, S2R_01 P-6)")

    @property
    def discrete(self) -> bool:
        return self.values is not None

    def _f(self, x: float) -> float:
        return math.log(x) if self.scale == "log" else float(x)

    def to_unit(self, x: float) -> float:
        """Physical value -> z in [-1, 1]."""
        lo, hi = self._f(self.lower), self._f(self.upper)
        return 2.0 * (self._f(x) - lo) / (hi - lo) - 1.0

    def from_unit(self, z: float) -> float:
        """z in [-1, 1] (clipped) -> physical value."""
        if z <= -1.0:
            return float(self.lower)
        if z >= 1.0:
            return float(self.upper)
        lo, hi = self._f(self.lower), self._f(self.upper)
        u = lo + 0.5 * (float(z) + 1.0) * (hi - lo)
        x = math.exp(u) if self.scale == "log" else u
        return float(min(max(x, self.lower), self.upper))

    def dx_dz(self, x: float) -> float:
        """d x / d z at the physical value x (inside the box)."""
        half = 0.5 * (self._f(self.upper) - self._f(self.lower))
        return half * float(x) if self.scale == "log" else half

    def dx_du(self, x: float) -> float:
        """d x / d u with u = f(x) / sigma: one prior sigma in x's units."""
        if self.sigma is None:
            raise RegistryError(f"{self.name}: no prior sigma")
        return self.sigma * float(x) if self.scale == "log" else self.sigma

    def prior_term(self, x: float) -> float:
        if self.sigma is None or self.discrete:
            return 0.0
        return 0.5 * ((self._f(x) - self._f(self.mean)) / self.sigma) ** 2

    def prior_distance(self, x: float) -> float | None:
        """Signed distance from the prior mean, in sigmas (A-2 (f))."""
        if self.sigma is None or self.discrete:
            return None
        return (self._f(x) - self._f(self.mean)) / self.sigma

    def to_entry(self) -> dict[str, Any]:
        entry: dict[str, Any] = {"value": _plain(self.value), "mean": _plain(self.mean), "sigma": _plain(self.sigma),
                                 "scale": self.scale, "frozen": bool(self.frozen), "modes": list(self.modes),
                                 "provenance": self.provenance}
        if self.discrete:
            entry["values"] = [_plain(v) for v in self.values]
        else:
            entry.update(lower=_plain(self.lower), upper=_plain(self.upper))
        return entry

    @classmethod
    def from_entry(cls, name: str, entry: Mapping[str, Any]) -> "ParameterSpec":
        unknown = sorted(set(entry) - _ENTRY_KEYS - ANNOTATION_KEYS)
        if unknown:
            raise RegistryError(f"{name}: unknown keys {unknown}")
        values = entry.get("values")
        sigma = entry.get("sigma")
        lower = min(values) if values else entry.get("lower", -math.inf)
        upper = max(values) if values else entry.get("upper", math.inf)
        return cls(name=name, value=float(entry["value"]), mean=float(entry.get("mean", entry["value"])),
                   sigma=None if sigma is None else float(sigma), scale=str(entry.get("scale", "linear")),
                   lower=float(lower), upper=float(upper),
                   frozen=bool(entry.get("frozen", True)), modes=tuple(entry.get("modes", REPLAY_MODES)),
                   provenance=str(entry.get("provenance", "")),
                   values=None if values is None else tuple(float(v) for v in values))


# ---------------------------------------------------------------------------
# Universe: every parameter name the plant knows, with its default value
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UniverseEntry:
    default: float | None  # None: the prior block must give the mean
    source: str            # where the default comes from


def parameter_universe(asset: Any, base: Any, runs: Iterable[str]) -> dict[str, UniverseEntry]:
    """Every parameter name for ``asset`` with its default (prior mean).

    ``base`` is a :class:`elastic_sim.sim2real.plant.PlantBase` (URDF
    inertials, round-6 nominals); ``runs`` the run ids that scope sensor
    offsets.
    """
    universe: dict[str, UniverseEntry] = {}
    for link, inertial in base.inertials.items():
        universe[f"inertial.{link}.mass"] = UniverseEntry(float(inertial.mass), "asset URDF")
        for axis, value in zip("xyz", inertial.com):
            universe[f"inertial.{link}.com_{axis}"] = UniverseEntry(float(value), "asset URDF")
        universe[f"inertial.{link}.inertia_scale"] = UniverseEntry(1.0, "asset URDF tensor x 1")
    for j, joint in enumerate(base.joint_names):
        lf, mf = base.link_friction, base.motor_friction
        universe[f"friction.link.{joint}.viscous"] = UniverseEntry(float(lf.viscous[j]), base.plant_base_source)
        universe[f"friction.link.{joint}.coulomb"] = UniverseEntry(float(lf.coulomb[j]), base.plant_base_source)
        universe[f"friction.motor.{joint}.viscous"] = UniverseEntry(float(mf.viscous[j]), "round-6 motor-friction rule")
        universe[f"friction.motor.{joint}.coulomb"] = UniverseEntry(float(mf.coulomb[j]), "round-6 motor-friction rule")
        universe[f"sensor.ft.{joint}.gain"] = UniverseEntry(None, "")
        for run in runs:
            universe[f"sensor.ft.{joint}.offset@{run}"] = UniverseEntry(None, "")
        universe[f"transmission.{joint}.stiffness"] = UniverseEntry(float(base.stiffness[j]), base.plant_base_source)
        universe[f"transmission.{joint}.damping_ratio"] = UniverseEntry(None, "")
        universe[f"transmission.{joint}.rotor_inertia"] = UniverseEntry(float(base.rotor_inertia[j]),
                                                                        base.plant_base_source)
        universe[f"controller.{joint}.bandwidth"] = UniverseEntry(None, "")
        universe[f"controller.{joint}.damping_ratio"] = UniverseEntry(None, "")
    for name, value in base.extras_defaults.items():
        universe[f"extras.{name}"] = UniverseEntry(float(value), base.plant_base_source)
    for name in ("model_scale", "delay_cycles", "drive_rate"):
        universe[f"controller.{name}"] = UniverseEntry(None, "")
    return universe


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class Registry:
    """The resolved parameter set: free parameters in a fixed order, the
    frozen ones at their values."""

    def __init__(self, specs: Iterable[ParameterSpec]):
        self.specs: dict[str, ParameterSpec] = {}
        for spec in specs:
            if spec.name in self.specs:
                raise RegistryError(f"duplicate parameter {spec.name}")
            self.specs[spec.name] = spec
        self.free_names: tuple[str, ...] = tuple(n for n, s in self.specs.items() if not s.frozen)

    # -- construction --------------------------------------------------------

    @classmethod
    def build(cls, universe: Mapping[str, UniverseEntry], priors: Mapping[str, Mapping[str, Any]],
              free: Sequence[str], frozen: Sequence[str] = (), *, mode: str | None = None) -> "Registry":
        """Resolve ``priors`` (pattern -> prior block) over ``universe``.

        ``free``/``frozen`` are wildcard lists; a name matched by both is an
        error, as is a pattern that matches nothing. A parameter without a
        mean (no default and none in its prior) is an error only if it's
        free; otherwise it's left out (the plant keeps its base value).
        ``mode``: a free parameter must act in it.
        """
        free_names = _expand(free, universe, "parameters.free")
        frozen_names = _expand(frozen, universe, "parameters.frozen")
        both = sorted(set(free_names) & set(frozen_names))
        if both:
            raise RegistryError(f"parameters both free and frozen: {', '.join(both)}")
        for pattern in priors:
            if not any(fnmatch.fnmatchcase(name, pattern) for name in universe):
                raise RegistryError(f"parameters.priors pattern {pattern!r} matches no parameter")
        specs = []
        for name, entry in universe.items():
            block: dict[str, Any] = {}
            for pattern, prior in priors.items():
                if fnmatch.fnmatchcase(name, pattern):
                    _check_prior_keys(pattern, prior)
                    block.update(prior)
            is_free = name in free_names
            mean = _mean(name, block.get("mean"), entry.default)
            if mean is None:
                if is_free:
                    raise RegistryError(f"{name}: free but has no prior mean (give parameters.priors.<pattern>.mean)")
                continue
            if not block and not is_free:
                specs.append(ParameterSpec(name=name, value=mean, mean=mean, sigma=None, scale="linear",
                                           lower=-math.inf, upper=math.inf, frozen=True,
                                           provenance=entry.source))
                continue
            spec = _spec_from_prior(name, block, mean, entry.source, frozen=not is_free)
            if is_free and mode is not None and mode not in spec.modes:
                raise RegistryError(f"{name}: free in replay mode {mode}, but it acts only in {list(spec.modes)}")
            specs.append(spec)
        missing = sorted(set(free_names) - {s.name for s in specs})
        if missing:
            raise RegistryError(f"free parameters without a spec: {', '.join(missing)}")
        return cls(specs)

    @classmethod
    def from_config(cls, config: Any, base: Any | None = None) -> "Registry":
        """The registry of a :class:`Sim2RealConfig` (base built if not given).
        With ``prior_from``, that overlay's values become the prior means and
        start values (S2R_01 §9)."""
        if base is None:
            from .plant import PlantBase

            base = PlantBase.from_config(config)
        universe = parameter_universe(base.asset, base, config.runs)
        registry = cls.build(universe, config.parameters.priors, config.parameters.free, config.parameters.frozen,
                             mode=config.replay.mode)
        prior_from = getattr(config, "prior_from", None)
        if prior_from is not None:
            if not Path(prior_from).is_file():
                raise RegistryError(f"prior_from overlay {prior_from} not found (run the previous stage first)")
            registry = registry.with_overlay(prior_from, as_prior=True)
        excluded = unseen_offsets(registry, config)
        return registry.freeze({n: registry.specs[n].value for n in excluded}) if excluded else registry

    # -- encode / decode -----------------------------------------------------

    @property
    def n_free(self) -> int:
        return len(self.free_names)

    def values(self) -> dict[str, float]:
        """theta: every parameter's current value (physical units)."""
        return {name: spec.value for name, spec in self.specs.items()}

    def decode(self, z: np.ndarray) -> dict[str, float]:
        """z in [-1, 1]^n_free -> the full theta (frozen ones at their value)."""
        z = np.asarray(z, dtype=float).reshape(-1)
        if z.shape != (self.n_free,):
            raise RegistryError(f"decode: z has {z.size} entries, the registry {self.n_free} free parameters")
        theta = self.values()
        for name, value in zip(self.free_names, z):
            theta[name] = self.specs[name].from_unit(float(value))
        return theta

    def encode(self, theta: Mapping[str, float]) -> np.ndarray:
        """theta (needs every free parameter) -> z."""
        missing = [n for n in self.free_names if n not in theta]
        if missing:
            raise RegistryError(f"encode: theta lacks free parameters {missing}")
        return np.asarray([self.specs[n].to_unit(float(theta[n])) for n in self.free_names], dtype=float)

    def z0(self) -> np.ndarray:
        return self.encode(self.values())

    def prior_penalty(self, theta: Mapping[str, float]) -> float:
        """0.5 * sum(((f(x) - f(mean)) / sigma)^2) over the free parameters."""
        return float(sum(self.specs[n].prior_term(float(theta[n])) for n in self.free_names))

    def prior_residuals(self, theta: Mapping[str, float]) -> np.ndarray:
        """(f(x) - f(mean)) / sigma per free parameter: the prior rows of the
        stacked least-squares residual (sum of squares / 2 = prior_penalty)."""
        return np.asarray([self.specs[n].prior_distance(float(theta[n])) for n in self.free_names], dtype=float)

    def dtheta_dz(self, theta: Mapping[str, float]) -> np.ndarray:
        """d theta / d z per free parameter (the chain rule's diagonal)."""
        return np.asarray([self.specs[n].dx_dz(float(theta[n])) for n in self.free_names], dtype=float)

    def prior_rows_dz(self) -> np.ndarray:
        """d prior_residuals / d z: (half the box width in f) / sigma, constant."""
        return np.asarray([0.5 * (self.specs[n]._f(self.specs[n].upper) - self.specs[n]._f(self.specs[n].lower))
                           / self.specs[n].sigma for n in self.free_names], dtype=float)

    def dtheta_du(self, theta: Mapping[str, float], names: Sequence[str] | None = None) -> np.ndarray:
        """d theta / d u, u in prior-sigma units (identifiability scaling)."""
        names = self.free_names if names is None else names
        return np.asarray([self.specs[n].dx_du(float(theta[n])) for n in names], dtype=float)

    # -- variants ------------------------------------------------------------

    def with_values(self, values: Mapping[str, float], *, as_prior: bool = False) -> "Registry":
        """A copy with new values (and, ``as_prior``, prior means: S2R_01 §9,
        the best theta of a stage is the prior mean of the next). Unknown names
        are an error; values are bound-checked."""
        unknown = sorted(set(values) - set(self.specs))
        if unknown:
            raise RegistryError(f"unknown parameters: {', '.join(unknown)}")
        specs = []
        for name, spec in self.specs.items():
            if name in values:
                value = float(values[name])
                spec = dataclasses.replace(spec, value=value, mean=value if as_prior else spec.mean)
            specs.append(spec)
        return Registry(specs)

    def with_overlay(self, overlay: "str | Path | Registry", *, as_prior: bool = False) -> "Registry":
        """Apply a working/calibrated overlay's values (the --theta of replay)."""
        source = overlay if isinstance(overlay, Registry) else Registry.from_overlay(overlay)
        return self.with_values({n: s.value for n, s in source.specs.items()}, as_prior=as_prior)

    def freeze(self, values: Mapping[str, float]) -> "Registry":
        """A copy with ``values``' parameters frozen at those values (A-12's
        freeze, A-13's exclusion); prior and bounds are kept for reports."""
        unknown = sorted(set(values) - set(self.specs))
        if unknown:
            raise RegistryError(f"unknown parameters: {', '.join(unknown)}")
        return Registry(dataclasses.replace(spec, value=float(values[name]), frozen=True) if name in values else spec
                        for name, spec in self.specs.items())

    def unfreeze(self, names: Sequence[str]) -> "Registry":
        """A copy with ``names`` free (an overlay of a stage's full set)."""
        return Registry(dataclasses.replace(spec, frozen=False) if name in names else spec
                        for name, spec in self.specs.items())

    # -- YAML ------------------------------------------------------------------

    def to_mapping(self, *, free_only: bool = False) -> dict[str, Any]:
        names = self.free_names if free_only else tuple(self.specs)
        return {name: self.specs[name].to_entry() for name in names}

    def to_yaml(self) -> str:
        return yaml.safe_dump({"schema": REGISTRY_SCHEMA, "parameters": self.to_mapping()}, sort_keys=False)

    @classmethod
    def from_yaml(cls, text: str) -> "Registry":
        raw = yaml.safe_load(text) or {}
        if raw.get("schema") not in (REGISTRY_SCHEMA, OVERLAY_SCHEMA):
            raise RegistryError(f"not a registry/overlay file (schema {raw.get('schema')!r})")
        return cls(ParameterSpec.from_entry(str(n), e) for n, e in (raw.get("parameters") or {}).items())

    def to_overlay(self, theta: Mapping[str, float] | None = None, *,
                   annotations: Mapping[str, Mapping[str, Any]] | None = None, **header: Any) -> str:
        """The overlay text: the free parameters at ``theta`` (default: their
        values), physical units, with prior, bounds and provenance, and per
        parameter the ``annotations`` (A-11: data share d)."""
        registry = self if theta is None else self.with_values({n: theta[n] for n in self.free_names})
        parameters = registry.to_mapping(free_only=True)
        for name, extra in (annotations or {}).items():
            unknown = sorted(set(extra) - ANNOTATION_KEYS)
            if unknown or name not in parameters:
                raise RegistryError(f"overlay annotation {name}: unknown parameter or keys {unknown}")
            parameters[name].update({k: _plain(v) for k, v in extra.items()})
        payload = {"schema": OVERLAY_SCHEMA, **{k: _plain(v) for k, v in header.items()}, "parameters": parameters}
        return yaml.safe_dump(payload, sort_keys=False)

    def write_overlay(self, path: str | Path, theta: Mapping[str, float] | None = None, *, publish: bool = False,
                      annotations: Mapping[str, Mapping[str, Any]] | None = None, **header: Any) -> Path:
        """Write the overlay atomically. Inside the repository it may only go
        under ``data/sim2real/`` (working) or, with ``publish=True``, under
        ``config/sim2real/calibrated/`` (the final file, A-3)."""
        path = Path(path).expanduser().resolve()
        check_write_target(path, publish=publish)
        header.setdefault("kind", "calibrated" if publish else "working")
        header.setdefault("written", datetime.now(timezone.utc).isoformat(timespec="seconds"))
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(self.to_overlay(theta, annotations=annotations, **header), encoding="utf-8")
        tmp.replace(path)
        return path

    @classmethod
    def from_overlay(cls, path: str | Path) -> "Registry":
        """The overlay's parameters as a registry (all free, as written)."""
        return cls.from_yaml(Path(path).read_text(encoding="utf-8"))

    # -- reports -----------------------------------------------------------------

    def rows(self, theta: Mapping[str, float] | None = None, *, free_only: bool = True) -> list[dict[str, Any]]:
        theta = self.values() if theta is None else theta
        names = self.free_names if free_only else tuple(self.specs)
        rows = []
        for name in names:
            spec = self.specs[name]
            value = float(theta.get(name, spec.value))
            rows.append({"name": name, "value": value, "mean": spec.mean, "sigma": spec.sigma, "scale": spec.scale,
                         "lower": spec.lower, "upper": spec.upper, "frozen": spec.frozen,
                         "prior_distance_sigma": spec.prior_distance(value), "provenance": spec.provenance})
        return rows

    def to_table(self, theta: Mapping[str, float] | None = None, *, free_only: bool = True) -> str:
        """Markdown table for reports: value, prior, bounds, distance in sigma."""
        lines = ["| parameter | value | prior mean | sigma | scale | bounds | dist. (sigma) | provenance |",
                 "|---|---|---|---|---|---|---|---|"]
        for row in self.rows(theta, free_only=free_only):
            distance = "" if row["prior_distance_sigma"] is None else f"{row['prior_distance_sigma']:+.2f}"
            sigma = "" if row["sigma"] is None else f"{row['sigma']:.3g}"
            lines.append(f"| `{row['name']}` | {row['value']:.6g} | {row['mean']:.6g} | {sigma} | {row['scale']} | "
                         f"[{row['lower']:.4g}, {row['upper']:.4g}] | {distance} | {row['provenance']} |")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def unseen_offsets(registry: Registry, config: Any) -> list[str]:
    """S2R_06 A-13: free ``sensor.ft.*.offset@<run>`` of runs no training
    pattern names (the test run): the stage cannot see them, the session
    zero (``split.zero``) fits them afterwards."""
    split = getattr(config, "split", None)
    if split is None:
        return []
    trained = set(split.runs_of("train", config.runs))
    return [n for n in registry.free_names if ".offset@" in n and n.rsplit("@", 1)[1] not in trained]


def check_write_target(path: Path, *, publish: bool = False) -> None:
    """Refuse to write inside the repository except under ``data/sim2real/``
    (and ``config/sim2real/calibrated/`` when publishing)."""
    path = Path(path).resolve()
    if not _inside(path, REPO_ROOT):
        return
    allowed = [WORKING_ROOT] + ([PUBLISH_ROOT] if publish else [])
    if not any(_inside(path, root) for root in allowed):
        where = " or ".join(str(r.relative_to(REPO_ROOT)) for r in allowed)
        raise RegistryError(f"refusing to write {path}: inside the repository, sim2real writes only under {where} "
                            "(S2R_02 A-3: asset, provenance and round-6 configs stay untouched)")


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _expand(patterns: Sequence[str], universe: Mapping[str, Any], label: str) -> list[str]:
    names = []
    for pattern in patterns:
        hits = [name for name in universe if fnmatch.fnmatchcase(name, pattern)]
        if not hits:
            raise RegistryError(f"{label}: pattern {pattern!r} matches no parameter")
        names.extend(h for h in hits if h not in names)
    return names


def _check_prior_keys(pattern: str, prior: Any) -> None:
    if not isinstance(prior, Mapping):
        raise RegistryError(f"parameters.priors.{pattern} must be a mapping")
    unknown = sorted(set(prior) - _PRIOR_KEYS)
    if unknown:
        raise RegistryError(f"unknown parameters.priors.{pattern} keys: {', '.join(unknown)}")


def _mean(name: str, given: Any, default: float | None) -> float | None:
    if given is None:
        return default
    if isinstance(given, (int, float)):
        return float(given)
    raise RegistryError(f"{name}: prior mean must be a number (or omitted to use the asset/plant-base value)")


def _spec_from_prior(name: str, block: Mapping[str, Any], mean: float, source: str, *, frozen: bool) -> ParameterSpec:
    provenance = str(block.get("provenance") or source)
    modes = tuple(block.get("modes", REPLAY_MODES))
    if "values" in block:
        values = tuple(float(v) for v in block["values"])
        return ParameterSpec(name=name, value=mean, mean=mean, sigma=None, scale="linear", lower=min(values),
                             upper=max(values), frozen=frozen, modes=modes, provenance=provenance, values=values)
    forms = [key for key in ("bounds", "factor", "offset") if key in block]
    if len(forms) != 1:
        raise RegistryError(f"{name}: its prior needs exactly one of bounds / factor / offset (got {forms or 'none'})")
    lo, hi = (float(v) for v in block[forms[0]])
    if forms[0] == "factor":
        lo, hi = mean * lo, mean * hi
    elif forms[0] == "offset":
        lo, hi = mean + lo, mean + hi
    sigma = block.get("sigma")
    return ParameterSpec(name=name, value=mean, mean=mean, sigma=None if sigma is None else float(sigma),
                         scale=str(block.get("scale", "linear")), lower=lo, upper=hi, frozen=frozen, modes=modes,
                         provenance=provenance)


def _plain(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


__all__ = ["unseen_offsets", "ANNOTATION_KEYS", "OVERLAY_SCHEMA", "REGISTRY_SCHEMA", "ParameterSpec", "Registry", "RegistryError", "UniverseEntry",
           "check_write_target", "parameter_universe"]
