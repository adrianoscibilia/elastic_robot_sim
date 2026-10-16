"""Configuration schema ``elastic_sim.sim2real/1`` (S2R_01 §7).

A config may ``extends:`` another file (path relative to the extending
file). Mappings merge recursively, everything else (lists, scalars) is
replaced by the extending file. Unknown keys fail with the offending block
and key named, in the ``dataset_config._check_keys`` style.

Relative data paths (``runs.*.path``, ``plant_base``, ``output``) resolve
against the repository root, like every other config in ``config/``.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

SCHEMA = "elastic_sim.sim2real/1"
REPO_ROOT = Path(__file__).resolve().parents[3]

SEGMENT_KINDS = ("excitation", "identify_excitation", "hold")
REPLAY_MODES = ("R0", "R1", "R2")
OPTIMIZER_METHODS = ("least_squares", "cma", "bo", "skrl_ppo", "random")
SPLITS = ("train", "val", "test")
LOSS_CHANNELS = ("ft", "tracking", "tau_cmd")

_TOP_KEYS = {"schema", "extends", "asset", "plant_base", "runs", "segments", "split", "replay",
             "parameters", "loss", "optimizer", "compute", "output"}
_RUN_KEYS = {"path", "sha256"}
_REPLAY_KEYS = {"mode", "r0", "r1", "r2"}
_R0_KEYS = {"sg_window", "sg_poly", "decimate", "deflection"}
_R1_KEYS = {"track_bandwidth", "windows", "burn_in_s"}
_R1_WINDOW_KEYS = {"length_s", "stride_s", "burn_in_s"}
_R2_KEYS = {"reference", "emulation"}
_PARAMETER_KEYS = {"free", "frozen"}
_LOSS_KEYS = {"channels", "prior_weight"}
_OPTIMIZER_KEYS = {"method", "budget", "seed", "cma", "bo", "skrl_ppo"}
_CMA_KEYS = {"sigma0", "popsize"}
_BO_KEYS = {"n_initial_points", "acq_func"}
_SKRL_KEYS = {"rollouts", "learning_epochs", "device"}
_COMPUTE_KEYS = {"jobs"}

_R0_DEFLECTIONS = ("measured_ft",)
_R2_REFERENCES = ("ipo_position",)
_R2_EMULATIONS = ("kuka_position", "pd_gravity")
_BO_ACQUISITIONS = ("EI", "LCB", "PI")
_RUN_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


class ConfigError(ValueError):
    """A sim2real config is malformed; the message names the block and key."""


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    path: Path
    sha256: str | None


@dataclass(frozen=True)
class SplitSpec:
    """Glob patterns ``<run>:<bag>`` per split."""

    train: tuple[str, ...] = ()
    val: tuple[str, ...] = ()
    test: tuple[str, ...] = ()

    def split_of(self, run_id: str, bag: str) -> str | None:
        """The split a bag belongs to, or ``None``; a bag in two splits is an error."""
        hits = [name for name in SPLITS
                if any(_pattern_matches(pattern, run_id, bag) for pattern in getattr(self, name))]
        if len(hits) > 1:
            raise ConfigError(f"split: {run_id}:{bag} matches more than one split ({', '.join(hits)})")
        return hits[0] if hits else None


@dataclass(frozen=True)
class R0Settings:
    sg_window: int = 31
    sg_poly: int = 3
    decimate: int = 5
    deflection: str = "measured_ft"


@dataclass(frozen=True)
class R1Windows:
    length_s: float
    stride_s: float
    burn_in_s: float = 0.5


@dataclass(frozen=True)
class R1Settings:
    track_bandwidth: float = 300.0
    windows: R1Windows | None = None
    burn_in_s: float = 0.0


@dataclass(frozen=True)
class R2Settings:
    reference: str = "ipo_position"
    emulation: str = "kuka_position"


@dataclass(frozen=True)
class ReplaySettings:
    mode: str = "R0"
    r0: R0Settings = field(default_factory=R0Settings)
    r1: R1Settings = field(default_factory=R1Settings)
    r2: R2Settings = field(default_factory=R2Settings)


@dataclass(frozen=True)
class ParameterSelection:
    free: tuple[str, ...] = ()
    frozen: tuple[str, ...] = ()


@dataclass(frozen=True)
class LossSettings:
    channels: Mapping[str, float] = field(default_factory=lambda: {"ft": 1.0})
    prior_weight: float | None = None  # None = "auto" (1 / residual rows per evaluation)


@dataclass(frozen=True)
class OptimizerSettings:
    method: str = "least_squares"
    budget: int = 200
    seed: int = 20261014
    cma: Mapping[str, Any] = field(default_factory=lambda: {"sigma0": 0.3, "popsize": None})
    bo: Mapping[str, Any] = field(default_factory=lambda: {"n_initial_points": 15, "acq_func": "EI"})
    skrl_ppo: Mapping[str, Any] = field(default_factory=lambda: {"rollouts": 16, "learning_epochs": 4,
                                                                   "device": "cpu"})


@dataclass(frozen=True)
class Sim2RealConfig:
    source: Path
    asset: str
    plant_base: Path | None
    runs: Mapping[str, RunSpec]
    segments: tuple[str, ...]
    split: SplitSpec
    replay: ReplaySettings
    parameters: ParameterSelection
    loss: LossSettings
    optimizer: OptimizerSettings
    jobs: int | None  # None = "auto" (nproc - 1)
    output: Path
    resolved: Mapping[str, Any]  # the merged mapping, for provenance records


def load_config(path: str | Path) -> Sim2RealConfig:
    source = Path(path).expanduser().resolve()
    return parse_config(_load_merged(source, ()), source)


def _load_merged(source: Path, chain: tuple[Path, ...]) -> dict[str, Any]:
    if source in chain:
        raise ConfigError(f"extends cycle: {' -> '.join(str(p) for p in chain + (source,))}")
    if not source.is_file():
        raise ConfigError(f"config not found: {source}")
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{source}: top level must be a mapping")
    _check_keys(raw, _TOP_KEYS, "top-level", source)
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    base = _load_merged((source.parent / str(parent)).resolve(), chain + (source,))
    return _merge(base, raw)


def _merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def parse_config(raw: Mapping[str, Any], source: Path) -> Sim2RealConfig:
    """Validate a merged mapping (``extends`` already resolved)."""
    _check_keys(raw, _TOP_KEYS - {"extends"}, "top-level", source)
    if raw.get("schema") != SCHEMA:
        raise ConfigError(f"{source}: schema must be {SCHEMA!r}, got {raw.get('schema')!r}")
    asset = raw.get("asset")
    if not isinstance(asset, str) or not asset:
        raise ConfigError(f"{source}: asset is required (the bare asset recorded as the contract's sim_asset)")
    runs = _runs(raw.get("runs"), source)
    segments = tuple(raw.get("segments", ["excitation"]))
    for kind in segments:
        if kind not in SEGMENT_KINDS:
            raise ConfigError(f"{source}: unknown segments entry {kind!r} (allowed: {', '.join(SEGMENT_KINDS)})")
    plant_base = raw.get("plant_base")
    return Sim2RealConfig(
        source=source,
        asset=asset,
        plant_base=_repo_path(plant_base) if plant_base else None,
        runs=runs,
        segments=segments,
        split=_split(raw.get("split") or {}, runs, source),
        replay=_replay(raw.get("replay") or {}, source),
        parameters=_parameters(raw.get("parameters") or {}, source),
        loss=_loss(raw.get("loss") or {}, source),
        optimizer=_optimizer(raw.get("optimizer") or {}, source),
        jobs=_jobs(raw.get("compute") or {}, source),
        output=_repo_path(raw.get("output", f"data/sim2real/{asset}")),
        resolved=dict(raw),
    )


def _check_keys(mapping: Any, allowed: set[str], block: str, source: Path) -> None:
    if not isinstance(mapping, Mapping):
        raise ConfigError(f"{source}: {block} must be a mapping, got {type(mapping).__name__}")
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ConfigError(f"unknown {block} keys in {source}: {', '.join(map(str, unknown))}")


def _repo_path(value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _runs(raw: Any, source: Path) -> dict[str, RunSpec]:
    if not raw:
        raise ConfigError(f"{source}: runs is required (run id -> {{path, sha256}})")
    _check_keys(raw, set(raw) if isinstance(raw, Mapping) else set(), "runs", source)
    runs = {}
    for run_id, spec in raw.items():
        run_id = str(run_id)
        if not _RUN_ID.match(run_id):
            raise ConfigError(f"{source}: run id {run_id!r} must be an identifier (letters, digits, _)")
        _check_keys(spec, _RUN_KEYS, f"runs.{run_id}", source)
        if "path" not in spec:
            raise ConfigError(f"{source}: runs.{run_id}.path is required")
        sha = spec.get("sha256")
        if sha is not None and not re.fullmatch(r"[0-9a-f]{64}", str(sha)):
            raise ConfigError(f"{source}: runs.{run_id}.sha256 must be 64 lowercase hex digits")
        runs[run_id] = RunSpec(run_id, _repo_path(spec["path"]), None if sha is None else str(sha))
    return runs


def _pattern_matches(pattern: str, run_id: str, bag: str) -> bool:
    run_pattern, bag_pattern = pattern.split(":", 1)
    return fnmatch.fnmatchcase(run_id, run_pattern) and fnmatch.fnmatchcase(bag, bag_pattern)


def _split(raw: Any, runs: Mapping[str, RunSpec], source: Path) -> SplitSpec:
    _check_keys(raw, set(SPLITS), "split", source)
    values = {}
    for name in SPLITS:
        patterns = raw.get(name) or []
        if isinstance(patterns, str) or not isinstance(patterns, (list, tuple)):
            raise ConfigError(f"{source}: split.{name} must be a list of '<run>:<bag>' patterns")
        for pattern in patterns:
            if not isinstance(pattern, str) or ":" not in pattern:
                raise ConfigError(f"{source}: split.{name} entry {pattern!r} must be '<run>:<bag-glob>'")
            run_pattern = pattern.split(":", 1)[0]
            if not any(fnmatch.fnmatchcase(run_id, run_pattern) for run_id in runs):
                raise ConfigError(f"{source}: split.{name} entry {pattern!r} names no run in runs "
                                  f"({', '.join(runs)})")
        values[name] = tuple(patterns)
    return SplitSpec(**values)


def _replay(raw: Any, source: Path) -> ReplaySettings:
    _check_keys(raw, _REPLAY_KEYS, "replay", source)
    mode = str(raw.get("mode", "R0"))
    if mode not in REPLAY_MODES:
        raise ConfigError(f"{source}: replay.mode must be one of {', '.join(REPLAY_MODES)}, got {mode!r}")
    r0_raw = raw.get("r0") or {}
    _check_keys(r0_raw, _R0_KEYS, "replay.r0", source)
    r0 = R0Settings(**{**R0Settings().__dict__, **r0_raw})
    if r0.sg_window % 2 == 0 or r0.sg_window <= r0.sg_poly:
        raise ConfigError(f"{source}: replay.r0.sg_window must be odd and > sg_poly")
    if int(r0.decimate) < 1:
        raise ConfigError(f"{source}: replay.r0.decimate must be >= 1")
    if r0.deflection not in _R0_DEFLECTIONS:
        raise ConfigError(f"{source}: replay.r0.deflection must be one of {', '.join(_R0_DEFLECTIONS)}")
    r1_raw = dict(raw.get("r1") or {})
    _check_keys(r1_raw, _R1_KEYS, "replay.r1", source)
    windows_raw = r1_raw.pop("windows", None)
    windows = None
    if windows_raw is not None:
        _check_keys(windows_raw, _R1_WINDOW_KEYS, "replay.r1.windows", source)
        if "length_s" not in windows_raw or "stride_s" not in windows_raw:
            raise ConfigError(f"{source}: replay.r1.windows needs length_s and stride_s")
        windows = R1Windows(**{k: float(v) for k, v in windows_raw.items()})
    r1 = R1Settings(**{**R1Settings().__dict__, **r1_raw, "windows": windows})
    r2_raw = raw.get("r2") or {}
    _check_keys(r2_raw, _R2_KEYS, "replay.r2", source)
    r2 = R2Settings(**{**R2Settings().__dict__, **r2_raw})
    if r2.reference not in _R2_REFERENCES:
        raise ConfigError(f"{source}: replay.r2.reference must be one of {', '.join(_R2_REFERENCES)}")
    if r2.emulation not in _R2_EMULATIONS:
        raise ConfigError(f"{source}: replay.r2.emulation must be one of {', '.join(_R2_EMULATIONS)}")
    return ReplaySettings(mode=mode, r0=r0, r1=r1, r2=r2)


def _name_list(raw: Mapping[str, Any], key: str, source: Path) -> tuple[str, ...]:
    values = raw.get(key) or []
    if isinstance(values, str) or not all(isinstance(v, str) for v in values):
        raise ConfigError(f"{source}: parameters.{key} must be a list of parameter names/wildcards")
    return tuple(values)


def _parameters(raw: Any, source: Path) -> ParameterSelection:
    _check_keys(raw, _PARAMETER_KEYS, "parameters", source)
    return ParameterSelection(free=_name_list(raw, "free", source), frozen=_name_list(raw, "frozen", source))


def _loss(raw: Any, source: Path) -> LossSettings:
    _check_keys(raw, _LOSS_KEYS, "loss", source)
    channels = raw.get("channels", {"ft": 1.0})
    _check_keys(channels, set(LOSS_CHANNELS), "loss.channels", source)
    weight = raw.get("prior_weight", "auto")
    if weight != "auto" and not isinstance(weight, (int, float)):
        raise ConfigError(f"{source}: loss.prior_weight must be 'auto' or a number")
    return LossSettings(channels={k: float(v) for k, v in channels.items()},
                        prior_weight=None if weight == "auto" else float(weight))


def _optimizer(raw: Any, source: Path) -> OptimizerSettings:
    _check_keys(raw, _OPTIMIZER_KEYS, "optimizer", source)
    defaults = OptimizerSettings()
    method = str(raw.get("method", defaults.method))
    if method not in OPTIMIZER_METHODS:
        raise ConfigError(f"{source}: optimizer.method must be one of {', '.join(OPTIMIZER_METHODS)}, "
                          f"got {method!r}")
    budget = int(raw.get("budget", defaults.budget))
    if budget < 1:
        raise ConfigError(f"{source}: optimizer.budget must be >= 1")
    blocks = {}
    for name, keys in (("cma", _CMA_KEYS), ("bo", _BO_KEYS), ("skrl_ppo", _SKRL_KEYS)):
        block = raw.get(name) or {}
        _check_keys(block, keys, f"optimizer.{name}", source)
        blocks[name] = {**getattr(defaults, name), **block}
    if blocks["bo"]["acq_func"] not in _BO_ACQUISITIONS:
        raise ConfigError(f"{source}: optimizer.bo.acq_func must be one of {', '.join(_BO_ACQUISITIONS)}")
    return OptimizerSettings(method=method, budget=budget, seed=int(raw.get("seed", defaults.seed)), **blocks)


def _jobs(raw: Any, source: Path) -> int | None:
    _check_keys(raw, _COMPUTE_KEYS, "compute", source)
    jobs = raw.get("jobs", "auto")
    if jobs == "auto":
        return None
    if not isinstance(jobs, int) or jobs < 1:
        raise ConfigError(f"{source}: compute.jobs must be 'auto' or a positive integer")
    return jobs
