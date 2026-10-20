"""Configuration schema ``elastic_sim.sim2real/1`` (S2R_01 §7).

A config may ``extends:`` another file (path relative to the extending
file). Mappings merge recursively, everything else (lists, scalars) is
replaced by the extending file. In ``parameters.priors`` the order is the
precedence, so a pattern the extending file re-declares moves to the end. Unknown keys fail with the offending block
and key named, in the ``dataset_config._check_keys`` style.

Relative data paths (``runs.*.path``, ``plant_base``, ``output``) resolve
against the repository root, like every other config in ``config/``.
"""

from __future__ import annotations

import dataclasses
import fnmatch
import re
import warnings
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
#: S2R_06 A-13: ``zero`` holds the test run's session-zero bags (its holds):
#: after the stage fit, only that run's sensor offsets are fitted on them.
SPLIT_KEYS = SPLITS + ("zero",)
SELECTIONS = ("val", "end_point")
LOSS_CHANNELS = ("ft", "tracking", "tau_cmd")

_TOP_KEYS = {"schema", "extends", "asset", "plant_base", "runs", "segments", "split", "replay",
             "parameters", "loss", "optimizer", "compute", "output", "prior_from", "selection", "identifiability"}
_RUN_KEYS = {"path", "sha256"}
_REPLAY_KEYS = {"mode", "r0", "r1", "r2"}
_R0_KEYS = {"sg_window", "sg_poly", "decimate", "deflection", "rows", "quasi_static_velocity",
            "quasi_static_acceleration", "hold_decimate"}
_R1_KEYS = {"track_bandwidth", "windows", "burn_in_s", "reference_filter", "extras"}
R1_REFERENCE_FILTERS = ("none", "savgol")
R1_EXTRAS = ("link_friction", "plant")
_R1_WINDOW_KEYS = {"length_s", "stride_s", "burn_in_s"}
_R2_KEYS = {"reference", "emulation"}
_PARAMETER_KEYS = {"free", "frozen", "priors"}
_LOSS_KEYS = {"channels", "prior_weight", "prior_weight_factor"}
_OPTIMIZER_KEYS = {"method", "budget", "seed", "cma", "bo", "skrl_ppo"}
_CMA_KEYS = {"sigma0", "popsize"}
_BO_KEYS = {"n_initial_points", "acq_func"}
_SKRL_KEYS = {"rollouts", "learning_epochs", "device"}
_COMPUTE_KEYS = {"jobs"}
_IDENTIFIABILITY_KEYS = {"d_post_min", "mode", "freeze_then_refit", "max_rounds", "partner_min"}
IDENTIFIABILITY_MODES = ("label", "freeze_refit")
_PRIOR_KEYS = {"mean", "sigma", "scale", "bounds", "factor", "offset", "values", "modes", "provenance"}

_R0_DEFLECTIONS = ("measured_ft", "none")
_R0_ROWS = ("all", "quasi_static")
#: S2R_04 A-10: the FRI command echo; ipo_position is the overlay base (the
#: held pose), never a tracking reference.
_R2_REFERENCES = ("commanded_position",)
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
    zero: tuple[str, ...] = ()

    def split_of(self, run_id: str, bag: str) -> str | None:
        """The split a bag belongs to, or ``None``; a bag in two splits is an error."""
        hits = [name for name in SPLIT_KEYS
                if any(_pattern_matches(pattern, run_id, bag) for pattern in getattr(self, name))]
        if len(hits) > 1:
            raise ConfigError(f"split: {run_id}:{bag} matches more than one split ({', '.join(hits)})")
        return hits[0] if hits else None

    def runs_of(self, split: str, run_ids: Any) -> tuple[str, ...]:
        """The runs (of ``run_ids``) some ``split`` pattern names."""
        return tuple(r for r in run_ids
                     if any(fnmatch.fnmatchcase(r, p.split(":", 1)[0]) for p in getattr(self, split)))


@dataclass(frozen=True)
class R0Settings:
    sg_window: int = 31
    sg_poly: int = 3
    decimate: int = 5
    #: ``measured_ft``: q_link = q - (ft - o) / (g K); ``none``: q_link = q.
    deflection: str = "measured_ft"
    #: ``quasi_static`` (S0, S2R_01 §9): keep holds and the excitation rows
    #: with every |dq| < quasi_static_velocity and |ddq| < quasi_static_acceleration.
    rows: str = "all"
    quasi_static_velocity: float = 0.05
    quasi_static_acceleration: float = 0.2
    #: Holds are static: 20 Hz carries all they have (S2R_05 note).
    hold_decimate: int = 50


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
    #: ``none``: the reference is the measured q (discrepancy-principle spline);
    #: ``savgol``: q pre-filtered by R0's SG (replay.r0.sg_window / sg_poly)
    #: first, so R1 imposes the motion R0 differentiates (gate G-3).
    reference_filter: str = "none"
    #: ``link_friction``: theta's link friction only; ``plant``: plus the
    #: round-6 nonlinear spring and transmission error (phase 0).
    extras: str = "link_friction"


@dataclass(frozen=True)
class R2Settings:
    reference: str = "commanded_position"
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
    #: pattern -> prior block (mean, sigma, scale, bounds|factor|offset|values,
    #: modes, provenance); resolved by ``parameters.Registry`` (T-2.1).
    priors: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class LossSettings:
    channels: Mapping[str, float] = field(default_factory=lambda: {"ft": 1.0})
    prior_weight: float | None = None  # None = "auto" (1 / residual rows per evaluation)
    #: Multiplies the auto weight (S2R_05: the twin shows auto is prior-dominated).
    prior_weight_factor: float = 1.0


@dataclass(frozen=True)
class IdentifiabilitySettings:
    """S2R_08 A-12' (calibrated is a label) / S2R_06 A-12 (freeze-then-refit),
    F-2 (the data share is d_post)."""

    #: A free parameter is data-determined (``calibrated: true``) when d_post >= this.
    d_post_min: float = 0.9
    #: ``label`` (S2R_08 A-12', default): one d_post on the full free set at
    #: the selected point labels each parameter; no value changes.
    #: ``freeze_refit`` (S2R_06 A-12, the owner's overrule): freeze the
    #: lowest-d_post parameter at its prior mean, repeat, refit.
    mode: str = "label"
    #: freeze_refit only: refits allowed when the refit point leaves a
    #: parameter below the threshold.
    max_rounds: int = 3
    #: S2R_08: |posterior correlation| from which a free parameter is listed
    #: as a not-calibrated parameter's partner.
    partner_min: float = 0.5

    @property
    def freeze_then_refit(self) -> bool:
        return self.mode == "freeze_refit"


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
    #: An overlay whose values become this stage's prior means (S2R_01 §9:
    #: the best theta of a stage is the prior mean of the next).
    prior_from: Path | None = None
    #: S2R_06 F-6: ``val`` (lowest val data loss) or ``end_point`` (the
    #: optimizer's own end point: a warm-start stage such as S0).
    selection: str = "val"
    identifiability: IdentifiabilitySettings = field(default_factory=IdentifiabilitySettings)


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


def _merge(base: Mapping[str, Any], override: Mapping[str, Any], path: tuple[str, ...] = ()) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if path == ("parameters", "priors"):
            # File order is precedence ("the last matching pattern wins"): an
            # overridden pattern moves to the end, child after base (S2R_04 R-1).
            # Its block still merges key by key with the base's.
            previous = merged.pop(key, None)
            merged[key] = _merge(previous, value, path + (key,)) \
                if isinstance(value, Mapping) and isinstance(previous, Mapping) else value
        elif isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _merge(merged[key], value, path + (str(key),))
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
        prior_from=_repo_path(raw["prior_from"]) if raw.get("prior_from") else None,
        selection=_selection(raw.get("selection", "val"), source),
        identifiability=_identifiability(raw.get("identifiability") or {}, source),
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
    _check_keys(raw, set(SPLIT_KEYS), "split", source)
    values = {}
    for name in SPLIT_KEYS:
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
    if int(r0.decimate) < 1 or int(r0.hold_decimate) < 1:
        raise ConfigError(f"{source}: replay.r0.decimate and hold_decimate must be >= 1")
    if r0.rows not in _R0_ROWS:
        raise ConfigError(f"{source}: replay.r0.rows must be one of {', '.join(_R0_ROWS)}")
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
    if r1.reference_filter not in R1_REFERENCE_FILTERS:
        raise ConfigError(f"{source}: replay.r1.reference_filter must be one of {', '.join(R1_REFERENCE_FILTERS)}")
    if r1.extras not in R1_EXTRAS:
        raise ConfigError(f"{source}: replay.r1.extras must be one of {', '.join(R1_EXTRAS)}")
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
    priors = raw.get("priors") or {}
    _check_keys(priors, set(priors) if isinstance(priors, Mapping) else set(), "parameters.priors", source)
    for pattern, block in priors.items():
        _check_keys(block, _PRIOR_KEYS, f"parameters.priors.{pattern}", source)
    return ParameterSelection(free=_name_list(raw, "free", source), frozen=_name_list(raw, "frozen", source),
                              priors={str(k): dict(v) for k, v in priors.items()})


def _loss(raw: Any, source: Path) -> LossSettings:
    _check_keys(raw, _LOSS_KEYS, "loss", source)
    channels = raw.get("channels", {"ft": 1.0})
    _check_keys(channels, set(LOSS_CHANNELS), "loss.channels", source)
    weight = raw.get("prior_weight", "auto")
    if weight != "auto" and not isinstance(weight, (int, float)):
        raise ConfigError(f"{source}: loss.prior_weight must be 'auto' or a number")
    factor = raw.get("prior_weight_factor", 1.0)
    if not isinstance(factor, (int, float)) or factor <= 0:
        raise ConfigError(f"{source}: loss.prior_weight_factor must be a positive number")
    if weight != "auto" and "prior_weight_factor" in raw:
        raise ConfigError(f"{source}: loss.prior_weight_factor scales the auto weight; drop it or set prior_weight: auto")
    return LossSettings(channels={k: float(v) for k, v in channels.items()},
                        prior_weight=None if weight == "auto" else float(weight), prior_weight_factor=float(factor))


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


def _selection(raw: Any, source: Path) -> str:
    if raw not in SELECTIONS:
        raise ConfigError(f"{source}: selection must be one of {', '.join(SELECTIONS)}, got {raw!r}")
    return str(raw)


def _identifiability(raw: Any, source: Path) -> IdentifiabilitySettings:
    _check_keys(raw, _IDENTIFIABILITY_KEYS, "identifiability", source)
    raw = dict(raw)
    if "freeze_then_refit" in raw:
        # S2R_08: the S2R_06 switch maps onto mode (deprecated). It wins over
        # a mode in the same merged mapping: after `extends` that mode is
        # usually the base config's default, not the stage's choice.
        legacy = raw.pop("freeze_then_refit")
        if not isinstance(legacy, bool):
            raise ConfigError(f"{source}: identifiability.freeze_then_refit must be true or false")
        raw["mode"] = "freeze_refit" if legacy else "label"
        warnings.warn(f"{source}: identifiability.freeze_then_refit is deprecated; read as mode: {raw['mode']} "
                      "(S2R_08)", DeprecationWarning, stacklevel=2)
    settings = IdentifiabilitySettings(**{**IdentifiabilitySettings().__dict__, **raw})
    if settings.mode not in IDENTIFIABILITY_MODES:
        raise ConfigError(f"{source}: identifiability.mode must be one of {', '.join(IDENTIFIABILITY_MODES)}")
    if not 0.0 < float(settings.d_post_min) < 1.0:
        raise ConfigError(f"{source}: identifiability.d_post_min must be in (0, 1)")
    if not 0.0 < float(settings.partner_min) < 1.0:
        raise ConfigError(f"{source}: identifiability.partner_min must be in (0, 1)")
    if not isinstance(settings.max_rounds, int) or settings.max_rounds < 1:
        raise ConfigError(f"{source}: identifiability.max_rounds must be a positive integer")
    return dataclasses.replace(settings, d_post_min=float(settings.d_post_min),
                               partner_min=float(settings.partner_min))


def _jobs(raw: Any, source: Path) -> int | None:
    _check_keys(raw, _COMPUTE_KEYS, "compute", source)
    jobs = raw.get("jobs", "auto")
    if jobs == "auto":
        return None
    if not isinstance(jobs, int) or jobs < 1:
        raise ConfigError(f"{source}: compute.jobs must be 'auto' or a positive integer")
    return jobs
