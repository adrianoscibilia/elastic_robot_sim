"""The identification pipeline: configuration, rows, base, solve, recovery.

One run reads the simulated training split (`01_zollo_plan.md` Sec 2 rule
1), builds Zollo's rows bag by bag (:mod:`.signals`, :mod:`.regressor`),
folds them into one triangular factor (:mod:`.basis`), selects the base
parameters and solves for them.  Two variants (plan Sec 5.5):

* ``A`` (Zollo-faithful): link and motor rows, target ``[0; u]``; uses the
  link truth and the command, both privileged;
* ``B`` (tau_J-supervised): link rows on ``tau_J`` and spring rows
  ``tau_J = K delta + D ddelta``; uses the link truth, privileged.

``transmission: pooled`` gives one K, D (and B, motor friction in A) per
joint over every training robot -- the parameter set the predictors use;
``per_robot`` gives each robot its own, for the recovery check (gate G2);
``per_robot_motor`` (variant A only: "A-pr", `04_variant_A_elastic.md`)
gives each robot its own rotor inertia and motor friction -- the nuisance
parameters of the motor rows -- and pools everything else, so the link
model, K and D still predict unseen robots;
``per_robot_transmission`` (variant A only: "A-pc", `06_variant_A_consistent.md`)
gives each robot every parameter the simulator draws per robot (K, D, rotor
inertia, motor friction), pools the link inertials and link friction, and by
default constrains the solve to physically consistent link inertials
(:mod:`.consistent`); its predictors take the median of the per-robot K, D.
``consistent`` switches that constraint on or off for any transmission (the
A-pc ablations).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .basis import Accumulator, Base, Solution, select_base, solve
from .data import Bag
from .regressor import MOTOR_GROUPS, TRANSMISSION_GROUPS, Layout, nominal_vector, rigid_regressor, sample_rows
from .signals import Preprocessing, kept_samples, low_speed_mask, parallel, prepare

TRANSMISSIONS = ("pooled", "per_robot", "per_robot_motor", "per_robot_transmission")


@dataclass(frozen=True)
class IdentificationConfig:
    variant: str = "B"
    transmission: str = "pooled"
    include_rigid_tier: bool = True
    link_friction: bool = True
    spring_damping: bool = True
    coupling_terms: bool = False
    epsilon_link: float = 1.0e-3
    epsilon_motor: float = 1.0e-2
    rank_tolerance: float = 1.0e-6
    weighted: bool = False
    ridge_to_nominal: float = 0.0
    consistent: bool | None = None           # C2; None: on for per_robot_transmission only
    consistency_epsilon: float = 1.0e-6      # J_i >= eps I, units of J (kg m^2, kg m, kg)
    consistency_solver: str = "CLARABEL"
    preprocessing: Preprocessing = field(default_factory=lambda: Preprocessing(cutoff_hz=None))

    def __post_init__(self) -> None:
        if self.transmission not in TRANSMISSIONS:
            raise ValueError(f"transmission must be one of {TRANSMISSIONS}")
        if self.transmission == "per_robot_motor" and self.variant != "A":
            raise ValueError("transmission per_robot_motor needs variant A (only its motor rows carry motor groups)")
        if self.transmission == "per_robot_transmission" and self.variant != "A":
            raise ValueError("transmission per_robot_transmission needs variant A (A-pc, 06_variant_A_consistent.md)")
        if self.is_consistent and self.ridge_to_nominal > 0.0:
            raise ValueError("the consistency constraint has no prior: ridge_to_nominal must be 0")
        if self.coupling_terms:
            # The simulator is Spong's reduced model (rotor = armature on the motor DOF,
            # 00_recon.md Sec 2): every coupling column's true coefficient is zero.
            raise NotImplementedError("coupling_terms: the simulated plant has no rotor-link coupling to identify")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any], preprocessing: Mapping[str, Any]) -> "IdentificationConfig":
        known = {f for f in cls.__dataclass_fields__ if f != "preprocessing"}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"unknown identification keys {sorted(unknown)}")
        return cls(**dict(values), preprocessing=Preprocessing(**dict(preprocessing)))

    def describe(self) -> dict[str, Any]:
        out = asdict(self)
        out["preprocessing"] = self.preprocessing.describe()
        return out

    @property
    def per_robot_groups(self) -> tuple[str, ...]:
        """The groups that get one column per robot (none when pooled)."""
        return {"pooled": (), "per_robot": TRANSMISSION_GROUPS, "per_robot_motor": MOTOR_GROUPS,
                "per_robot_transmission": TRANSMISSION_GROUPS}[self.transmission]

    @property
    def is_consistent(self) -> bool:
        """Whether the solve is constrained to physically consistent inertials (C2)."""
        return self.transmission == "per_robot_transmission" if self.consistent is None else bool(self.consistent)

    @property
    def pooled_predictor(self) -> bool:
        """Whether the predictors can be built: K, D pooled, or per robot with the C3 median (A-pc)."""
        return self.transmission in ("pooled", "per_robot_motor", "per_robot_transmission")

    @property
    def label(self) -> str:
        """The output name: ``<variant>_<transmission>``, A-pr as ``Apr_pooled``, A-pc as ``Apc_pooled``.

        (A-pr, A-pc: their predictors' parameters are pooled.)  A solve that
        departs from its transmission's default constraint says so:
        ``_C2`` (constrained), ``_noC2`` (A-pc unconstrained).
        """
        if self.transmission == "per_robot_motor":
            label = f"{self.variant}pr_pooled"
        elif self.transmission == "per_robot_transmission":
            label = f"{self.variant}pc_pooled"
        else:
            label = f"{self.variant}_{self.transmission}"
        default = self.transmission == "per_robot_transmission"
        if self.is_consistent != default:
            label += "_C2" if self.is_consistent else "_noC2"
        return label


@dataclass
class RowStats:
    bags: int = 0
    samples: int = 0
    rows: int = 0
    rows_dropped_low_speed: list[int] = field(default_factory=list)

    def describe(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Identification:
    """The result: the layout, the base, the solution and how they were obtained."""

    layout: Layout
    base: Base
    solution: Solution
    config: IdentificationConfig
    stats: RowStats
    nominal: np.ndarray                      # full vector: URDF inertials + nominal/true-mean values
    meta: dict[str, Any] = field(default_factory=dict)
    row_residual_rms: dict[str, float] = field(default_factory=dict)
    accumulator: Accumulator | None = field(default=None, repr=False)    # not saved; see save_factor

    @property
    def full(self) -> np.ndarray:
        """A full parameter vector equal to the identified one on every row.

        The constrained solve's own full vector (consistent body by body);
        otherwise dependent columns 0.
        """
        if self.solution.full is not None:
            return self.solution.full
        return self.base.full_of(self.solution.base, self.layout.size)

    def value(self, group: str, joint: int, robot: str | None = None) -> float:
        return float(self.full[self.layout.index(group, joint, robot)])

    def to_json(self) -> dict[str, Any]:
        names = self.layout.names
        return {
            "method": "Zollo et al. 2015 elastic-joint identification (REFACTOR_SPECS/model_based_baseline)",
            "config": self.config.describe(), "layout": self.layout.describe(), "stats": self.stats.describe(),
            "base": self.base.describe(names),
            "independent": self.base.independent.tolist(), "dependent": self.base.dependent.tolist(),
            "zero": self.base.zero.tolist(), "relation": self.base.relation.tolist(),
            "column_rms": self.base.column_rms.tolist(),
            "solution": {"base": self.solution.base.tolist(), "std": self.solution.std.tolist(),
                         "rss": self.solution.rss, "rows": self.solution.rows,
                         "residual_rms": self.solution.residual_rms, "ridge": self.solution.ridge,
                         "full": None if self.solution.full is None else self.solution.full.tolist()},
            "row_residual_rms": self.row_residual_rms,
            "nominal": [None if not np.isfinite(v) else float(v) for v in self.nominal],
            "meta": self.meta,
        }

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_json(), indent=1), encoding="utf-8")

    def save_factor(self, path: str | Path) -> None:
        """The triangular factor ``[R_W | z]`` and its row count: re-solve without re-reading the bags."""
        if self.accumulator is None:
            raise ValueError("no accumulator attached")
        np.savez_compressed(path, R=self.accumulator.R, rows=self.accumulator.rows)


def load_factor(path: str | Path) -> Accumulator:
    record = np.load(path)
    acc = Accumulator(record["R"].shape[1] - 1)
    acc.R, acc.rows = record["R"], int(record["rows"])
    return acc


def load_identification(path: str | Path) -> Identification:
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    lay = record["layout"]
    layout = Layout.build(lay["n"], lay["variant"], link_friction=lay["link_friction"],
                          spring_damping=lay["spring_damping"], robots=lay["robots"],
                          per_robot_groups=lay["per_robot_groups"])
    if layout.names != lay["columns"]:
        raise ValueError(f"{path}: the layout no longer rebuilds to the saved columns")
    names = layout.names
    categories = [record["base"]["categories"][name] for name in names]
    base = Base(np.asarray(record["independent"], dtype=int), np.asarray(record["dependent"], dtype=int),
                np.asarray(record["zero"], dtype=int),
                np.asarray(record["relation"], dtype=float).reshape(len(record["independent"]), -1),
                categories, np.asarray(record["column_rms"], dtype=float),
                record["base"]["condition_scaled"], record["base"]["condition_raw"], record["base"]["tolerance"])
    sol = record["solution"]
    solution = Solution(np.asarray(sol["base"]), np.asarray(sol["std"]), sol["rss"], sol["rows"],
                        sol["residual_rms"], sol["ridge"],
                        None if sol.get("full") is None else np.asarray(sol["full"], dtype=float))
    cfg = record["config"]
    config = IdentificationConfig.from_mapping({k: v for k, v in cfg.items() if k != "preprocessing"},
                                               cfg["preprocessing"])
    nominal = np.asarray([np.nan if v is None else v for v in record["nominal"]], dtype=float)
    return Identification(layout, base, solution, config, RowStats(**record["stats"]), nominal,
                          record.get("meta", {}), record.get("row_residual_rms", {}))


def bag_rows(bag: Bag, layout: Layout, config: IdentificationConfig, rigid_model: tuple,
             stats: RowStats | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(W [m, p], columns [p], target [m], group [m])`` of one bag, after preprocessing.

    ``group`` is the row's equation index in ``0 .. 2n-1``
    (:meth:`Layout.row_labels`), for the per-equation residuals and weights.
    """
    if bag.q is None or (layout.variant == "A" and bag.u is None):
        raise ValueError(f"bag {bag.name}: the identification needs the privileged channels")
    pre = config.preprocessing
    n = layout.n
    step = bag.step
    raw = {"theta": bag.theta, "q": bag.q, "tau_J": bag.tau_J, "u": bag.u}
    sig = bag.exact if bag.exact is not None else prepare(raw, step, pre)
    Y = rigid_regressor(*rigid_model, sig["q"], sig["dq"], sig["ddq"])
    W, columns, target, valid = sample_rows(layout, bag.tier, sig, Y, epsilon_link=config.epsilon_link,
                                            epsilon_motor=config.epsilon_motor)
    W, target = parallel(W, target, step, pre)
    keep = kept_samples(len(W), step, pre)
    W, target, valid = W[keep], target[keep], valid[keep]
    fast = low_speed_mask(sig["dtheta"][keep], pre)
    before = valid.copy()
    if layout.link_friction:
        valid[:, :n] &= fast
    if layout.variant == "A":
        valid[:, n:] &= fast
    if stats is not None:
        stats.bags += 1
        stats.samples += len(keep)
        stats.rows += int(valid.sum())
        dropped = (before & ~valid).reshape(len(keep), 2, n).sum(axis=(0, 1))
        stats.rows_dropped_low_speed = (np.add(stats.rows_dropped_low_speed, dropped).tolist()
                                        if stats.rows_dropped_low_speed else dropped.tolist())
    flat = valid.reshape(-1)
    group = np.tile(np.arange(2 * n), len(keep))
    return W.reshape(-1, W.shape[-1])[flat], columns, target.reshape(-1)[flat], group[flat]


def _usable(bags: Iterable[Bag], config: IdentificationConfig) -> list[Bag]:
    out = []
    for bag in bags:
        if bag.tier == "rigid" and (config.variant == "A" or not config.include_rigid_tier):
            continue
        out.append(bag)
    return out


def identify(bags: Sequence[Bag], config: IdentificationConfig, rigid_model: tuple, *,
             truth: Mapping[str, Mapping[str, Sequence[float]]] | None = None,
             standard_inertial: np.ndarray | None = None, verbose: bool = False) -> Identification:
    """Identify one parameter set from ``bags`` (privileged channels required).

    ``rigid_model`` is ``(pin, model, data)``.  ``truth`` (per robot, from
    :func:`.data.load_sim`) and ``standard_inertial`` (the URDF's 10n vector)
    build the nominal vector the ridge prior and the recovery tables use.
    """
    bags = _usable(bags, config)
    if not bags:
        raise ValueError("no usable bags")
    pin, model, _ = rigid_model
    n = model.nv
    robots = sorted({b.tier for b in bags if b.tier != "rigid"})
    layout = Layout.build(n, config.variant, link_friction=config.link_friction,
                          spring_damping=config.spring_damping,
                          robots=robots if config.per_robot_groups else None,
                          per_robot_groups=config.per_robot_groups or TRANSMISSION_GROUPS)
    if standard_inertial is None:
        standard_inertial = np.concatenate([model.inertias[i].toDynamicParameters() for i in range(1, model.njoints)])
    groups = {group: {r: truth[r][group] for r in robots if r in truth and group in truth[r]}
              for group in ("stiffness", "damping", "rotor", "motor_viscous", "motor_coulomb",
                            "link_viscous", "link_coulomb")} if truth else {}
    nominal = nominal_vector(layout, standard_inertial, {g: v for g, v in groups.items() if v})

    def accumulate(weights: np.ndarray | None, stats: RowStats | None) -> Accumulator:
        acc = Accumulator(layout.size)
        for index, bag in enumerate(bags):
            W, columns, target, group = bag_rows(bag, layout, config, rigid_model, stats)
            acc.add(W, target, columns, None if weights is None else weights[group])
            if verbose:
                print(f"  [{index + 1}/{len(bags)}] {bag.name}: {len(target)} rows", flush=True)
        return acc

    stats = RowStats()
    acc = accumulate(None, stats)
    base = select_base(acc, tolerance=config.rank_tolerance)
    prior = base.base_of(nominal) if config.ridge_to_nominal > 0.0 else None
    consistency: dict[str, Any] = {}
    solution = _solve(acc, base, layout, config, prior, consistency)
    result = Identification(layout, base, solution, config, stats, nominal, accumulator=acc)
    result.row_residual_rms = row_residuals(bags, result, rigid_model)
    if config.weighted:
        sigma = np.asarray([result.row_residual_rms[label] for label in layout.row_labels()])
        weights = 1.0 / np.where(sigma > 0.0, sigma, np.inf)
        acc = accumulate(weights, None)
        result.solution = _solve(acc, base, layout, config, prior, consistency)
        result.accumulator = acc
        result.row_residual_rms = row_residuals(bags, result, rigid_model)
    result.meta = {"robots": robots, "bags": [b.name for b in bags], "n_dof": n}
    if consistency:
        result.meta["consistency"] = consistency
    return result


def _solve(acc: Accumulator, base: Base, layout: Layout, config: IdentificationConfig, prior: np.ndarray | None,
           report: dict[str, Any]) -> Solution:
    """Zollo's LS (eq. 42), or with C2 the same cost constrained to consistent inertials (``report`` filled)."""
    unconstrained = solve(acc, base, ridge=config.ridge_to_nominal, prior=prior)
    if not config.is_consistent:
        return unconstrained
    from .consistent import solve_consistent

    fit = solve_consistent(acc, base, layout, epsilon=config.consistency_epsilon, solver=config.consistency_solver)
    report.clear()
    report.update(fit.report)
    # The std stays the unconstrained LS covariance (labelled); the base vector is the constrained one's.
    return Solution(base.base_of(fit.full), unconstrained.std, fit.cost, acc.rows, float(np.sqrt(fit.cost / acc.rows)),
                    0.0, fit.full)


def row_residuals(bags: Sequence[Bag], result: Identification, rigid_model: tuple) -> dict[str, float]:
    """RMS of ``W Pi* - T`` per equation (link[j], motor[j] or spring[j]) over the identification rows."""
    full = result.full
    labels = result.layout.row_labels()
    sse = np.zeros(len(labels))
    count = np.zeros(len(labels))
    for bag in bags:
        W, columns, target, group = bag_rows(bag, result.layout, result.config, rigid_model)
        error = W @ full[columns] - target
        np.add.at(sse, group, error ** 2)
        np.add.at(count, group, 1)
    return {label: float(np.sqrt(s / c)) if c else float("nan") for label, s, c in zip(labels, sse, count)}


def recovery_table(result: Identification, truth: Mapping[str, Mapping[str, Sequence[float]]] | None = None,
                   ) -> list[dict[str, Any]]:
    """Base parameters against their true values (Zollo Tables 1, 3-6 style).

    The true base vector is the true full vector folded through the same
    relation; for a pooled transmission column the "true" value is the mean
    over the identification robots, and ``true_min``/``true_max`` and the
    effective linear spring (``stiffness_effective``) are given beside it.
    """
    names = result.layout.names
    true_base = result.base.base_of(result.nominal)
    rows = []
    robots = result.meta.get("robots", [])
    for i, position in enumerate(result.base.independent):
        column = result.layout.columns[position]
        value, std = float(result.solution.base[i]), float(result.solution.std[i])
        true = float(true_base[i])
        row = {"base": i, "column": names[position], "group": column.group, "joint": column.joint + 1,
               "robot": column.robot, "category": result.base.categories[position],
               "expression": result.base.expression(i, names), "identified": value, "std": std,
               "std_pct": 100.0 * std / abs(value) if value else float("nan"), "true": true,
               "error_pct": 100.0 * (value - true) / abs(true) if np.isfinite(true) and true else float("nan")}
        if truth and column.group in ("stiffness", "damping", "rotor", "motor_viscous", "motor_coulomb"):
            owners = [column.robot] if column.robot else robots
            values = [truth[r][column.group][column.joint] for r in owners if r in truth]
            if values:
                row["true_min"], row["true_max"] = float(np.min(values)), float(np.max(values))
            effective = f"{column.group}_effective"
            eff = [truth[r][effective][column.joint] for r in owners if r in truth and effective in truth[r]]
            if eff:
                row["true_effective"] = float(np.mean(eff))
                row["error_effective_pct"] = 100.0 * (value - row["true_effective"]) / abs(row["true_effective"])
        rows.append(row)
    return rows


def with_overrides(config: IdentificationConfig, **changes: Any) -> IdentificationConfig:
    """``config`` with top-level and ``preprocessing.*`` fields replaced (``None`` values ignored)."""
    pre = {k.split(".", 1)[1]: v for k, v in changes.items() if k.startswith("preprocessing.") and v is not None}
    top = {k: v for k, v in changes.items() if not k.startswith("preprocessing.") and v is not None}
    if pre:
        top["preprocessing"] = replace(config.preprocessing, **pre)
    return replace(config, **top)
