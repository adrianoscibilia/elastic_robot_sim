"""MuJoCo/Newton agreement on identical identification conditions.

Both backends receive the same trajectory, controller, friction and
transmission, so any disagreement between two bags that differ only by
backend is simulator disagreement.  The torque is closed-loop, so it is
compared too: a state difference feeds back into the recorded label.

A pair is only a genuine cross-check when Newton ran its own solver.  On
elastic bags Newton falls back to ``SolverMuJoCo`` (see the torque runners),
and the report says so rather than presenting agreement as independent
confirmation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

REFERENCE_BACKEND = "mujoco"


@dataclass(frozen=True)
class ComparisonThresholds:
    """Pass limits, applied to the worst joint of a pair."""

    q_link_rms: float = 1.0e-4          # rad
    ft_relative_rms: float = 0.01       # link torque difference / link torque RMS
    deflection_relative_rms: float = 0.05  # elastic bags only

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "ComparisonThresholds":
        raw = dict(raw or {})
        unknown = sorted(set(raw) - set(asdict(cls())))
        if unknown:
            raise ValueError(f"unknown comparison thresholds: {', '.join(unknown)}")
        return cls(**{key: float(value) for key, value in raw.items()})


def _pair_key(bag: str, backend: str) -> str:
    suffix = f"_{backend}"
    return bag[: -len(suffix)] if bag.endswith(suffix) else bag


def _rms(values: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(np.square(values), axis=0))


def _block(frame: pd.DataFrame, prefix: str, n_dof: int) -> np.ndarray:
    return frame[[f"{prefix}{index}" for index in range(n_dof)]].to_numpy(float)


def _n_dof(frame: pd.DataFrame) -> int:
    n_dof = 0
    while f"q{n_dof}" in frame.columns:
        n_dof += 1
    if n_dof == 0:
        raise ValueError("frame has no q0, q1, ... columns")
    return n_dof


def compare_backends(
    frame: pd.DataFrame,
    records: Iterable[Mapping[str, Any]] = (),
    thresholds: ComparisonThresholds | None = None,
    *,
    reference: str = REFERENCE_BACKEND,
) -> pd.DataFrame:
    """Return one row per bag pair that differs only by backend.

    ``records`` are the manifest's per-bag records; they are optional and only
    add the solver used.  Per-joint metrics are kept as ``<metric><joint>``
    columns next to the worst-joint value the pass decision uses.
    """
    thresholds = thresholds or ComparisonThresholds()
    solvers = {record["bag"]: record.get("solver") for record in records}
    n_dof = _n_dof(frame)
    groups: dict[str, dict[str, pd.DataFrame]] = {}
    for bag, rows in frame.groupby("bag", sort=False):
        backend = str(rows["backend"].iloc[0])
        groups.setdefault(_pair_key(str(bag), backend), {})[backend] = rows

    report = []
    for key, by_backend in groups.items():
        if reference not in by_backend:
            continue
        base = by_backend[reference]
        for backend, other in by_backend.items():
            if backend == reference:
                continue
            report.append(_compare_pair(key, base, other, reference, backend, solvers, n_dof, thresholds))
    return pd.DataFrame(report)


def _compare_pair(
    key: str, base: pd.DataFrame, other: pd.DataFrame, reference: str, backend: str,
    solvers: Mapping[str, Any], n_dof: int, thresholds: ComparisonThresholds,
) -> dict[str, Any]:
    # Bags share one resampling grid; a viewer closed early only shortens one.
    length = min(len(base), len(other))
    base, other = base.iloc[:length], other.iloc[:length]
    if not np.allclose(base["t"].to_numpy(float), other["t"].to_numpy(float), atol=1e-9):
        raise ValueError(f"bags of pair {key!r} are not on the same time grid")

    first = {prefix: _block(base, prefix, n_dof) for prefix in ("q", "dq", "q_motor", "ft", "tau")}
    second = {prefix: _block(other, prefix, n_dof) for prefix in first}
    delta = {prefix: first[prefix] - second[prefix] for prefix in first}
    deflection = first["q_motor"] - first["q"]
    deflection_diff = delta["q_motor"] - delta["q"]

    tier = str(base["tier"].iloc[0])
    # Rigid bags record q_motor as an exact copy of q.
    elastic = bool(np.abs(deflection).max() > 0.0)
    metrics = {
        "q_link_rms": _rms(delta["q"]),
        "q_link_max": np.abs(delta["q"]).max(axis=0),
        "dq_link_rms": _rms(delta["dq"]),
        "q_motor_rms": _rms(delta["q_motor"]),
        "ft_relative_rms": _rms(delta["ft"]) / np.maximum(_rms(first["ft"]), 1e-9),
        "tau_relative_rms": _rms(delta["tau"]) / np.maximum(_rms(first["tau"]), 1e-9),
        "deflection_relative_rms": (_rms(deflection_diff) / np.maximum(_rms(deflection), 1e-12)
                                    if elastic else np.full(n_dof, np.nan)),
    }
    other_bag = str(other["bag"].iloc[0])
    solver = solvers.get(other_bag)
    row: dict[str, Any] = {
        "pair": key, "tier": tier, "elastic": elastic,
        "reference_backend": reference, "backend": backend,
        "solver": solver,
        # SolverMuJoCo runs MuJoCo's engine; unknown when no manifest is given.
        "independent": None if solver is None else solver != "SolverMuJoCo",
        "samples": length,
    }
    for name, values in metrics.items():
        row[name] = float(np.nanmax(values)) if np.isfinite(values).any() else float("nan")
        if name == "q_link_rms":
            row["q_link_worst_joint"] = int(np.argmax(values))
    checks = [row["q_link_rms"] <= thresholds.q_link_rms, row["ft_relative_rms"] <= thresholds.ft_relative_rms]
    if elastic:
        checks.append(row["deflection_relative_rms"] <= thresholds.deflection_relative_rms)
    row["pass"] = bool(all(checks))
    for name in ("q_link_rms", "ft_relative_rms", "deflection_relative_rms"):
        for index, value in enumerate(metrics[name]):
            row[f"{name}{index}"] = float(value)
    return row


def summarize(report: pd.DataFrame, thresholds: ComparisonThresholds) -> dict[str, Any]:
    """Compact, JSON-serializable summary for the manifest."""
    if report.empty:
        return {"pairs": 0, "thresholds": asdict(thresholds)}
    return {
        "pairs": int(len(report)),
        "passed": int(report["pass"].sum()),
        "failed": [str(pair) for pair in report.loc[~report["pass"], "pair"]],
        "independent_pairs": int((report["independent"] == True).sum()),  # noqa: E712 - may hold None
        "worst_q_link_rms": float(report["q_link_rms"].max()),
        "worst_ft_relative_rms": float(report["ft_relative_rms"].max()),
        "worst_deflection_relative_rms": (float(report["deflection_relative_rms"].max())
                                          if report["elastic"].any() else None),
        "thresholds": asdict(thresholds),
    }


def format_report(report: pd.DataFrame, thresholds: ComparisonThresholds) -> str:
    """Human-readable table, one line per pair."""
    if report.empty:
        return "backend comparison: no bag pairs (needs mujoco and at least one other backend)"
    lines = [
        f"backend comparison (limits: q_link RMS {thresholds.q_link_rms:.1e} rad, "
        f"ft rel {thresholds.ft_relative_rms:.1%}, deflection rel {thresholds.deflection_relative_rms:.1%})",
        f"  {'pair':26s} {'solver':18s} {'q_link rms':>10s} {'jnt':>3s} {'ft rel':>8s} "
        f"{'tau rel':>8s} {'defl rel':>8s}  result",
    ]
    for row in report.to_dict("records"):
        deflection = "-" if not row["elastic"] else f"{row['deflection_relative_rms']:.2%}"
        note = "" if row["independent"] is not False else "  (same engine)"
        lines.append(
            f"  {row['pair']:26s} {str(row['solver'] or '?'):18s} {row['q_link_rms']:10.2e} "
            f"{row['q_link_worst_joint']:3d} {row['ft_relative_rms']:8.2%} {row['tau_relative_rms']:8.2%} "
            f"{deflection:>8s}  {'PASS' if row['pass'] else 'FAIL'}{note}"
        )
    passed = int(report["pass"].sum())
    lines.append(f"  {passed}/{len(report)} pairs within limits")
    return "\n".join(lines)
