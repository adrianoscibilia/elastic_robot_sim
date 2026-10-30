"""Loading the producer's files for the identification and the predictors.

Simulated files carry the recorded channels (``q``, ``dq``, ``ft``: one
sample late, `00_recon.md` Sec 1.2) and the privileged ones the
identification may use (``q_link``, ``dq_link``: link truth; ``tau``: the
command), which sit on the true grid.  :func:`load_sim` delays the
privileged channels by the contract's ``noise.delay_samples`` so every
channel of a bag is on the recorded grid, then drops the first
``delay`` rows.  The predictors never see the privileged channels (except
ID-inverse, a simulation-only diagnostic).

Real files carry ``q``, ``dq``, ``ft`` only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

TRUTH = {  # ground-truth group -> per-joint metadata column prefix
    "stiffness": "stiffness__", "damping": "damping__", "rotor": "rotor_inertia__",
    "motor_viscous": "viscous__", "motor_coulomb": "coulomb__",
    "link_viscous": "link_viscous__", "link_coulomb": "link_coulomb__",
}


@dataclass
class Bag:
    name: str
    tier: str
    split: str
    t: np.ndarray
    theta: np.ndarray              # recorded motor position q*
    dtheta: np.ndarray             # recorded motor velocity dq*
    tau_J: np.ndarray              # recorded target ft*
    u: np.ndarray | None = None    # commanded motor torque tau* (privileged)
    q: np.ndarray | None = None    # link position q_link* (privileged)
    dq: np.ndarray | None = None   # link velocity dq_link* (privileged)
    meta: dict[str, Any] = field(default_factory=dict)
    # Exact signals (theta, dtheta, ddtheta, q, dq, ddq, tau_J, u) from a simulator that has them
    # (:mod:`.synthetic`); the identification then skips filtering and differentiation.
    exact: dict[str, np.ndarray] | None = None

    @property
    def step(self) -> float:
        return float(np.median(np.diff(self.t)))


def sidecar(path: str | Path, kind: str) -> dict[str, Any]:
    path = Path(path)
    return json.loads(path.with_suffix(f".{kind}.json").read_text(encoding="utf-8"))


def _block(frame: Any, prefix: str, n: int, dtype: Any = float) -> np.ndarray:
    return frame[[f"{prefix}{i}" for i in range(n)]].to_numpy(dtype=dtype).astype(float)


def _delayed(x: np.ndarray, delay: int) -> np.ndarray:
    """``x`` seen ``delay`` samples late: row k holds row k - delay (first rows dropped by the caller)."""
    if delay == 0:
        return x
    out = np.empty_like(x)
    out[delay:] = x[:-delay]
    out[:delay] = x[0]
    return out


def load_sim(path: str | Path, *, splits: Sequence[str] = ("train",), privileged: bool = True,
             truth: bool = True, bags: Iterable[str] | None = None) -> tuple[list[Bag], dict[str, Any]]:
    """The bags of ``splits`` (or the named ``bags``), channels aligned on the recorded grid.

    Returns ``(bags, info)``; ``info`` holds the contract, the joint names, the
    delay applied and, with ``truth``, the per-robot ground truth
    (:func:`ground_truth`).
    """
    import pyarrow.parquet as pq

    path = Path(path)
    contract = sidecar(path, "contract")
    manifest = sidecar(path, "manifest")
    n = int(contract["n_dof"])
    names = list(manifest["joint_names"])
    delay = int((contract.get("noise") or {}).get("delay_samples") or 0)
    columns = ["t", "bag", "tier", "split"] + [f"{p}{i}" for p in ("q", "dq", "ft") for i in range(n)]
    if privileged:
        columns += [f"{p}{i}" for p in ("tau", "q_link", "dq_link") for i in range(n)]
    if truth:
        columns += [f"{p}{i}" for p in ("q_motor_clean", "q_link_clean", "dq_motor_clean", "dq_link_clean",
                                        "tau_link_clean") for i in range(n)]
        columns += [f"{prefix}{name}" for prefix in TRUTH.values() for name in names]
    schema = set(pq.read_schema(path).names)
    columns = [c for c in columns if c in schema]
    filters = [("bag", "in", list(bags))] if bags is not None else [("split", "in", list(splits))]
    frame = pq.read_table(path, columns=columns, filters=filters).to_pandas()
    if frame.empty:
        raise ValueError(f"{path}: no rows for {bags if bags is not None else splits}")
    out: list[Bag] = []
    clean: dict[str, list[dict[str, np.ndarray]]] = {}
    tiers: dict[str, Any] = {}
    for name, group in frame.groupby("bag", sort=False):
        tier = str(group["tier"].iloc[0])
        keep = slice(delay, None)
        bag = Bag(str(name), tier, str(group["split"].iloc[0]), group["t"].to_numpy(float)[keep],
                  _block(group, "q", n)[keep], _block(group, "dq", n)[keep], _block(group, "ft", n)[keep])
        if privileged:
            bag.u = _delayed(_block(group, "tau", n), delay)[keep]
            bag.q = _delayed(_block(group, "q_link", n), delay)[keep]
            bag.dq = _delayed(_block(group, "dq_link", n), delay)[keep]
        if truth and tier != "rigid":
            tiers.setdefault(tier, group.iloc[0])
            clean.setdefault(tier, []).append({
                "delta": _block(group, "q_motor_clean", n) - _block(group, "q_link_clean", n),
                "ddelta": _block(group, "dq_motor_clean", n) - _block(group, "dq_link_clean", n),
                "tau": _block(group, "tau_link_clean", n)})
        out.append(bag)
    info = {"path": str(path), "contract": contract, "manifest": manifest, "n": n, "joint_names": names,
            "delay_samples": delay, "step": float(manifest["sample_time_step"])}
    if truth:
        info["truth"] = ground_truth(tiers, clean, names)
    return out, info


def ground_truth(rows: dict[str, Any], clean: dict[str, list[dict[str, np.ndarray]]],
                 names: Sequence[str]) -> dict[str, dict[str, list[float]]]:
    """Per robot: the drawn parameters, and the best linear spring on its clean signals.

    ``stiffness_effective`` / ``damping_effective`` are the per-joint
    least-squares fit of the clean transmission torque on the true deflection
    and its rate, over the robot's bags: what a linear spring identified
    without noise can at best return, given the simulator's nonlinear spring
    and transmission error (`00_recon.md` Sec 2).
    """
    out: dict[str, dict[str, list[float]]] = {}
    for tier, row in rows.items():
        entry = {group: [float(row[f"{prefix}{name}"]) for name in names] for group, prefix in TRUTH.items()
                 if all(f"{prefix}{name}" in row.index for name in names)}
        segments = clean.get(tier, [])
        if segments:
            delta = np.vstack([s["delta"] for s in segments])
            ddelta = np.vstack([s["ddelta"] for s in segments])
            tau = np.vstack([s["tau"] for s in segments])
            k_eff, d_eff = [], []
            for j in range(delta.shape[1]):
                (k, d), *_ = np.linalg.lstsq(np.column_stack([delta[:, j], ddelta[:, j]]), tau[:, j], rcond=None)
                k_eff.append(float(k))
                d_eff.append(float(d))
            entry["stiffness_effective"], entry["damping_effective"] = k_eff, d_eff
        out[tier] = entry
    return out


def load_real(path: str | Path, *, float32: bool = True) -> tuple[list[Bag], dict[str, Any]]:
    """A real file's bags (``q``, ``dq``, ``ft``); ``float32`` reads like the consumer (torch) does."""
    import pyarrow.parquet as pq

    path = Path(path)
    contract = sidecar(path, "contract")
    n = int(contract["n_dof"])
    frame = pq.read_table(path, columns=["t", "bag"] + [f"{p}{i}" for p in ("q", "dq", "ft")
                                                         for i in range(n)]).to_pandas()
    dtype = np.float32 if float32 else float
    out = []
    for name, group in frame.groupby("bag", sort=False):
        out.append(Bag(str(name), "real", "test", group["t"].to_numpy(float), _block(group, "q", n, dtype),
                       _block(group, "dq", n, dtype), _block(group, "ft", n, dtype)))
    return out, {"path": str(path), "contract": contract, "n": n}
