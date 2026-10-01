"""The real-data `/3` contract and baselines (RR_01 S8.2; RR_02 CR-4).

``dataset_io._round6_contract()`` (elastic_sim) hard-codes simulation-only
claims for every ``schema_version >= 3`` manifest (``target_instrument:
simulated_torque_measurement | simulator_ground_truth``, ``inputs.q.side:
motor`` unconditionally, ``controller.model_free: true``, a ``noise`` block).
Writing real data through it would publish false statements, so the real
converter writes the parquet + manifest with ``schema_version: 2`` (the
generic writer, which never calls ``_round6_contract``) and then this module
replaces ``<stem>.contract.json`` with its own ``/3`` document built from the
same keys plus the real-specific ones. RR_02 CR-4 asks the simulation side to
own this branch; until then, ``contract_owner`` says so explicitly so no
consumer mistakes this for an upstream-blessed contract shape.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

CONTRACT_SCHEMA = "elastic_sim.identification/3"
CONTRACT_OWNER_PENDING = "pending RR_02 CR-4"


def load_reference(path: str | Path, *, robot: str, joint_order: tuple[str, ...]) -> dict[str, Any]:
    reference = json.loads(Path(path).read_text())
    if reference.get("schema") != CONTRACT_SCHEMA or reference.get("n_dof") != len(joint_order):
        raise ValueError("reference contract schema or n_dof mismatch")
    if reference.get("joint_names") != list(joint_order):
        raise ValueError("reference contract joint order mismatch")
    if reference.get("robot", robot) != robot:
        raise ValueError("reference contract robot mismatch")
    return reference


def _round_to_odd(value: float) -> int:
    rounded = int(round(value))
    return rounded if rounded % 2 == 1 else rounded + 1


def differentiation_from_reference(
    reference_contract: Mapping[str, Any], *, rate_real: float, probe_top_real: float,
    reference_path: str | Path,
) -> dict[str, Any]:
    """RR_01 S8.2's differentiation rule, exactly: same rate as the training
    reference copies its window (``sg_poly``/``sg_window``); a different rate
    keeps the reference's **cutoff**, solving for the window that gives that
    cutoff at ``rate_real``
    (``W = odd(round(0.45 * rate_real / sg_cutoff_hz_ref))``, floored at
    ``sg_poly + 2``). This is the UR10 at 125 Hz against a 500 Hz reference;
    the iiwa at 1 kHz against an ``iiwa_drive`` reference is the same-rate
    case.

    ``reference_path`` is hashed into ``derived_from`` so the contract names
    exactly which training contract file produced this window.
    """
    ref_diff = reference_contract["differentiation"]
    ref_rate = float(ref_diff["rate"])
    sg_poly = int(ref_diff["sg_poly"])
    sg_cutoff_hz = float(ref_diff["sg_cutoff_hz"])

    if abs(rate_real - ref_rate) <= 1.0e-6:
        sg_window = int(ref_diff["sg_window"])
    else:
        sg_window = _round_to_odd(0.45 * rate_real / sg_cutoff_hz)
        floor = sg_poly + 2
        if sg_window < floor:
            sg_window = floor if floor % 2 == 1 else floor + 1

    digest = hashlib.sha256(Path(reference_path).read_bytes()).hexdigest()
    return {
        "rate": rate_real, "sg_window": sg_window, "sg_poly": sg_poly, "sg_cutoff_hz": sg_cutoff_hz,
        "probe_top_hz": probe_top_real, "derived_from": f"{Path(reference_path)}#sha256:{digest}",
    }


def _block(frame: pd.DataFrame, prefix: str, n: int) -> np.ndarray:
    return frame[[f"{prefix}{i}" for i in range(n)]].to_numpy(dtype=float)


#: Keys carried in each split's entry, exactly as
#: ``scripts/diagnose_controller_modes.py``'s ``_CONTRACT_BASELINES`` (RR_04
#: A-9). ``link_friction_share``/``noise_share``/``differentiation_share``
#: are always ``None`` on a real recording (no ground truth to separate them
#: from the model residual) but are still written so the key set matches.
_CONTRACT_BASELINE_COLUMNS = (
    "target_rms", "baseline_rms_mean", "baseline_rms_tau", "baseline_rms_state", "baseline_rms_state_tau",
    "baseline_rms_rigid",
)


def _per_bag_baselines(
    bag_frame: pd.DataFrame, *, pin: Any, model: Any, data: Any, basis: np.ndarray, idn: Any,
    n_dof: int, time_step: float, sg_window: int, sg_poly: int,
) -> dict[str, float]:
    q = _block(bag_frame, "q", n_dof)
    dq = _block(bag_frame, "dq", n_dof)
    tau = _block(bag_frame, "tau", n_dof)
    target = _block(bag_frame, "ft", n_dof)
    ddq = savgol_filter(dq, int(sg_window), int(sg_poly), deriv=1, delta=time_step, axis=0, mode="interp")

    regressor = idn.stack_regressor(pin, model, data, q, dq, ddq)
    ones = np.ones((len(regressor), 1))
    tau_column = tau.reshape(-1, 1)
    target_column = target.reshape(-1)

    def _fit_rms(design: np.ndarray) -> float:
        solution, *_ = np.linalg.lstsq(design, target_column, rcond=None)
        return float(np.sqrt(np.mean((design @ solution - target_column) ** 2)))

    projected = regressor @ basis
    rigid = np.asarray([idn.inverse_dynamics(pin, model, data, q[i], dq[i], ddq[i]) for i in range(len(q))])
    return {
        "target_rms": float(np.sqrt(np.mean(target**2))),
        "baseline_rms_mean": float(np.sqrt(np.mean((target_column - target_column.mean()) ** 2))),
        "baseline_rms_tau": _fit_rms(np.hstack([tau_column, ones])),
        "baseline_rms_state": _fit_rms(np.hstack([projected, ones])),
        "baseline_rms_state_tau": _fit_rms(np.hstack([projected, tau_column, ones])),
        "baseline_rms_rigid": float(np.sqrt(np.mean((target - rigid) ** 2))),
    }


def real_baselines(
    frame: pd.DataFrame, *, asset: Any, n_dof: int, time_step: float, sg_window: int, sg_poly: int = 3,
) -> dict[str, Any]:
    """T-2's linear baselines on the *recorded* state (RR_01 S8.2, RR_04 A-9).

    Mirrors ``elastic_sim.diagnostics.bag_diagnostics``'s T-2 block (same
    regressor, same basis, same four fits) on ``q``/``dq``/``tau``/``ft`` --
    the only state a real recording has -- instead of ``q_link``/``dq_link``
    (which do not exist here, and which ``bag_diagnostics`` itself requires,
    RR_04 A-9's finding). **Computed per bag, then the median across bags per
    split** (RR_04 A-9: the original single fit over the concatenated frame
    ran the SG derivative across bag boundaries, which
    ``scripts/diagnose_controller_modes.add_baselines_to_contract`` never
    does for the simulation side either), with keys exactly matching that
    function's ``_CONTRACT_BASELINES``, including ``target_rms`` and
    ``n_bags``. ``link_friction_share``/``noise_share`` are always ``None``:
    separating them from the model residual needs ground truth this file
    does not have.
    """
    from elastic_sim import identification as idn

    if "bag" not in frame.columns:
        raise ValueError("real_baselines: frame has no 'bag' column to compute per-bag statistics from")

    pin, model, data = idn.build_model(asset)
    basis = idn.base_parameter_basis(pin, model, data, n_samples=200, seed=0)

    bag_ids: list[Any] = []
    bag_splits: list[str] = []
    rows: list[dict[str, float]] = []
    for bag_id, bag_frame in frame.groupby("bag", sort=False):
        rows.append(_per_bag_baselines(bag_frame, pin=pin, model=model, data=data, basis=basis, idn=idn,
                                       n_dof=n_dof, time_step=time_step, sg_window=sg_window, sg_poly=sg_poly))
        bag_ids.append(bag_id)
        bag_splits.append(str(bag_frame["split"].iloc[0]) if "split" in bag_frame.columns else "test")
    table = pd.DataFrame(rows)

    splits: dict[str, dict[str, float | int | None]] = {}
    for split_name in ["all"] + sorted(set(bag_splits)):
        mask = np.ones(len(table), dtype=bool) if split_name == "all" \
            else np.asarray([s == split_name for s in bag_splits])
        subset = table[mask]
        entry: dict[str, float | int | None] = {"n_bags": int(len(subset))}
        for column in _CONTRACT_BASELINE_COLUMNS:
            value = float(np.nanmedian(subset[column].to_numpy(dtype=float))) if len(subset) else float("nan")
            entry[column] = value if np.isfinite(value) else None
        entry["link_friction_share"] = None
        entry["noise_share"] = None
        entry["differentiation_share"] = None
        splits[split_name] = entry

    return {
        "baseline_state_side": "recorded",
        "sg_window": int(sg_window),
        "statistic": "median over bags of the per-bag value (mirrors elastic_sim.diagnostics.bag_diagnostics's "
                     "T-2 block and scripts/diagnose_controller_modes.add_baselines_to_contract), computed on "
                     "the recorded state -- there is no separate link-side state on a real recording",
        "splits": splits,
    }


def real_contract(
    manifest: Mapping[str, Any],
    *,
    n_dof: int,
    target_instrument: str,
    target_semantics: str,
    q_side: str,
    dq_side: str,
    dq_source: str,
    tau_instrument: str,
    controller: Mapping[str, Any],
    differentiation: Mapping[str, Any],
    baselines: Mapping[str, Any],
    real_block: Mapping[str, Any],
    friction_split: str | None = None,
) -> dict[str, Any]:
    """Build the real-data ``/3`` contract document (RR_01 S8.2).

    Keeps every key the consumer (``dynamic_model_nn/dataset.py``,
    ``evaluate_on.py``) actually reads: ``inputs.tau`` (must declare
    ``commanded_motor_torque``/``noise_free: true``), ``target_kind:
    per_joint_torque``, ``differentiation`` (``sg_window``/``sg_poly``), and
    ``baselines.splits.test``. Everything else documents provenance for a
    human reader or a later CR-4 migration.

    ``baselines`` is :func:`real_baselines`'s return value (or an
    equivalent ``{"statistic": ..., "splits": {"all": {...}, "test": {...}}}``
    mapping for a synthetic/unit-test fixture): the whole per-split medians
    block is carried into the contract, with its statistic string (RR_01
    S8.2: "with the statistic string of add_baselines_to_contract"), not just
    the ``test`` split alone.
    """
    contract: dict[str, Any] = {
        "schema": CONTRACT_SCHEMA,
        "source": "real",
        "n_dof": n_dof,
        "input_columns": [f"q0..q{n_dof - 1}", f"dq0..dq{n_dof - 1}", f"tau0..tau{n_dof - 1}"],
        "target_columns": [f"ft0..ft{n_dof - 1}"],
        "target_kind": "per_joint_torque",
        "target_instrument": target_instrument,
        "target_semantics": target_semantics,
        "target_source": "measured",
        "target_clean_column": None,
        "inputs": {
            "q": {"side": q_side, "measured": True},
            "dq": {"side": dq_side, "measured": True, "source": dq_source},
            "tau": {"kind": "commanded_motor_torque", "noise_free": True, "instrument": tau_instrument},
        },
        "controller": dict(controller),
        "differentiation": dict(differentiation),
        "split": "test",
        "noise": None,
        "friction_split": friction_split,
        "baselines": {"statistic": baselines.get("statistic"), "splits": dict(baselines.get("splits", {}))},
        "real": dict(real_block),
        "contract_owner": CONTRACT_OWNER_PENDING,
        "requires_consumer": "dynamic_model_nn dataset.py with the general ft0..ft{dof-1} branch",
    }
    return contract


def write_real_contract(dataset_path: Path, contract: Mapping[str, Any]) -> Path:
    """Overwrite ``<stem>.contract.json`` written by ``write_dataset`` with
    the real contract (RR_02 CR-4: the ROS 2 side's interim writer)."""
    contract_path = Path(dataset_path).with_suffix(".contract.json")
    contract_path.write_text(json.dumps(dict(contract), indent=2), encoding="utf-8")
    return contract_path
