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


#: RR_12 C-1: an achieved SG cutoff within this relative distance of the
#: training reference's counts as matched.
CUTOFF_MATCH_RTOL = 0.01


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_path(path: str | Path) -> str:
    """RR_12 C-3: ``path`` relative to ``$ERD_REPO_ROOT`` when it lies inside
    the repository (runs move between the laptop and peepo), else unchanged."""
    import os

    resolved = Path(path).expanduser().resolve()
    root = os.environ.get("ERD_REPO_ROOT")
    if root:
        try:
            return str(resolved.relative_to(Path(root).expanduser().resolve()))
        except ValueError:
            pass
    return str(resolved)


def reference_manifest_path(contract_path: str | Path) -> Path:
    """``<stem>.contract.json`` -> ``<stem>.manifest.json`` (the production layout)."""
    path = Path(contract_path)
    name = path.name
    if not name.endswith(".contract.json"):
        raise ValueError(f"reference contract {path} must be named <stem>.contract.json")
    return path.with_name(name[: -len(".contract.json")] + ".manifest.json")


def resolve_reference(config_path: str | Path, configured: str | None, configured_sha256: str | None,
                      cli_path: str | None = None, cli_sha256: str | None = None) -> tuple[str, str | None]:
    """RR_12 C-1: the reference contract a stage uses, and the sha256 it must
    have. A path on the command line replaces the configured one together with
    its pin: the configured sha256 only ever applies to the configured file."""
    if cli_path:
        path, expected = cli_path, cli_sha256
    else:
        path, expected = configured, configured_sha256
    if not path:
        raise ValueError("no --reference-contract given and consumer.reference_contract is null (RR_01 S8.2)")
    resolved = (Path(config_path).expanduser().resolve().parent / Path(path).expanduser()).resolve()
    return str(resolved), (expected.lower() if expected else None)


def compare_arm(reference_asset: str, sim_asset: str, *, tolerance: float = 1e-9) -> dict[str, Any]:
    """RR_12 C-1: the reference manifest's ``asset`` (a ``*_table`` variant)
    must carry the same arm as ``description.sim_asset``: joint names and
    types/axes, every joint placement but the first (the table only moves
    the base) and every link inertia. Needs Pinocchio (clean subprocess)."""
    from elastic_sim import identification as idn
    from elastic_sim.assets import AssetRegistry

    registry = AssetRegistry.for_repository()
    _, ref_model, _ = idn.build_model(registry.load(reference_asset))
    _, sim_model, _ = idn.build_model(registry.load(sim_asset))
    problems: list[str] = []
    ref_names = list(ref_model.names)[1:]
    sim_names = list(sim_model.names)[1:]
    if ref_names != sim_names:
        problems.append(f"joint names {ref_names} != {sim_names}")
    else:
        for index in range(1, sim_model.njoints):
            name = sim_model.names[index]
            ref_joint, sim_joint = ref_model.joints[index], sim_model.joints[index]
            if ref_joint.shortname() != sim_joint.shortname():
                problems.append(f"{name}: joint type {ref_joint.shortname()} != {sim_joint.shortname()}")
            elif hasattr(sim_joint, "axis") and np.abs(np.asarray(ref_joint.axis) - np.asarray(sim_joint.axis)).max() > tolerance:
                problems.append(f"{name}: axis differs")
            if ref_model.parents[index] != sim_model.parents[index]:
                problems.append(f"{name}: parent differs")
            if index > 1:
                placement = np.abs(ref_model.jointPlacements[index].homogeneous
                                   - sim_model.jointPlacements[index].homogeneous).max()
                if placement > tolerance:
                    problems.append(f"{name}: joint placement differs by {placement:.3g}")
            ref_inertia, sim_inertia = ref_model.inertias[index], sim_model.inertias[index]
            inertia = max(abs(ref_inertia.mass - sim_inertia.mass),
                          float(np.abs(ref_inertia.lever - sim_inertia.lever).max()),
                          float(np.abs(ref_inertia.inertia - sim_inertia.inertia).max()))
            if inertia > tolerance:
                problems.append(f"{name}: link inertia differs by {inertia:.3g}")
    return {"ok": not problems, "reference_asset": reference_asset, "sim_asset": sim_asset,
            "compared": "joint names, types/axes, placements of joints 2..n, link inertias", "problems": problems}


def load_reference(
    path: str | Path, *, robot: str, joint_order: tuple[str, ...], expected_sha256: str | None = None,
    hardware: str | None = None, sim_asset: str | None = None,
) -> dict[str, Any]:
    """Read a training reference contract (RR_01 S8.2; RR_12 C-1).

    Production contracts carry neither ``joint_names`` nor a rate: both come
    from the sibling ``<stem>.manifest.json`` (``joint_names``,
    ``sample_time_step``/``differentiation.sample_rate_hz``, ``asset``). The
    checked-in synthetic references have no manifest and keep both keys in
    the contract. Refuses: a sha256 other than ``expected_sha256``; a joint
    order other than ``joint_order``; ``hardware: real`` against a
    ``synthetic: true`` reference; a manifest asset whose arm differs from
    ``sim_asset`` (checked only when ``sim_asset`` is given).

    The returned dict is the contract plus ``_path``, ``_sha256``,
    ``_joint_names``, ``_rate_hz``, ``_asset`` and ``_arm_check``.
    """
    path = Path(path)
    digest = file_sha256(path)
    if expected_sha256 and digest != expected_sha256.lower():
        raise ValueError(f"reference contract {path} has sha256 {digest}, the config pins {expected_sha256}")
    reference = json.loads(path.read_text())
    if reference.get("schema") != CONTRACT_SCHEMA or reference.get("n_dof") != len(joint_order):
        raise ValueError("reference contract schema or n_dof mismatch")
    if reference.get("robot", robot) != robot:
        raise ValueError("reference contract robot mismatch")
    if hardware == "real" and reference.get("synthetic", False):
        raise ValueError(f"{path} is a synthetic reference; hardware: real needs the production contract (RR_12 C-1)")

    manifest_path = reference_manifest_path(path)
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
    elif not reference.get("synthetic", False):
        raise ValueError(f"reference manifest {manifest_path} is missing (copy it next to the contract, RR_12 S4.2)")
    joint_names = manifest.get("joint_names", reference.get("joint_names"))
    if joint_names != list(joint_order):
        raise ValueError("reference contract joint order mismatch")
    differentiation = reference.get("differentiation") or {}
    rate = differentiation.get("sample_rate_hz", differentiation.get("rate"))
    if rate is None and manifest.get("sample_time_step"):
        rate = 1.0 / float(manifest["sample_time_step"])
    if rate is None:
        raise ValueError(f"{path}: no differentiation.sample_rate_hz/rate and no manifest sample_time_step")

    arm_check = None
    asset = manifest.get("asset")
    if asset and sim_asset:
        arm_check = compare_arm(asset, sim_asset)
        if not arm_check["ok"]:
            raise ValueError(f"reference asset {asset!r} is not the arm of {sim_asset!r}: {arm_check['problems']}")

    result = dict(reference)
    result.update(_path=str(path), _sha256=digest, _joint_names=list(joint_names), _rate_hz=float(rate),
                  _asset=asset, _arm_check=arm_check)
    return result


def _round_to_odd(value: float) -> int:
    rounded = int(round(value))
    return rounded if rounded % 2 == 1 else rounded + 1


def sg_cutoff_hz(rate_hz: float, sg_window: int) -> float:
    """The order-3 SG differentiator's working cutoff, ``0.45 * rate / W``
    (the convention the production contracts use: 1000 Hz, W 5 -> 90 Hz)."""
    return 0.45 * float(rate_hz) / int(sg_window)


def differentiation_from_reference(
    reference_contract: Mapping[str, Any], *, rate_real: float, probe_top_real: float,
    reference_path: str | Path,
) -> dict[str, Any]:
    """RR_01 S8.2's differentiation rule: same rate as the training
    reference copies its window (``sg_poly``/``sg_window``); a different rate
    keeps the reference's **cutoff**, solving for the window that gives that
    cutoff at ``rate_real`` (``W = odd(round(0.45 * rate_real /
    sg_cutoff_hz_ref))``, floored at ``sg_poly + 2``).

    RR_12 C-1/F-5: the block uses the production key names, states the cutoff
    the window **achieves** at ``rate_real`` (``sg_cutoff_hz``) next to the
    reference's (``reference_cutoff_hz``), and whether they match
    (``cutoff_matched``): the UR10 at 125 Hz floors at W 5 = 11.25 Hz against
    a 45 Hz reference. ``derived_from`` is the repo-relative path plus the
    file's sha256 (RR_12 C-3), so the block is the same on every machine.
    """
    ref_diff = reference_contract["differentiation"]
    ref_rate = float(ref_diff.get("sample_rate_hz", ref_diff.get("rate", reference_contract.get("_rate_hz", 0.0))))
    sg_poly = int(ref_diff["sg_poly"])
    reference_cutoff = float(ref_diff["sg_cutoff_hz"])

    floored = False
    if abs(rate_real - ref_rate) <= 1.0e-6:
        sg_window = int(ref_diff["sg_window"])
    else:
        sg_window = _round_to_odd(0.45 * rate_real / reference_cutoff)
        floor = sg_poly + 2
        if sg_window < floor:
            sg_window = floor if floor % 2 == 1 else floor + 1
            floored = True
    achieved = sg_cutoff_hz(rate_real, sg_window)
    matched = abs(achieved - reference_cutoff) <= CUTOFF_MATCH_RTOL * reference_cutoff
    digest = file_sha256(reference_path)
    return {
        "sample_rate_hz": float(rate_real), "sg_window": sg_window, "sg_poly": sg_poly,
        "sg_cutoff_hz": achieved, "reference_cutoff_hz": reference_cutoff,
        "reference_sample_rate_hz": ref_rate, "cutoff_matched": bool(matched), "window_floored": floored,
        "probe_top_hz": probe_top_real,
        "derived_from": f"{portable_path(reference_path)}#sha256:{digest}",
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


#: RR_02 CR-4's ``source`` values: ``real`` only for ``hardware: real``; mock,
#: the FRI emulator and URSim all write ``synthetic_mock`` (RR_14 P-1).
CONTRACT_SOURCES = ("sim", "real", "synthetic_mock")


def contract_source(hardware: str) -> str:
    """The contract's ``source`` for a lab file's ``hardware`` value."""
    if hardware not in ("mock", "emulator", "ursim", "real"):
        raise ValueError(f"unknown hardware {hardware!r}")
    return "real" if hardware == "real" else "synthetic_mock"


def real_contract(
    manifest: Mapping[str, Any],
    *,
    hardware: str,
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

    ``source`` follows ``hardware`` (:func:`contract_source`), which the
    ``real`` block also records.
    """
    contract: dict[str, Any] = {
        "schema": CONTRACT_SCHEMA,
        "source": contract_source(hardware),
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
        "real": {"hardware": hardware, **dict(real_block)},
        "contract_owner": CONTRACT_OWNER_PENDING,
        "requires_consumer": "dynamic_model_nn dataset.py with the general ft0..ft{dof-1} branch",
    }
    return contract


def write_real_contract(dataset_path: Path, contract: Mapping[str, Any]) -> Path:
    """Overwrite ``<stem>.contract.json`` written by ``write_dataset`` with
    the real contract (RR_02 CR-4: the ROS 2 side's interim writer)."""
    contract_path = Path(dataset_path).with_suffix(".contract.json")
    contract_path.write_text(json.dumps(dict(contract), indent=2,
                                        default=lambda v: v.item() if isinstance(v, np.generic) else str(v)),
                             encoding="utf-8")
    return contract_path
