"""Write an identification dataset: the flat file, its manifest and the consumer contract.

Split out of ``dataset.py`` (`R5_10` T-7) with no behaviour change; import
from :mod:`elastic_sim.dataset`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .dataset_config import _TARGET_SEMANTICS


def _target_contains(manifest: Mapping[str, Any]) -> str:
    """One sentence on what the target channel physically includes.

    "whole arm" versus "the tool beyond the sensor only" is the distinction that
    decides whether a model is being asked to explain the robot's dynamics or a
    5 kg handle's, and it is invisible in the column names (R5_03 T-1).
    """
    target = str((manifest.get("signals") or {}).get("target", "link_torque"))
    if target == "ee_wrench_joint":
        sensor = manifest.get("force_torque_sensor") or {}
        measures = ", ".join(sensor.get("measures", [])) or "the bodies past the sensor"
        return (
            f"only the bodies mounted past the {sensor.get('frame', 'sensor')} frame "
            f"({measures}, {sensor.get('mass_kg', float('nan')):.3g} kg), plus any external "
            "contact force; NOT the arm's own dynamics"
        )
    if target == "motor_torque":
        return "the whole arm seen from the motor side, motor inertia and friction included"
    return "the whole arm seen from the link side, transmission excluded"


_METADATA_COLUMN_PREFIXES = (
    "viscous__", "coulomb__", "stiffness__", "damping__", "damping_ratio__", "rotor_inertia__",
    "payload_", "exc_",
    # Round 5, same reasoning: one value per bag repeated on every row.
    "gain_motor__", "gain_link__", "link_viscous__", "link_coulomb__", "ripple_",
    # Round 6: the bag's instrument draws and transmission-error parameters.
    "noise_", "te_",
)


def write_dataset(
    frame: pd.DataFrame, manifest: Mapping[str, Any], output: str | Path,
    comparison: pd.DataFrame | None = None, *, metadata_columns: str = "inline",
) -> tuple[Path, Path, Path | None]:
    """Write the flat dataset, its manifest, a consumer contract sidecar and,
    if any pairs, the backend comparison.

    Dispatches on ``output``'s suffix: ``.csv`` (the default; written with
    ``float_format="%.7g"``, since the consumer casts to float32 on load
    anyway) or ``.parquet`` (typed, compressed, ~10x smaller; written as
    float32 directly).  With ``metadata_columns="sidecar"``, the ~42
    per-row-constant metadata columns (transmission, friction, payload,
    excitation-regime) are looked up by ``bag`` in a sidecar JSON instead of
    repeated on every row -- worth it once the robot count pushes past ~25
    (``R3_06``); ``"inline"`` (the default) keeps the historical, simpler
    single-file behaviour.
    """
    if metadata_columns not in ("inline", "sidecar"):
        raise ValueError("metadata_columns must be 'inline' or 'sidecar'")
    csv_path = Path(output).expanduser().resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if metadata_columns == "sidecar":
        meta_cols = [c for c in frame.columns if c.startswith(_METADATA_COLUMN_PREFIXES)]
        if meta_cols:
            per_bag = frame.groupby("bag", sort=False)[meta_cols].first()
            sidecar = {
                str(bag): {col: (None if pd.isna(value) else value) for col, value in row.items()}
                for bag, row in per_bag.to_dict(orient="index").items()
            }
            sidecar_path = csv_path.with_suffix(".bag_metadata.json")
            sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
            frame = frame.drop(columns=meta_cols)

    suffix = csv_path.suffix.lower()
    if suffix == ".parquet":
        # Keep "t" at float64: it is what the uniform-time-step assertion
        # above and the consumer's Savitzky-Golay filter depend on, and
        # float32's ~1e-6 s spacing at 10 s degrades further on longer
        # trajectories (R3_10 Sec 3.5) -- everything else is a signal or
        # metadata value the consumer casts to float32 on load anyway.
        float_cols = frame.select_dtypes(include="float64").columns.drop("t", errors="ignore")
        frame.astype({col: "float32" for col in float_cols}).to_parquet(csv_path, index=False)
    elif suffix == ".csv":
        frame.to_csv(csv_path, index=False, float_format="%.7g")
    else:
        raise ValueError(f"unsupported dataset output suffix {suffix!r}; use .csv or .parquet")

    manifest_path = csv_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(dict(manifest), indent=2), encoding="utf-8")
    n_dof = int(manifest.get("n_dof", 0))
    contract = {
        "schema": "elastic_sim.identification/2",
        "n_dof": n_dof,
        "input_columns": [f"q0..q{n_dof - 1}", f"dq0..dq{n_dof - 1}", f"tau0..tau{n_dof - 1}"],
        "target_columns": [f"ft0..ft{n_dof - 1}"],
        # Machine-readable target kind, keyed on directly by the consumer at
        # dof == 6 (where "ft0..ft5" is ambiguous between this and a legacy
        # wrench) instead of pattern-matching target_columns' string form
        # (R3_14 Sec 2).
        "target_kind": "per_joint_torque",
        "target_semantics": _TARGET_SEMANTICS[str((manifest.get("signals") or {}).get("target", "link_torque"))],
        # Which instrument the target comes from, and what it physically
        # contains.  The three round-5 platforms have three different answers
        # and a consumer cannot infer it from the column names (R5_03 T-1).
        "target_instrument": str((manifest.get("signals") or {}).get("target_kind", "joint_torque_sensor")),
        "target_contains": _target_contains(manifest),
        "force_torque_sensor": manifest.get("force_torque_sensor"),
        # Diagnostic columns beside the target, when there is a cell (T-15).
        # `sigma_min_j` is sigma_min(J(q)) at the recorded configuration: near
        # a singularity whole wrench directions map to near-zero joint torque,
        # so the target stops carrying part of what the cell measured.  A
        # consumer weighting or filtering samples by posture reads this; it is
        # never a model input.
        "posture_columns": (
            None if not (manifest.get("force_torque_sensor")) else {
                "sigma_min_j": (
                    "smallest singular value of the 6xn frame Jacobian at the recorded "
                    "configuration; information the J^T mapping can still carry, not a "
                    "numerical-failure flag (R5_07 T-15)"
                ),
                "ft_equals": "ft = J(q)^T w on the recorded q* and w_* columns, exactly (R5_06 Sec 3)",
            }
        ),
        # How to differentiate this dataset's velocity (R5_03 T-4).  The
        # consumer reads it instead of hard-coding a window.
        "differentiation": manifest.get("differentiation"),
        # Which side of the spring each column is measured on.  A consumer
        # comparing an elastic model class against a residual one needs this:
        # on a collocated pair the link equation has no elastic term at all,
        # so the comparison measures input availability rather than elasticity
        # (R5_01 Sec 3, R5_00 Amendment 1).
        "signals": manifest.get("signals"),
        "input_semantics": manifest.get("input"),
        "measurement": manifest.get("measurement"),
        "controller": manifest.get("controller"),
        "requires_consumer": "dynamic_model_nn dataset.py with the general ft0..ft{dof-1} branch",
        "split": manifest.get("split"),
    }
    if int(manifest.get("schema_version", 0) or 0) >= 3:
        contract.update(_round6_contract(manifest, n_dof))
    contract_path = csv_path.with_suffix(".contract.json")
    contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    comparison_path = None
    if comparison is not None and not comparison.empty:
        comparison_path = comparison_report_path(csv_path)
        comparison.to_csv(comparison_path, index=False)
    return csv_path, manifest_path, comparison_path


def _round6_contract(manifest: Mapping[str, Any], n_dof: int) -> dict[str, Any]:
    """The contract fields `R6_00` Sec 1 adds; they override the round-5 ones."""
    signals = manifest.get("signals") or {}
    noise = manifest.get("noise") or {}
    measured = signals.get("target_source", "measured") == "measured" and bool(noise.get("enabled", True))
    controller = dict(manifest.get("controller") or {})
    return {
        "schema": "elastic_sim.identification/3",
        "target_kind": "per_joint_torque",
        "target_instrument": "simulated_torque_measurement" if measured else "simulator_ground_truth",
        "target_semantics": "transmission output torque applied to the link [N·m | N]",
        "target_contains": (
            "the torque the transmission applies to the link: the whole robot's rigid dynamics and link "
            "friction seen from the link side, plus elasticity, spring nonlinearity and transmission error "
            "seen from the motor-side inputs"
        ),
        "target_source": "measured" if measured else "clean",
        "target_clean_column": "ft_clean",
        "input_semantics": manifest.get("input"),
        # The machine-readable form the consumer checks (`R6_00` Sec 10):
        # the elastic classes observe tau_cmd and must know it is exact.
        "inputs": {
            "q": {"side": "motor", "measured": True},
            "dq": {"side": "motor", "measured": True,
                   "source": (noise.get("dq") or {}).get("source", "sensor")},
            "tau": {"kind": "commanded_motor_torque", "noise_free": True},
        },
        "controller": {**controller, "model_free": True},
        "noise": noise,
        "motor_friction": manifest.get("motor_friction"),
        "control_timing": manifest.get("control_timing"),
        "posture_columns": (
            {"sigma_min_j": "smallest singular value of the 6xn frame Jacobian at the recorded configuration "
                            "(R5_07 T-15); diagnostic for the wrench columns only"}
            if signals.get("wrench_columns") else None
        ),
        "wrench_columns": (
            {
                "columns": ["w_fx", "w_fy", "w_fz", "w_tx", "w_ty", "w_tz", "sigma_min_j"],
                "acceleration_source": (
                    "Savitzky-Golay derivative (the contract's differentiation window) of the clean link "
                    "velocity on the output grid, not the simulator's point-sampled qacc (R6_00 Sec 5.5)"
                ),
                "instrument": "noise.wrench",
                "use": "post-training chain only; never a model input or target",
            } if signals.get("wrench_columns") else None
        ),
    }


def comparison_report_path(csv_path: str | Path) -> Path:
    return Path(csv_path).with_suffix(".backend_comparison.csv")
