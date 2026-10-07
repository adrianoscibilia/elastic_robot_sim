"""``evaluate`` stage (RR_12 C-4): every checkpoint of a chain summary on a
converted real dataset, in the consumer's own environment. Offline, never part
of ``all``; runs where the checkpoints are (peepo).

``evaluate_on.py --chain-summary <summary> --dataset <real> --split test``
does the work. Two things are done around it:

* the chain summaries name checkpoints by their training-time path
  (``models/<name>_checkpoint.pt``); ``dynamic_model_nn`` has since moved them
  into month folders and logged every move in ``models/RELOCATIONS.tsv``. Each
  row is resolved through that table, its sha256 checked, and the resolved
  summary is what ``evaluate_on.py`` reads;
* each result row gets the real baselines (from the dataset contract), the
  dataset's validation status and, when the real differentiation cutoff
  doesn't match the training one (UR10, RR_02 CR-2), ``rate_mismatch: true``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

RELOCATIONS = Path("models") / "RELOCATIONS.tsv"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_checkpoint(path: str, consumer_repo: Path, relocations: pd.DataFrame | None) -> dict[str, Any]:
    """A chain-summary checkpoint path, resolved inside ``consumer_repo``."""
    direct = consumer_repo / path
    if direct.is_file():
        return {"checkpoint": str(direct), "relocated_from": None, "sha256": _sha256(direct)}
    if relocations is not None:
        rows = relocations[relocations["old_path"] == path]
        if len(rows):
            row = rows.iloc[-1]
            target = consumer_repo / row["new_path"]
            if not target.is_file():
                raise FileNotFoundError(f"{path}: relocated to {row['new_path']}, which is missing")
            digest = _sha256(target)
            if digest != row["sha256"]:
                raise ValueError(f"{target}: sha256 {digest} != RELOCATIONS.tsv {row['sha256']}")
            return {"checkpoint": str(target), "relocated_from": path, "sha256": digest}
    raise FileNotFoundError(f"checkpoint {path} not found in {consumer_repo} (nor in {RELOCATIONS})")


def dataset_label(contract: dict[str, Any], validation: dict[str, Any]) -> dict[str, Any]:
    """RR_14 P-1: what a result row may call the dataset. Only a contract
    with ``source: real`` written from ``hardware: real`` can carry a real
    validation status; anything else is refused, never relabelled."""
    source = contract.get("source")
    hardware = (contract.get("real") or {}).get("hardware")
    validated_on = validation.get("hardware")
    status = validation.get("status")
    if source == "real" and (hardware != "real" or (validated_on is not None and validated_on != "real")):
        raise ValueError(f"contract says source: real but the data comes from hardware: {hardware or validated_on} "
                         "-- re-convert with the current code (RR_14 P-1)")
    if source != "real" and (status == "valid" or validated_on == "real"):
        raise ValueError(f"contract source {source!r} cannot carry a real validation (status {status!r}, "
                         f"hardware {validated_on!r})")
    return {"dataset_source": source, "dataset_hardware": hardware, "dataset_status": status}


def evaluate(run_dir: Path, *, robot: str, checkpoints_csv: Path, consumer_repo: Path, consumer_python: Path,
             split: str = "test") -> dict[str, Any]:
    dataset = run_dir / "dataset" / f"{robot}.parquet"
    if not dataset.is_file():
        raise FileNotFoundError(f"{dataset}: run `convert` first")
    contract = json.loads(dataset.with_suffix(".contract.json").read_text())
    validation_path = run_dir / "validation.json"
    validation = json.loads(validation_path.read_text()) if validation_path.is_file() else {}
    label = dataset_label(contract, validation)
    out_dir = run_dir / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(checkpoints_csv)
    relocations_path = consumer_repo / RELOCATIONS
    relocations = pd.read_csv(relocations_path, sep="\t") if relocations_path.is_file() else None
    column = "checkpoint_path" if "checkpoint_path" in summary.columns else "model_path"
    resolved = [resolve_checkpoint(p, consumer_repo, relocations) for p in summary[column]]
    resolved_summary = summary.copy()
    resolved_summary["checkpoint_path"] = [r["checkpoint"] for r in resolved]
    resolved_csv = out_dir / "checkpoints.resolved.csv"
    resolved_summary.to_csv(resolved_csv, index=False)

    raw_csv = out_dir / "evaluate_on.csv"
    raw_csv.unlink(missing_ok=True)  # evaluate_on appends
    command = [str(consumer_python), "evaluate_on.py", "--chain-summary", str(resolved_csv),
               "--dataset", str(dataset.resolve()), "--split", split, "--out", str(raw_csv)]
    result = subprocess.run(command, cwd=consumer_repo, capture_output=True, text=True)
    (out_dir / "evaluate_on.log").write_text(f"$ {' '.join(command)}\n\n{result.stdout}\n{result.stderr}")
    if result.returncode != 0 and not raw_csv.is_file():
        raise RuntimeError(f"evaluate_on.py failed ({result.returncode}):\n{result.stderr[-3000:]}")

    rows = pd.read_csv(raw_csv)
    differentiation = contract.get("differentiation") or {}
    rate_mismatch = not differentiation.get("cutoff_matched", True)
    splits = ((contract.get("baselines") or {}).get("splits") or {}).get(split) or {}
    rows.insert(0, "base_name", list(summary["base_name"]) if "base_name" in summary else None)
    rows["checkpoint_sha256"] = [r["sha256"] for r in resolved]
    rows["relocated_from"] = [r["relocated_from"] for r in resolved]
    for key, value in label.items():
        rows[key] = value
    rows["rate_mismatch"] = rate_mismatch
    rows["sg_cutoff_hz"] = differentiation.get("sg_cutoff_hz")
    rows["reference_cutoff_hz"] = differentiation.get("reference_cutoff_hz")
    rows["baseline_rms_state"] = splits.get("baseline_rms_state")
    rows["baseline_rms_state_tau"] = splits.get("baseline_rms_state_tau")
    rows.to_csv(out_dir / "evaluation.csv", index=False)

    lines = [f"# Evaluation on `{dataset.name}` ({split} split)", "",
             f"Dataset source: **{label['dataset_source']}** ({label['dataset_hardware']}), "
             f"status: **{validation.get('status')}**. Rate mismatch: **{rate_mismatch}**"
             + (f" (SG cutoff {differentiation.get('sg_cutoff_hz'):.4g} Hz vs training "
                f"{differentiation.get('reference_cutoff_hz'):.4g} Hz, RR_02 CR-2)" if rate_mismatch else ""), "",
             "| model | RMSE | zero model | baseline_rms_tau | baseline_rms_rigid | baseline_rms_state | rows |",
             "|---|---|---|---|---|---|---|"]
    for _, row in rows.iterrows():
        lines.append(f"| {row['base_name']} | {row['rmse']:.4g} | {row['zero_model_rmse']:.4g} | "
                     f"{_fmt(row.get('baseline_rms_tau'))} | {_fmt(row.get('baseline_rms_rigid'))} | "
                     f"{_fmt(row.get('baseline_rms_state'))} | {row['n_rows']} |")
    if validation.get("status") != "valid":
        lines += ["", f"The dataset is `{validation.get('status')}`: these numbers carry no physical meaning."]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")
    return {"ok": True, "rows": int(len(rows)), "evaluation": str(out_dir / "evaluation.csv"),
            "rate_mismatch": rate_mismatch, "finite": bool(pd.to_numeric(rows["rmse"]).notna().all())}


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.4g}"
    except (TypeError, ValueError):
        return "n/a"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="erd_recording.evaluate_cli")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--robot", required=True)
    parser.add_argument("--checkpoints", required=True, help="train_chain_from_config summary CSV")
    parser.add_argument("--consumer-repo", required=True)
    parser.add_argument("--consumer-python", required=True)
    parser.add_argument("--split", default="test")
    args = parser.parse_args(argv or sys.argv[1:])
    result = evaluate(Path(args.run_dir), robot=args.robot, checkpoints_csv=Path(args.checkpoints),
                      consumer_repo=Path(args.consumer_repo), consumer_python=Path(args.consumer_python),
                      split=args.split)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
