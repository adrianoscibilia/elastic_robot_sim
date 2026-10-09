#!/usr/bin/env python3
"""FR_03 T-6: tables and figures of the round-6 retrain pass.

Reads one ``scripts/round6_retrain.sh`` run folder:

* ``chain-<v>-retrain.yaml``            (the hyperparameters and their Optuna objective);
* ``chain-<v>-retrain.summary.csv``     (every seed's training: selection, best epoch, best val RMSE);
* ``eval-<v>-{test,gainshift}.csv``     (``evaluate_on`` on the original production file and its gain-shift twin);
* ``eval-iiwa_drive-ablation_{off,clean}.csv``.

Writes, under ``--out-dir`` (default ``REFACTOR_SPECS/final_results``):

* ``data/model_comparison_retrain.csv``: one row per (variant, class, seed),
  the columns of ``model_comparison_round6.csv`` plus ``seed``, ``selection``,
  ``best_epoch``, ``best_val_rmse``, ``mean_nrmse``, the ablation RMSEs,
  ``flag`` and ``checkpoint``;
* ``data/model_comparison_retrain_summary.csv``: mean, sd, min, max per
  (variant, class) of the test RMSE and of the mean NRMSE (plus the
  gain-shift ratio), the flagged seeds and the best-val seed's checkpoint;
* ``figures/retrain_fig1_rmse_vs_rigid.png``, ``retrain_fig2_per_joint_nrmse.png``,
  ``retrain_fig3_advantage_vs_content.png``: FR_02's figures, error bars = sd over seeds.

A seed whose test RMSE exceeds 2x the median of the other seeds of its
(variant, class) is flagged, not dropped (FR_02 P-2).

    python scripts/round6_retrain_report.py --run runs/r6-retrain-20261010-0900
    python scripts/round6_retrain_report.py --run runs/<preflight> --out-dir runs/<preflight>/report   # preflight
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

_REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = _REPO / "REFACTOR_SPECS" / "final_results"
VARIANTS = ("iiwa_drive", "iiwa_bus", "ur10", "fmrr")
LABELS = {"iiwa_drive": "iiwa (drive)", "iiwa_bus": "iiwa (bus)", "ur10": "UR10 (drive)", "fmrr": "FMRR (drive)"}
JOINTS = {"iiwa_drive": [f"A{i}" for i in range(1, 8)], "iiwa_bus": [f"A{i}" for i in range(1, 8)],
          "ur10": ["pan", "lift", "elbow", "w1", "w2", "w3"], "fmrr": ["y", "x", "z"]}
#: Elastic content of tau_s (% of variance), FR_01 Sec 6.
ELASTIC_CONTENT = {"iiwa_drive": 6.40, "iiwa_bus": 1.26, "ur10": 1.98, "fmrr": 54.9}
CLASSES = ("lnn_tau_elastic", "kalnn_tau_elastic", "lnn_tau_res", "kalnn_tau_res", "lnn_tau_res_tau", "kalnn_tau_res_tau")
CLASS_LABEL = {"lnn_tau_elastic": "LNN elastic", "kalnn_tau_elastic": "KALNN elastic",
               "lnn_tau_res": "LNN residual", "kalnn_tau_res": "KALNN residual",
               "lnn_tau_res_tau": "LNN residual + τ_cmd", "kalnn_tau_res_tau": "KALNN residual + τ_cmd"}
# FR_02's colours (elastic blues, residual oranges, tau-fit grey) plus a green pair for the
# residual + tau_cmd control, which is also hatched so it never relies on colour alone.
COLOR = {"lnn_tau_elastic": "#1f6f8b", "kalnn_tau_elastic": "#5fb3c9", "lnn_tau_res": "#c0603a",
         "kalnn_tau_res": "#e6a07c", "lnn_tau_res_tau": "#5b7f2a", "kalnn_tau_res_tau": "#a9c97a", "tau_fit": "#999999"}
HATCH = {"lnn_tau_res_tau": "//", "kalnn_tau_res_tau": "//"}
FAMILY = {"elastic": ("lnn_tau_elastic", "kalnn_tau_elastic"), "residual": ("lnn_tau_res", "kalnn_tau_res"),
          "residual_tau": ("lnn_tau_res_tau", "kalnn_tau_res_tau")}
ROUND6_COLUMNS = ["variant", "model", "test_rmse", "gainshift_rmse", "gs_ratio", "zero", "mean", "rigid", "tau_fit",
                  "skill_vs_rigid", "skill_vs_taufit", "skill_vs_zero", "optuna_val_objective", "test_over_val",
                  "n_params", "epochs_run", "batch", "lr", "l1", "loss", "nrmse_joints", "target_rms_joints",
                  "n_test_bags", "dataset_sha256"]


def _read(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.is_file() and path.stat().st_size else None


def _by_checkpoint(frame: pd.DataFrame | None) -> dict[str, pd.Series]:
    if frame is None:
        return {}
    # A re-run stage may have appended twice: the last row of a checkpoint wins.
    return {str(row["checkpoint"]): row for _, row in frame.iterrows()}


def _joint_values(row: pd.Series, prefix: str) -> np.ndarray:
    keys = sorted((k for k in row.index if k.startswith(prefix)), key=lambda k: int(k.rsplit("_", 1)[1]))
    return np.array([float(row[k]) for k in keys])


def variant_rows(run: Path, variant: str) -> list[dict]:
    summary = _read(run / f"chain-{variant}-retrain.summary.csv")
    if summary is None:
        return []
    chain = yaml.safe_load((run / f"chain-{variant}-retrain.yaml").read_text(encoding="utf-8"))
    objective = {r["base_name"]: (r.get("_provenance") or {}).get("objective") for r in chain["runs"]
                 if "parent" not in (r.get("_provenance") or {})}
    evals = {name: _by_checkpoint(_read(run / f"eval-{variant}-{name}.csv"))
             for name in ("test", "gainshift", "ablation_off", "ablation_clean")}
    rows = []
    for _, train in summary.iterrows():
        checkpoint = str(train["checkpoint_path"])
        test = evals["test"].get(checkpoint)
        if test is None:
            continue
        gain = evals["gainshift"].get(checkpoint)
        rmse = float(test["rmse"])
        rmse_j, target_j = _joint_values(test, "rmse_joint_"), _joint_values(test, "target_rms_joint_")
        nrmse = rmse_j / target_j
        zero, mean, rigid, tau = (float(test[k]) if pd.notna(test.get(k)) else math.nan
                                  for k in ("zero_model_rmse", "baseline_rms_mean", "baseline_rms_rigid", "baseline_rms_tau"))
        val_obj = objective.get(train["base_name"])
        gs = float(gain["rmse"]) if gain is not None else math.nan
        row = {
            "variant": variant, "model": train["base_name"], "test_rmse": rmse, "gainshift_rmse": gs,
            "gs_ratio": gs / rmse, "zero": zero, "mean": mean, "rigid": rigid, "tau_fit": tau,
            "skill_vs_rigid": 1 - rmse / rigid, "skill_vs_taufit": 1 - rmse / tau, "skill_vs_zero": 1 - rmse / zero,
            "optuna_val_objective": val_obj if val_obj is not None else math.nan,
            "test_over_val": rmse / val_obj if val_obj else math.nan,
            "n_params": int(train["n_params"]), "epochs_run": int(train["epochs_run"]),
            "batch": int(train["train_batch_size"]), "lr": f"{float(train['train_learning_rate']):.2e}",
            "l1": f"{float(train['train_l1_lambda']):.2e}", "loss": train["train_loss_type"],
            "nrmse_joints": " ".join(f"{v:.2f}" for v in nrmse),
            "target_rms_joints": " ".join(f"{v:.1f}" for v in target_j),
            "n_test_bags": int(test["n_bags"]), "dataset_sha256": str(test["dataset_sha256"])[:12],
            "seed": int(train["seed"]) if pd.notna(train.get("seed")) else None,
            "selection": train.get("selection"),
            "best_epoch": int(train["best_epoch"]) if pd.notna(train.get("best_epoch")) else None,
            "best_val_rmse": float(train["best_val_rmse"]) if pd.notna(train.get("best_val_rmse")) else math.nan,
            "mean_nrmse": float(np.mean(nrmse)),
        }
        for name in ("ablation_off", "ablation_clean"):
            other = evals[name].get(checkpoint)
            row[f"{name}_rmse"] = float(other["rmse"]) if other is not None else math.nan
        row["checkpoint"] = checkpoint
        rows.append(row)
    return rows


def flag_outliers(frame: pd.DataFrame) -> pd.Series:
    """'outlier: ...' for a seed whose test RMSE exceeds 2x the median of the other seeds."""
    flags = pd.Series("", index=frame.index, dtype=object)
    for _, group in frame.groupby(["variant", "model"]):
        if len(group) < 2:
            continue
        for index, value in group["test_rmse"].items():
            others = float(group["test_rmse"].drop(index).median())
            if value > 2 * others:
                flags[index] = f"outlier: test RMSE {value:.3g} > 2 x median of the other seeds ({others:.3g})"
    return flags


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (variant, model), group in frame.groupby(["variant", "model"], sort=False):
        row = {"variant": variant, "model": model, "n_seeds": len(group)}
        for column in ("test_rmse", "mean_nrmse", "gs_ratio"):
            values = group[column].astype(float)
            row.update({f"{column}_mean": values.mean(), f"{column}_sd": values.std(ddof=1) if len(values) > 1 else math.nan,
                        f"{column}_min": values.min(), f"{column}_max": values.max()})
        for column in ("rigid", "tau_fit", "zero"):
            row[column] = float(group[column].iloc[0])
        best = group.loc[group["best_val_rmse"].astype(float).idxmin()] if group["best_val_rmse"].notna().any() else group.iloc[0]
        row.update({"best_val_seed": best["seed"], "best_val_seed_test_rmse": best["test_rmse"],
                    "best_val_seed_checkpoint": best["checkpoint"],
                    "flagged_seeds": " ".join(str(s) for s in group.loc[group["flag"] != "", "seed"])})
        joints = np.array([[float(v) for v in s.split()] for s in group["nrmse_joints"]])
        row["nrmse_joints_mean"] = " ".join(f"{v:.2f}" for v in joints.mean(axis=0))
        row["nrmse_joints_sd"] = " ".join(f"{v:.2f}" for v in (joints.std(axis=0, ddof=1) if len(joints) > 1 else 0 * joints[0]))
        rows.append(row)
    return pd.DataFrame(rows)


def _best(summary: pd.DataFrame, variant: str, family: str) -> pd.Series | None:
    rows = summary[(summary["variant"] == variant) & summary["model"].isin(FAMILY[family])]
    return None if rows.empty else rows.loc[rows["test_rmse_mean"].idxmin()]


def _ratio(a_mean, a_sd, b_mean, b_sd):
    r = a_mean / b_mean
    rel = math.sqrt(((a_sd if np.isfinite(a_sd) else 0) / a_mean) ** 2 + ((b_sd if np.isfinite(b_sd) else 0) / b_mean) ** 2)
    return r, r * rel


def figures(summary: pd.DataFrame, out: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    variants = [v for v in VARIANTS if v in set(summary["variant"])]
    written = []

    # Fig 1: test RMSE / rigid, mean +- sd over seeds, 6 classes + tau-fit.
    fig, ax = plt.subplots(figsize=(11, 4.6))
    width = 0.8 / (len(CLASSES) + 1)
    for i, model in enumerate(list(CLASSES) + ["tau_fit"]):
        xs, ys, es = [], [], []
        for k, variant in enumerate(variants):
            rows = summary[(summary["variant"] == variant)]
            if model == "tau_fit":
                if rows.empty:
                    continue
                xs.append(k); ys.append(rows["tau_fit"].iloc[0] / rows["rigid"].iloc[0]); es.append(0)
                continue
            row = rows[rows["model"] == model]
            if row.empty:
                continue
            row = row.iloc[0]
            xs.append(k); ys.append(row["test_rmse_mean"] / row["rigid"])
            es.append((row["test_rmse_sd"] if np.isfinite(row["test_rmse_sd"]) else 0) / row["rigid"])
        x = np.array(xs) - 0.4 + width * (i + 0.5)
        ax.bar(x, ys, width, yerr=None if model == "tau_fit" or not any(es) else es, capsize=2, color=COLOR[model],
               hatch=HATCH.get(model), edgecolor="white" if model in HATCH else None, linewidth=0,
               label="τ-fit baseline" if model == "tau_fit" else CLASS_LABEL[model], error_kw={"elinewidth": 1})
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(range(len(variants)), [LABELS[v] for v in variants])
    ax.set_ylabel("test RMSE / rigid-model RMSE")
    ax.set_title("Retrain: test RMSE relative to the nominal rigid model (lower is better)\n"
                 "dashed line = nominal rigid URDF model; error bars = sd over seeds", fontsize=11)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4, frameon=False, fontsize=9)
    fig.tight_layout()
    written.append(out / "retrain_fig1_rmse_vs_rigid.png"); fig.savefig(written[-1], dpi=150); plt.close(fig)

    # Fig 2: per-joint NRMSE of the best class of each family, mean +- sd over seeds.
    widths = [len(JOINTS[v]) for v in variants]
    fig, axes = plt.subplots(1, len(variants), figsize=(4.2 * len(variants), 4.6), sharey=True,
                             gridspec_kw={"width_ratios": widths}, squeeze=False)
    for ax, variant in zip(axes[0], variants):
        picks = [(_best(summary, variant, f), f) for f in ("elastic", "residual", "residual_tau")]
        picks = [(row, f) for row, f in picks if row is not None]
        bar = 0.8 / max(len(picks), 1)
        for i, (row, family) in enumerate(picks):
            mean = np.array([float(v) for v in row["nrmse_joints_mean"].split()])
            sd = np.array([float(v) for v in row["nrmse_joints_sd"].split()])
            x = np.arange(len(mean)) - 0.4 + bar * (i + 0.5)
            ax.bar(x, mean, bar, yerr=sd if sd.any() else None, capsize=2, color=COLOR[row["model"]], hatch=HATCH.get(row["model"]),
                   edgecolor="white" if row["model"] in HATCH else None, linewidth=0,
                   label=CLASS_LABEL[row["model"]], error_kw={"elinewidth": 1})
        ax.set_xticks(range(len(JOINTS[variant])), JOINTS[variant])
        ax.set_title(LABELS[variant])
        ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=1)
    axes[0][0].set_ylabel("RMSE / target RMS (per joint)")
    fig.suptitle("Best elastic vs best residual vs best residual + τ_cmd, per joint (test robots, mean ± sd over seeds)")
    fig.tight_layout()
    written.append(out / "retrain_fig2_per_joint_nrmse.png"); fig.savefig(written[-1], dpi=150); plt.close(fig)

    # Fig 3: advantage of the best elastic class vs elastic content.
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    series = (("residual", "o", COLOR["lnn_tau_elastic"], "best residual / best elastic"),
              ("residual_tau", "D", COLOR["lnn_tau_res_tau"], "best residual + τ_cmd / best elastic"),
              ("tau_fit", "s", COLOR["tau_fit"], "τ-fit / best elastic"))
    for family, marker, color, label in series:
        xs, ys, es = [], [], []
        for variant in variants:
            elastic = _best(summary, variant, "elastic")
            if elastic is None:
                continue
            if family == "tau_fit":
                r, e = _ratio(elastic["tau_fit"], 0.0, elastic["test_rmse_mean"], elastic["test_rmse_sd"])
            else:
                other = _best(summary, variant, family)
                if other is None:
                    continue
                r, e = _ratio(other["test_rmse_mean"], other["test_rmse_sd"], elastic["test_rmse_mean"], elastic["test_rmse_sd"])
            xs.append(ELASTIC_CONTENT[variant]); ys.append(r); es.append(e)
        ax.errorbar(xs, ys, yerr=es, fmt=marker, color=color, markersize=7, capsize=3, label=label)
    for variant in variants:
        elastic, residual = _best(summary, variant, "elastic"), _best(summary, variant, "residual")
        if elastic is not None and residual is not None:
            ax.annotate(LABELS[variant], (ELASTIC_CONTENT[variant], residual["test_rmse_mean"] / elastic["test_rmse_mean"]),
                        textcoords="offset points", xytext=(7, 5), fontsize=9)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20, 50], ["1", "2", "5", "10", "20", "50"])
    ax.set_xlabel("elastic content of τ_s (budget, % of variance)")
    ax.set_ylabel("RMSE ratio (>1: elastic class better)")
    ax.set_title("Elastic-class advantage vs elastic content (± sd over seeds)")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    written.append(out / "retrain_fig3_advantage_vs_content.png"); fig.savefig(written[-1], dpi=150); plt.close(fig)
    return written


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="runs/<id> of a round6_retrain.sh run (or preflight)")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT), help=f"default {DEFAULT_OUT}")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--show", action="store_true", help="print the summary table, write nothing")
    args = parser.parse_args(argv)

    run = Path(args.run)
    rows = [row for variant in VARIANTS for row in variant_rows(run, variant)]
    if not rows:
        raise SystemExit(f"{run}: no evaluated chain summaries (chain-<v>-retrain.summary.csv + eval-<v>-test.csv)")
    frame = pd.DataFrame(rows)
    frame["flag"] = flag_outliers(frame)
    extra = ["seed", "selection", "best_epoch", "best_val_rmse", "mean_nrmse", "ablation_off_rmse",
             "ablation_clean_rmse", "flag", "checkpoint"]
    frame = frame[ROUND6_COLUMNS + extra]
    summary = summarize(frame)
    if args.show:
        with pd.option_context("display.width", 200, "display.max_columns", 12):
            print(summary[["variant", "model", "n_seeds", "test_rmse_mean", "test_rmse_sd", "mean_nrmse_mean",
                           "gs_ratio_mean", "flagged_seeds"]])
        return 0
    out = Path(args.out_dir)
    (out / "data").mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "data" / "model_comparison_retrain.csv", index=False, float_format="%.6g")
    summary.to_csv(out / "data" / "model_comparison_retrain_summary.csv", index=False, float_format="%.6g")
    print(f"wrote {out / 'data' / 'model_comparison_retrain.csv'} ({len(frame)} rows) and the summary "
          f"({len(summary)} rows); flagged seeds: {int((frame['flag'] != '').sum())}")
    if not args.no_figures:
        for path in figures(summary, out / "figures"):
            print(f"wrote {path}")
    (out / "data" / "model_comparison_retrain.source.json").write_text(
        json.dumps({"run": str(run.resolve()), "rows": len(frame)}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
