#!/usr/bin/env python3
"""FR_04 Sec 3.7: tables and figures of the torque-free comparison.

Reads one ``scripts/round7_taufree.sh`` run folder:

* ``chain-<v>-taufree.yaml`` and ``chain-<v>-taufree.summary.csv`` (every seed's training);
* ``eval-<v>-{test,gainshift}.csv`` (``evaluate_on --exclude-head-s <burn-in>`` on the r7 production
  file and its gain-shift twin: masked RMSE, unmasked RMSE, the two-mass baselines, Q-E columns);
* ``s1-<v>*.json`` (the S-1 identity checks) and, under ``--data-root``, the r7 contracts;
* ``reports/round7/budget_<stem>.json`` (the r7 budget, for the elastic content).

Writes, under ``--out-dir`` (default ``REFACTOR_SPECS/final_results``):

* ``data/model_comparison_taufree.csv``: one row per (variant, class, seed), the retrain report's columns
  (``rigid`` here is the masked rigid baseline) plus ``tau_input`` (yes/no), ``history``
  (none/filter/tcn), ``rigid_elastic_nominal``, ``rigid_elastic_oracle`` (the contract's baseline with the
  run's own coupling and integrator, named in ``baseline_filter``), ``test_rmse_unmasked``;
* ``data/model_comparison_taufree_summary.csv``: mean, sd, min, max per (variant, class);
* ``data/taufree_qe.csv``: Q-E per (variant, two-mass class, seed, joint): corr and RMSE of the
  latent deflection against the simulator's, learned K and zeta, the test robots' true K, the prior;
* ``data/taufree_s1_budget.csv``: the S-1 checks and the r7 budget per variant;
* ``figures/taufree_fig1_rmse_vs_rigid.png``, ``taufree_fig2_per_joint_nrmse.png``,
  ``taufree_fig3_gain_vs_elastic_content.png``.

    python scripts/round7_taufree_report.py --run runs/r7-taufree-20261013-0900
    python scripts/round7_taufree_report.py --run runs/<preflight> --data-root data/identification/<preflight> \\
        --out-dir runs/<preflight>/report
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))
from round6_retrain_report import JOINTS, LABELS, _by_checkpoint, _joint_values, _ratio, _read, flag_outliers  # noqa: E402

DEFAULT_OUT = _REPO / "REFACTOR_SPECS" / "final_results"
DEFAULT_DATA_ROOT = "data/identification/r7-nopayload"
VARIANTS = ("iiwa_drive", "iiwa_bus", "ur10", "fmrr")
STEMS = {"iiwa_drive": "kuka_lbr_iiwa_14_r820_table_drive_round7_nopayload",
         "iiwa_bus": "kuka_lbr_iiwa_14_r820_table_bus_round7_nopayload",
         "ur10": "ur10_table_round7_nopayload", "fmrr": "fmrr_tecnobody_round7_nopayload"}
#: Round-6 elastic content (FR_01 Sec 6), the fallback when no r7 budget was written.
ELASTIC_CONTENT_R6 = {"iiwa_drive": 6.40, "iiwa_bus": 1.26, "ur10": 1.98, "fmrr": 54.9}
CLASSES = ("lnn_tau_res", "kalnn_tau_res", "lnn_twomass", "kalnn_twomass", "lnn_res_hist", "kalnn_res_hist",
           "lnn_elastic_qdq", "kalnn_elastic_qdq", "lnn_tau_elastic", "kalnn_tau_elastic")
CLASS_LABEL = {"lnn_tau_res": "LNN residual", "kalnn_tau_res": "KALNN residual",
               "lnn_twomass": "LNN two-mass", "kalnn_twomass": "KALNN two-mass",
               "lnn_res_hist": "LNN residual + TCN", "kalnn_res_hist": "KALNN residual + TCN",
               "lnn_elastic_qdq": "LNN elastic z(q, q̇)", "kalnn_elastic_qdq": "KALNN elastic z(q, q̇)",
               "lnn_tau_elastic": "LNN elastic + τ_cmd", "kalnn_tau_elastic": "KALNN elastic + τ_cmd"}
TAU_INPUT = {name: "yes" if name.endswith("tau_elastic") else "no" for name in CLASSES}
HISTORY = {name: "filter" if "twomass" in name else ("tcn" if "res_hist" in name else "none") for name in CLASSES}
FAMILY = {"residual": ("lnn_tau_res", "kalnn_tau_res"), "twomass": ("lnn_twomass", "kalnn_twomass"),
          "tcn": ("lnn_res_hist", "kalnn_res_hist"), "elastic_qdq": ("lnn_elastic_qdq", "kalnn_elastic_qdq"),
          "tau_elastic": ("lnn_tau_elastic", "kalnn_tau_elastic")}
# Reference-palette slots (dataviz skill): one hue per family, KALNN a lighter step of its LNN hue;
# the tau-informed reference is hatched, so tau input never rests on colour alone.
COLOR = {"lnn_tau_res": "#eb6834", "kalnn_tau_res": "#f4a483", "lnn_twomass": "#2a78d6", "kalnn_twomass": "#8db8ec",
         "lnn_res_hist": "#4a3aa7", "kalnn_res_hist": "#9d93d6", "lnn_elastic_qdq": "#eda100",
         "kalnn_elastic_qdq": "#f5cb6b", "lnn_tau_elastic": "#1baf7a", "kalnn_tau_elastic": "#8bd8bb"}
HATCH = {"lnn_tau_elastic": "//", "kalnn_tau_elastic": "//"}
BASELINE_STYLE = {"rigid_elastic_nominal": (":", "#52514e", "two-mass filter, nominal K (baseline)"),
                  "rigid_elastic_oracle": ("-", "#0b0b0b", "two-mass filter, true K and D (oracle baseline)")}
COLUMNS = ["variant", "model", "tau_input", "history", "test_rmse", "test_rmse_unmasked", "gainshift_rmse",
           "gs_ratio", "zero", "rigid", "rigid_elastic_nominal", "rigid_elastic_oracle", "tau_fit",
           "skill_vs_rigid", "skill_vs_oracle", "n_params", "epochs_run", "batch", "lr", "l1", "loss",
           "nrmse_joints", "target_rms_joints", "n_test_bags", "n_test_rows", "exclude_head_s", "baselines_rows_match", "baseline_filter",
           "dataset_sha256", "seed", "selection", "best_epoch", "best_val_rmse", "mean_nrmse", "flag", "checkpoint"]


def _float(row, key):
    value = row.get(key) if row is not None else None
    return float(value) if value is not None and pd.notna(value) else math.nan


def variant_rows(run: Path, variant: str) -> tuple[list[dict], list[dict]]:
    summary = _read(run / f"chain-{variant}-taufree.summary.csv")
    if summary is None:
        return [], []
    evals = {name: _by_checkpoint(_read(run / f"eval-{variant}-{name}.csv")) for name in ("test", "gainshift")}
    suffix = filter_suffix(run, variant)
    rows, qe = [], []
    for _, train in summary.iterrows():
        checkpoint = str(train["checkpoint_path"])
        test = evals["test"].get(checkpoint)
        if test is None:
            continue
        gain = evals["gainshift"].get(checkpoint)
        rmse = float(test["rmse"])
        rmse_j, target_j = _joint_values(test, "rmse_joint_"), _joint_values(test, "target_rms_joint_")
        nrmse = rmse_j / target_j
        rigid, oracle = _float(test, "baseline_rigid_masked"), _float(test, f"baseline_rigid_elastic_oracle{suffix}")
        gs = _float(gain, "rmse")
        name = str(train["base_name"])
        row = {
            "variant": variant, "model": name, "tau_input": TAU_INPUT.get(name, "?"), "history": HISTORY.get(name, "?"),
            "test_rmse": rmse, "test_rmse_unmasked": _float(test, "rmse_unmasked"), "gainshift_rmse": gs,
            "gs_ratio": gs / rmse, "zero": _float(test, "zero_model_rmse"), "rigid": rigid,
            "rigid_elastic_nominal": _float(test, f"baseline_rigid_elastic_nominal{suffix}"), "rigid_elastic_oracle": oracle,
            "baseline_filter": suffix or "diagonal/trapezoidal",
            "tau_fit": _float(test, "baseline_rms_tau"),
            "skill_vs_rigid": 1 - rmse / rigid, "skill_vs_oracle": 1 - rmse / oracle,
            "n_params": int(train["n_params"]), "epochs_run": int(train["epochs_run"]),
            "batch": int(train["train_batch_size"]), "lr": f"{float(train['train_learning_rate']):.2e}",
            "l1": f"{float(train['train_l1_lambda']):.2e}", "loss": train["train_loss_type"],
            "nrmse_joints": " ".join(f"{v:.2f}" for v in nrmse),
            "target_rms_joints": " ".join(f"{v:.1f}" for v in target_j),
            "n_test_bags": int(test["n_bags"]), "n_test_rows": int(test["n_rows"]),
            "exclude_head_s": _float(test, "exclude_head_s"),
            "baselines_rows_match": bool(test.get("baselines_two_mass_rows_match")) if pd.notna(
                test.get("baselines_two_mass_rows_match")) else False,
            "dataset_sha256": str(test["dataset_sha256"])[:12],
            "seed": int(train["seed"]) if pd.notna(train.get("seed")) else None, "selection": train.get("selection"),
            "best_epoch": int(train["best_epoch"]) if pd.notna(train.get("best_epoch")) else None,
            "best_val_rmse": _float(train, "best_val_rmse"), "mean_nrmse": float(np.mean(nrmse)),
            "checkpoint": checkpoint,
        }
        rows.append(row)
        for j in range(len(rmse_j)):
            if f"qe_corr_joint_{j}" not in test.index or pd.isna(test.get(f"qe_corr_joint_{j}")):
                continue
            qe.append({"variant": variant, "model": name, "seed": row["seed"],
                       "joint": JOINTS.get(variant, [str(i) for i in range(len(rmse_j))])[j],
                       "corr_delta": _float(test, f"qe_corr_joint_{j}"), "rmse_delta": _float(test, f"qe_rmse_joint_{j}"),
                       "rms_true_delta": _float(test, f"qe_defl_rms_joint_{j}"),
                       "learned_K": _float(test, f"learned_K_joint_{j}"),
                       "true_K_mean": _float(test, f"true_K_mean_joint_{j}"),
                       "true_K_min": _float(test, f"true_K_min_joint_{j}"),
                       "true_K_max": _float(test, f"true_K_max_joint_{j}"),
                       "prior_K_low": _float(test, f"prior_K_low_joint_{j}"),
                       "prior_K_high": _float(test, f"prior_K_high_joint_{j}"),
                       "learned_zeta": _float(test, f"learned_zeta_joint_{j}")})
    return rows, qe


def filter_suffix(run: Path, variant: str) -> str:
    """The contract baseline suffix matching the run's two-mass filter (coupling, integrator)."""
    import yaml

    path = run / f"chain-{variant}-taufree.yaml"
    kwargs = {}
    if path.is_file():
        for entry in yaml.safe_load(path.read_text(encoding="utf-8"))["runs"]:
            if "twomass" in entry["base_name"]:
                kwargs = entry.get("model_kwargs") or {}
                break
    coupling, integrator = kwargs.get("coupling", "diagonal"), kwargs.get("integrator", "trapezoidal")
    return {("diagonal", "trapezoidal"): "", ("full", "trapezoidal"): "_full",
            ("diagonal", "backward_euler"): "_be", ("full", "backward_euler"): "_full_be"}[(coupling, integrator)]


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (variant, model), group in frame.groupby(["variant", "model"], sort=False):
        row = {"variant": variant, "model": model, "tau_input": group["tau_input"].iloc[0],
               "history": group["history"].iloc[0], "n_seeds": len(group), "n_params": int(group["n_params"].iloc[0])}
        for column in ("test_rmse", "test_rmse_unmasked", "mean_nrmse", "gs_ratio"):
            values = group[column].astype(float)
            row.update({f"{column}_mean": values.mean(), f"{column}_sd": values.std(ddof=1) if len(values) > 1 else math.nan,
                        f"{column}_min": values.min(), f"{column}_max": values.max()})
        for column in ("rigid", "rigid_elastic_nominal", "rigid_elastic_oracle", "tau_fit", "zero", "exclude_head_s"):
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


def s1_budget_table(run: Path) -> pd.DataFrame:
    """Per variant: the S-1 identity verdicts and the r7 budget's pooled shares (documentation)."""
    rows = []
    for variant in VARIANTS:
        row = {"variant": variant}
        for suffix in ("", "_gainshift"):
            path = run / f"s1-{variant}{suffix}.json"
            if path.is_file():
                report = json.loads(path.read_text(encoding="utf-8"))
                for check in report["checks"]:
                    key = check["check"].split(" (")[0].split(" equal")[0].replace(" ", "_")
                    row[f"s1{suffix}_{key}"] = bool(check["ok"])
                row[f"s1{suffix}_ok"] = bool(report["ok"])
        budget = _REPO / "reports" / "round7" / f"budget_{STEMS[variant]}.json"
        if budget.is_file():
            data = json.loads(budget.read_text(encoding="utf-8"))
            shares = data.get("pooled_shares_mostly_observable") or data.get("pooled_shares_observable") or {}
            for key in ("elastic content", "non-rigid part", "rigid dynamics", "noise", "loop-noise content",
                        "motor friction", "link friction", "all extras together"):
                if key in shares:
                    row[f"budget_{key.replace(' ', '_').replace('-', '_')}_pct"] = 100.0 * float(shares[key])
            row["budget_gates_ok"] = bool(data.get("ok_mostly_observable"))
        if len(row) > 1:
            rows.append(row)
    return pd.DataFrame(rows)


def elastic_content(table: pd.DataFrame) -> dict[str, tuple[float, str]]:
    out = {}
    for variant in VARIANTS:
        rows = table[table["variant"] == variant] if not table.empty else table
        column = "budget_elastic_content_pct"
        if not rows.empty and column in rows and pd.notna(rows[column].iloc[0]):
            out[variant] = (float(rows[column].iloc[0]), "r7 budget")
        else:
            out[variant] = (ELASTIC_CONTENT_R6[variant], "round-6 budget (no r7 budget found)")
    return out


def _best(summary: pd.DataFrame, variant: str, family: str) -> pd.Series | None:
    rows = summary[(summary["variant"] == variant) & summary["model"].isin(FAMILY[family])]
    return None if rows.empty else rows.loc[rows["test_rmse_mean"].idxmin()]


def figures(summary: pd.DataFrame, content: dict, out: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    variants = [v for v in VARIANTS if v in set(summary["variant"])]
    present = [c for c in CLASSES if c in set(summary["model"])]
    written = []
    grid = {"color": "#e4e3df", "linewidth": 0.6}

    # Fig 1: test RMSE / masked rigid, mean +- sd over seeds; the two-mass baselines as markers.
    fig, ax = plt.subplots(figsize=(12, 4.8))
    width = 0.8 / max(len(present), 1)
    for i, model in enumerate(present):
        xs, ys, es = [], [], []
        for k, variant in enumerate(variants):
            row = summary[(summary["variant"] == variant) & (summary["model"] == model)]
            if row.empty:
                continue
            row = row.iloc[0]
            xs.append(k); ys.append(row["test_rmse_mean"] / row["rigid"])
            es.append((row["test_rmse_sd"] if np.isfinite(row["test_rmse_sd"]) else 0) / row["rigid"])
        ax.bar(np.array(xs) - 0.4 + width * (i + 0.5), ys, width * 0.9, yerr=es if any(es) else None, capsize=2,
               color=COLOR[model], hatch=HATCH.get(model), edgecolor="white" if model in HATCH else None, linewidth=0,
               label=CLASS_LABEL[model], error_kw={"elinewidth": 1}, zorder=2)
    for name, (style, color, label) in BASELINE_STYLE.items():
        for k, variant in enumerate(variants):
            rows = summary[summary["variant"] == variant]
            if not rows.empty and np.isfinite(rows[name].iloc[0]):
                y = rows[name].iloc[0] / rows["rigid"].iloc[0]
                ax.hlines(y, k - 0.42, k + 0.42, colors=color, linestyles=style, linewidth=2, zorder=3,
                          label=label if k == 0 else None)
    ax.axhline(1.0, color="#0b0b0b", linestyle="--", linewidth=1)
    ax.yaxis.grid(True, **grid); ax.set_axisbelow(True)
    ax.set_xticks(range(len(variants)), [LABELS[v] for v in variants])
    ax.set_ylabel("test RMSE / rigid-model RMSE")
    ax.set_title("Torque-free comparison: test RMSE relative to the nominal rigid model (lower is better)\n"
                 "same rows for every model and baseline (each bag's burn-in excluded); error bars = sd over seeds",
                 fontsize=11)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4, frameon=False, fontsize=9)
    fig.tight_layout()
    written.append(out / "taufree_fig1_rmse_vs_rigid.png"); fig.savefig(written[-1], dpi=150); plt.close(fig)

    # Fig 2: per-joint NRMSE, best of each family.
    widths = [len(JOINTS[v]) for v in variants]
    fig, axes = plt.subplots(1, len(variants), figsize=(4.4 * len(variants), 4.8), sharey=True,
                             gridspec_kw={"width_ratios": widths}, squeeze=False)
    for ax, variant in zip(axes[0], variants):
        picks = [p for p in (_best(summary, variant, f) for f in FAMILY) if p is not None]
        bar = 0.8 / max(len(picks), 1)
        for i, row in enumerate(picks):
            mean = np.array([float(v) for v in row["nrmse_joints_mean"].split()])
            sd = np.array([float(v) for v in row["nrmse_joints_sd"].split()])
            ax.bar(np.arange(len(mean)) - 0.4 + bar * (i + 0.5), mean, bar * 0.9, yerr=sd if sd.any() else None,
                   capsize=2, color=COLOR[row["model"]], hatch=HATCH.get(row["model"]),
                   edgecolor="white" if row["model"] in HATCH else None, linewidth=0, label=CLASS_LABEL[row["model"]],
                   error_kw={"elinewidth": 1}, zorder=2)
        ax.yaxis.grid(True, **grid); ax.set_axisbelow(True)
        ax.set_xticks(range(len(JOINTS[variant])), JOINTS[variant])
        ax.set_title(LABELS[variant])
        ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=1)
    axes[0][0].set_ylabel("RMSE / target RMS (per joint)")
    fig.suptitle("Best class per family, per joint\n(test robots, mean ± sd over seeds)", fontsize=11)
    fig.tight_layout()
    written.append(out / "taufree_fig2_per_joint_nrmse.png"); fig.savefig(written[-1], dpi=150); plt.close(fig)

    # Fig 3: two-mass / residual and two-mass / TCN control vs the r7 elastic content.
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    series = (("residual", "o", COLOR["lnn_tau_res"], "best two-mass / best residual (Q-B)"),
              ("tcn", "D", COLOR["lnn_res_hist"], "best two-mass / best TCN control (Q-C)"))
    for family, marker, color, label in series:
        xs, ys, es = [], [], []
        for variant in variants:
            two_mass, other = _best(summary, variant, "twomass"), _best(summary, variant, family)
            if two_mass is None or other is None:
                continue
            r, e = _ratio(two_mass["test_rmse_mean"], two_mass["test_rmse_sd"], other["test_rmse_mean"], other["test_rmse_sd"])
            xs.append(content[variant][0]); ys.append(r); es.append(e)
        ax.errorbar(xs, ys, yerr=es, fmt=marker, color=color, markersize=8, capsize=3, label=label)
    for variant in variants:
        two_mass, residual = _best(summary, variant, "twomass"), _best(summary, variant, "residual")
        if two_mass is not None and residual is not None:
            ax.annotate(LABELS[variant], (content[variant][0], two_mass["test_rmse_mean"] / residual["test_rmse_mean"]),
                        textcoords="offset points", xytext=(7, 5), fontsize=9)
    ax.axhline(1.0, color="#0b0b0b", linestyle="--", linewidth=0.8)
    ax.yaxis.grid(True, **grid); ax.set_axisbelow(True)
    ax.set_xscale("log")
    ax.set_xlim(0.7, 150)
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100], ["1", "2", "5", "10", "20", "50", "100"])
    ax.set_xlabel("elastic content of τ_s, r7 budget (% of variance)")
    ax.set_ylabel("RMSE ratio (<1: two-mass better)")
    ax.set_title("Two-mass gain vs elastic content (± sd over seeds)")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    written.append(out / "taufree_fig3_gain_vs_elastic_content.png"); fig.savefig(written[-1], dpi=150); plt.close(fig)
    return written


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="runs/<id> of a round7_taufree.sh run (or preflight)")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT, help=f"where the r7 files are (default {DEFAULT_DATA_ROOT})")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT), help=f"default {DEFAULT_OUT}")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--show", action="store_true", help="print the summary table, write nothing")
    args = parser.parse_args(argv)

    run = Path(args.run)
    rows, qe = [], []
    for variant in VARIANTS:
        r, q = variant_rows(run, variant)
        rows += r; qe += q
    if not rows:
        raise SystemExit(f"{run}: no evaluated chain summaries (chain-<v>-taufree.summary.csv + eval-<v>-test.csv)")
    frame = pd.DataFrame(rows)
    frame["flag"] = flag_outliers(frame)
    frame = frame[COLUMNS]
    summary = summarize(frame)
    table = s1_budget_table(run)
    content = elastic_content(table)
    if args.show:
        with pd.option_context("display.width", 220, "display.max_columns", 14):
            print(summary[["variant", "model", "n_seeds", "test_rmse_mean", "test_rmse_sd", "rigid",
                           "rigid_elastic_nominal", "rigid_elastic_oracle", "gs_ratio_mean", "flagged_seeds"]])
        return 0
    out = Path(args.out_dir)
    (out / "data").mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "data" / "model_comparison_taufree.csv", index=False, float_format="%.6g")
    summary.to_csv(out / "data" / "model_comparison_taufree_summary.csv", index=False, float_format="%.6g")
    pd.DataFrame(qe).to_csv(out / "data" / "taufree_qe.csv", index=False, float_format="%.6g")
    table.to_csv(out / "data" / "taufree_s1_budget.csv", index=False, float_format="%.6g")
    print(f"wrote {out / 'data' / 'model_comparison_taufree.csv'} ({len(frame)} rows), the summary ({len(summary)}), "
          f"Q-E ({len(qe)} rows), S-1/budget ({len(table)} variants); flagged seeds: {int((frame['flag'] != '').sum())}")
    if not args.no_figures:
        for path in figures(summary, content, out / "figures"):
            print(f"wrote {path}")
    (out / "data" / "model_comparison_taufree.source.json").write_text(json.dumps(
        {"run": str(run.resolve()), "rows": len(frame), "elastic_content": content}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
