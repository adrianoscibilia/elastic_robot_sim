#!/usr/bin/env python3
"""identified_filter/01 Sec 7: assemble ``results/SUMMARY.md`` from what the campaign wrote (descriptive only).

Reads ``runs/<run_id>/`` (campaign.log, jobs, eval), ``results/{datasets.md, campaign_plan.json, progress.md,
phase0/, real/}`` and writes ``results/SUMMARY.md`` plus figures under ``results/figures/``:

1. what ran, what was dropped, wall times, failures;
2. simulated table: J1-J11 against rigid / nominal / oracle, per platform and file;
3. identification table: K_hat, D_hat, M_phi vs truth on held-out and out-of-prior robots, by regime and
   observability (+ the K_hat vs K scatter);
4. Phase 0 (0a-0d) and the r8 0c rerun; 5-6. old and new real files; 7. figures; 8. a claims table
   with its evidence column left for the reader (supported / not supported is the owner's call).

    python scripts/r8_summary.py [--run-id r8-campaign]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SIM = Path(__file__).resolve().parents[1]
RESULTS = SIM / "REFACTOR_SPECS" / "identified_filter" / "results"
JOB_NAMES = {"J1": "2M-KALNN-T (teacher, privileged)", "J2": "2M-KALNN-PHYS", "J3": "2M-KALNN-ID",
             "J4": "KALNN residual (r7)", "J5": "2M-KALNN (r7 config)", "J6": "J3, lambda_p = 0",
             "J7": "J3, lambda_delta = 0", "J8": "J3, causal encoder", "J9": "2M-LNN-ID", "J10": "J3, q_dq residual",
             "J11": "KALNN residual + 2.1 + 2.2"}


def _read(path):
    return pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()


def simulated_table(run: Path) -> pd.DataFrame:
    rows = []
    for csv in sorted((run / "eval").glob("*/*.csv")):
        if csv.name.endswith(".per_bag.csv"):
            continue
        frame = _read(csv)
        if not len(frame):
            continue
        job = csv.parent.name
        tag = csv.stem
        for _, r in frame.iterrows():
            rows.append({"job": job, "model": JOB_NAMES.get(job, job), "file": tag, "rmse": r["rmse"],
                         "rigid": r.get("baseline_rigid_masked"), "nominal": r.get("baseline_rigid_elastic_nominal"),
                         "oracle": r.get("baseline_rigid_elastic_oracle"), "privileged_eval": r.get("privileged_eval"),
                         "checkpoint": r["checkpoint"]})
    table = pd.DataFrame(rows)
    if not len(table):
        return table
    return table.groupby(["file", "job", "model", "privileged_eval"], dropna=False).agg(
        rmse_mean=("rmse", "mean"), rmse_min=("rmse", "min"), seeds=("rmse", "size"), rigid=("rigid", "first"),
        nominal=("nominal", "first"), oracle=("oracle", "first")).reset_index()


def _log_r2(est, truth):
    ok = np.isfinite(est) & np.isfinite(truth) & (est > 0) & (truth > 0)
    if ok.sum() < 3:
        return np.nan
    x, y = np.log(truth[ok]), np.log(est[ok])
    return 1 - np.sum((y - x) ** 2) / max(np.sum((x - x.mean()) ** 2), 1e-30)


def identification_table(run: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts = []
    for csv in sorted((run / "eval").glob("*/*.per_bag.csv")):
        frame = _read(csv)
        if len(frame) and "K_hat" in frame.columns:
            frame["job"], frame["file"] = csv.parent.name, csv.name.replace(".per_bag.csv", "")
            parts.append(frame)
    if not parts:
        return pd.DataFrame(), pd.DataFrame()
    table = pd.concat(parts, ignore_index=True)
    rows = []
    keys = ["job", "file", "joint_name", "regime"]
    for key, part in table[np.isfinite(table["K_true"])].groupby(keys, dropna=False):
        truth = part["K_eff"] if "K_eff" in part and np.isfinite(part["K_eff"]).any() else part["K_true"]
        rows.append({**dict(zip(keys, key)), "bags": len(part),
                     "logR2_K": _log_r2(part["K_hat"].to_numpy(), part["K_true"].to_numpy()),
                     "logR2_Keff": _log_r2(part["K_hat"].to_numpy(), truth.to_numpy()),
                     "med_relerr_K": float(np.nanmedian(np.abs(part["K_hat"] / part["K_true"] - 1))),
                     "med_sigma_logK": float(np.nanmedian(part.get("sigma_log_K", pd.Series(np.nan)))),
                     "med_M_ratio": float(np.nanmedian(part.get("M_phi_mean", np.nan) / part.get("M_true_mean", np.nan))),
                     "med_delta_amp": float(np.nanmedian(part.get("delta_amp_ratio", np.nan))),
                     "med_delta_nrmse": float(np.nanmedian(part.get("delta_nrmse", np.nan))),
                     "med_corr_delta (secondary)": float(np.nanmedian(part.get("delta_corr", np.nan))),
                     "observable_share": float(np.nanmean(part.get("observable", np.nan)))})
    return table, pd.DataFrame(rows)


def scatter(table: pd.DataFrame, out: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written = []
    for (job, file), part in table[np.isfinite(table["K_true"])].groupby(["job", "file"]):
        if job not in ("J3", "J8", "J9") or "-privileged" in file:
            continue
        joints = sorted(part["joint_name"].unique())
        fig, axes = plt.subplots(1, len(joints), figsize=(3 * len(joints), 3), squeeze=False)
        for ax, joint in zip(axes[0], joints):
            p = part[part["joint_name"] == joint]
            for regime, marker in (("full", "o"), ("real_like", "s"), ("off", "x")):
                q = p[p["regime"] == regime]
                ax.loglog(q["K_true"], q["K_hat"], marker, ms=3, label=regime, alpha=0.7)
            lo, hi = p[["K_true", "K_hat"]].min().min(), p[["K_true", "K_hat"]].max().max()
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
            ax.set_title(joint, fontsize=8)
            ax.set_xlabel("K true")
        axes[0][0].set_ylabel("K hat")
        axes[0][0].legend(fontsize=6)
        fig.tight_layout()
        path = out / f"K_scatter_{job}_{file}.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        written.append(str(path.relative_to(RESULTS)))
    return written


def band_figure(table: pd.DataFrame, out: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if "rmse_band" not in table:
        return []
    pivot = table[table["file"].str.endswith("production")].groupby(["job", "joint_name"])["rmse_band"].median().unstack()
    if not len(pivot):
        return []
    ax = pivot.T.plot.bar(figsize=(10, 3.5))
    ax.set_ylabel("median in-band RMSE [0.5, 2] f_n")
    plt.tight_layout()
    path = out / "band_errors.png"
    plt.savefig(path, dpi=130)
    plt.close()
    return [str(path.relative_to(RESULTS))]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", default="r8-campaign")
    args = parser.parse_args(argv)
    run = SIM / "runs" / args.run_id
    figures = RESULTS / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    lines = ["# 2M-ID weekend campaign: summary (descriptive)", "",
             "Generated by `scripts/r8_summary.py`; the owner decides what goes to `final_results/` and the paper.", ""]

    lines += ["## 1. What ran", ""]
    plan = RESULTS / "campaign_plan.json"
    if plan.exists():
        p = json.loads(plan.read_text())
        lines.append(f"Plan {p['created']}: {p['total_hours']:.1f} h projected at {p['parallel_seeds']} parallel seeds; "
                     f"dropped: {p['dropped'] or 'none'}.")
    progress = RESULTS / "progress.md"
    if progress.exists():
        lines += ["", progress.read_text().strip()]
    log = run / "campaign.log"
    if log.exists():
        fails = [line for line in log.read_text().splitlines() if "FAIL" in line or "STOP" in line or "SKIP" in line]
        lines += ["", f"Failures / skips ({len(fails)}):", ""] + [f"* `{f}`" for f in fails[:60]]
    if (RESULTS / "datasets.md").exists():
        lines += ["", "Datasets: see `datasets.md`."]

    sim = simulated_table(run)
    lines += ["", "## 2. Simulated (test / gain-shift / out-of-prior / modal files)", ""]
    lines.append(sim.round(4).to_markdown(index=False) if len(sim) else "pending")

    table, ident = identification_table(run)
    lines += ["", "## 3. Identification (K_hat vs truth per held-out robot)", "",
              "`logR2_Keff` is against the bag's effective linear pair (the identifiable target of a nonlinear spring); "
              "corr(delta_hat, delta) is secondary (00_plan D-4).", ""]
    lines.append(ident.round(3).to_markdown(index=False) if len(ident) else "pending")
    if len(table):
        from_consistency = table[np.isfinite(table["K_true"])].groupby(["job", "file", "tier", "joint_name"])["K_hat"]
        spread = (from_consistency.max() / from_consistency.min() - 1).groupby(["job", "file", "joint_name"]).median()
        lines += ["", "Consistency of K_hat across a test robot's bags (median max/min - 1):", "",
                  spread.rename("K_hat_spread").reset_index().round(3).to_markdown(index=False)]

    lines += ["", "## 4. Phase 0", ""]
    for path in sorted((RESULTS / "phase0").glob("*.json")):
        lines += [f"### {path.name}", "", "```json", path.read_text()[:4000], "```", ""]

    lines += ["## 5-6. Real files", ""]
    real = RESULTS / "real"
    for csv in sorted(real.glob("*.csv")):
        if csv.name.endswith(".per_bag.csv"):
            continue
        frame = _read(csv)
        if len(frame):
            cols = [c for c in ("checkpoint", "input", "exclude_head_s", "filter_time_step", "rmse") if c in frame] \
                + [c for c in frame.columns if c.startswith("rmse_joint_")]
            lines += [f"### {csv.stem}", "", frame[cols].round(4).to_markdown(index=False), ""]
    for js in sorted(real.glob("modal_fit_*.json")):
        lines += [f"### {js.stem}", "", "```json", js.read_text()[:3000], "```", ""]
    if not any(real.glob("*.csv")):
        lines.append("pending")

    written = []
    if len(table):
        written += scatter(table, figures) + band_figure(table, figures)
    lines += ["", "## 7. Figures", ""] + [f"* `{w}`" for w in written] + (["pending"] if not written else [])
    lines += ["", "## 8. Claims (evidence filled by the owner from the tables above)", "",
              "| claim | evidence | supported? |", "|---|---|---|",
              "| 2M-ID identifies per-robot K on observable joint-bags (log R^2 >= 0.6, median rel. err. <= 25 %) | Sec 3 | |",
              "| sigma is large on unobservable joint-bags and on the probe-off real files | Sec 3, 5 | |",
              "| anchoring (dq_joint residual) puts M_phi within x[0.7, 1.4] of the true M_jj | Sec 3 (M ratio) | |",
              "| 2M-ID torque RMSE no worse than Phase 1 (J2) | Sec 2 | |",
              "| the r8 data alone (J5) fixes the r7 two-mass behaviour | Sec 2, 3 | |",
              "| zero-shot real: any model beats the URDF | Sec 5-6 | |", ""]
    (RESULTS / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {RESULTS / 'SUMMARY.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
