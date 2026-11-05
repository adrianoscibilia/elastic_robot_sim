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

    python scripts/r8_summary.py [--run-id r8-campaign] [--config config/campaigns/r8_campaign.yaml]

r9 (``identified_filter/09`` Sec 3): ``--config`` (default: ``config/campaigns/<rN>_campaign.yaml`` of the run id,
when it exists) sets the results folder (``results_dir``); with a ``filter:`` block the summary adds the deployed
seed per job and the 04 B4 table on the gated joint-regimes (06 Sec 1.3), the other joint-regimes in a second table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SIM = Path(__file__).resolve().parents[1]
RESULTS = SIM / "REFACTOR_SPECS" / "identified_filter" / "results"
FINAL_STAGES = ("J", "T", "P", "R")   # r8_real_eval: the last stage a job's seed finishes
JOB_NAMES = {"J1": "2M-KALNN-T (teacher, privileged)", "J2": "2M-KALNN-PHYS", "J3": "2M-KALNN-ID",
             "J4": "KALNN residual (r7)", "J5": "2M-KALNN (r7 config)", "J6": "J3, lambda_p = 0",
             "J7": "J3, lambda_delta = 0", "J8": "J3, causal encoder", "J9": "2M-LNN-ID", "J10": "J3, q_dq residual",
             "J11": "KALNN residual + 2.1 + 2.2",
             # identified_filter/10: the LNN (MLP) counterparts and the r7-config residual on r9.
             "J12": "LNN residual + 2.1 + 2.2", "J13": "LNN residual (r7)", "J14": "2M-LNN-PHYS",
             "J15": "2M-LNN (r7 config)",
             # identified_filter/11 Part B: the anchored residuals (PHYS without the filter) and the continuations.
             "J16": "KALNN anchored residual (PHYS, no filter)", "J17": "LNN anchored residual (PHYS, no filter)",
             "J14x": "2M-LNN-PHYS, continued to 300 epochs", "J15x": "2M-LNN (r7 config), iiwa s2 continued"}


#: 06 Sec 1.3 (07 Sec 2): the joint-regimes the physics gate passed; B4 is read on these only.
GATED = {("iiwa_drive", "iiwa_A2", "full"), ("ur10", "shoulder_lift_joint", "full"),
         ("ur10", "shoulder_lift_joint", "real_like"), ("ur10", "elbow_joint", "full")}
#: 04 B4's thresholds.
B4_CORR, B4_FN, B4_RIGID = 0.7, 0.25, 0.9
#: The B4 rows: the identified jobs, the r7 two-mass; J11 (R) is the accuracy reference.
B4_JOBS = ("J1", "J2", "J3", "J5", "J14", "J15", "J14x", "J15x")


def _read(path):
    return pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()


def markdown(frame: pd.DataFrame) -> str:
    """A GitHub table of ``frame`` (the environment has no ``tabulate`` for ``DataFrame.to_markdown``)."""
    def cell(value):
        return "" if value is None or value != value else str(value)

    header = [str(c) for c in frame.columns]
    rows = ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header), *rows])


def deployed_checkpoints(run: Path) -> dict[str, dict]:
    """04 C1: per job and platform, the seed a deployment would pick: the final stage's best validation RMSE.

    ``checkpoint -> {job, platform, seed, best_val_rmse}`` of those seeds only.  Selection on validation, never
    on the test RMSE the tables report.
    """
    best: dict = {}
    for done in sorted((run / "jobs").glob("*/*/s*_*.done.json")):
        job, platform = done.parts[-3], done.parts[-2]
        seed, stage = done.name.split(".")[0].split("_")
        row = json.loads(done.read_text())
        try:
            value = float(row["best_val_rmse"])
        except (KeyError, TypeError, ValueError):
            continue
        best.setdefault((job, platform, seed), {})[stage] = {"checkpoint": row["checkpoint_path"], "value": value}
    out: dict = {}
    for (job, platform, seed), stages in best.items():
        final = stages[next(s for s in FINAL_STAGES if s in stages)] if any(s in stages for s in FINAL_STAGES) else None
        if final is None:
            continue
        current = out.get((job, platform))
        if current is None or final["value"] < current["best_val_rmse"]:
            out[(job, platform)] = {"job": job, "platform": platform, "seed": seed, "best_val_rmse": final["value"],
                                    "checkpoint": final["checkpoint"]}
    return {entry["checkpoint"]: entry for entry in out.values()}


def simulated_table(run: Path) -> pd.DataFrame:
    rows = []
    deployed = deployed_checkpoints(run)
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
                         "checkpoint": r["checkpoint"],
                         "deployed_seed": deployed.get(r["checkpoint"], {}).get("seed")})
    table = pd.DataFrame(rows)
    if not len(table):
        return table
    keys = ["file", "job", "model", "privileged_eval"]
    grouped = table.groupby(keys, dropna=False).agg(
        rmse_mean=("rmse", "mean"), seeds=("rmse", "size"), rigid=("rigid", "first"),
        nominal=("nominal", "first"), oracle=("oracle", "first")).reset_index()
    chosen = table[table["deployed_seed"].notna()].groupby(keys, dropna=False).agg(
        deployed_seed=("deployed_seed", "first"), rmse_deployed=("rmse", "first")).reset_index()
    columns = keys + ["deployed_seed", "rmse_deployed", "rmse_mean", "seeds", "rigid", "nominal", "oracle"]
    return grouped.merge(chosen, on=keys, how="left")[columns]


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
    if "checkpoint" in table.columns:  # 04 C1: the deployed seed of each job, not a pool with its failed seeds
        table = table[table["checkpoint"].isin(deployed_checkpoints(run))]
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


def campaign_settings(run_id: str, config: str | None) -> tuple[dict, Path | None]:
    """The campaign's settings: ``--config``, else ``config/campaigns/<rN>_campaign.yaml`` of ``rN-campaign``."""
    import yaml

    path = Path(config) if config else SIM / "config" / "campaigns" / f"{run_id.split('-')[0]}_campaign.yaml"
    if not path.is_absolute():
        path = SIM / path
    return (yaml.safe_load(path.read_text(encoding="utf-8")), path) if path.exists() else ({}, None)


def rigid_per_bag(settings: dict, platform: str) -> pd.DataFrame:
    """Per test bag and joint of the production file, the nominal rigid model's RMSE on the scored rows.

    The contract's ``rigid`` baseline (bare URDF RNEA at the recorded q, dq; ddq the consumer's SG derivative; rows
    with ``t - t_bag_start >= exclude_head_s``), per bag instead of pooled over the split, so it can be read per
    joint-regime.  Cached as ``rigid_per_bag_<platform>.csv`` beside the summary.
    """
    cache = RESULTS / f"rigid_per_bag_{platform}.csv"
    if cache.exists():
        return pd.read_csv(cache)
    sys.path[:0] = [str(SIM / "src"), str(SIM / "scripts")]
    import r8_campaign
    from diagnose_controller_modes import _load_asset
    from elastic_sim import identification as idn
    from elastic_sim.dataset import load_config
    from elastic_sim.diagnostics import savitzky_golay_acceleration

    paths = r8_campaign.platform_paths(settings, platform)
    contract = json.loads(paths["production"].with_suffix(".contract.json").read_text(encoding="utf-8"))
    priors = contract["transmission_priors"]
    joints = [j["joint"] for j in priors["joints"]]
    head = float(contract["baselines"]["two_mass"]["exclude_head_s"])
    step = float(priors["sample_time_step"])
    window = int(contract["differentiation"]["sg_window"])
    pin, model, data = idn.build_model(_load_asset(load_config(paths["config"]).asset))
    n = len(joints)
    columns = ["bag", "t", "split", "probe_regime", *[f"{p}{j}" for p in ("q", "dq", "ft") for j in range(n)]]
    frame = pd.read_parquet(paths["production"], columns=columns)
    rows = []
    for bag, part in frame[frame["split"] == "test"].groupby("bag", sort=False):
        q, dq, ft = (part[[f"{p}{j}" for j in range(n)]].to_numpy(float) for p in ("q", "dq", "ft"))
        ddq = savitzky_golay_acceleration(dq, step, window=window)
        t = part["t"].to_numpy(float)
        kept = (t - t[0]) >= head - 1e-9
        rigid = np.asarray([idn.inverse_dynamics(pin, model, data, q[i], dq[i], ddq[i]) for i in np.flatnonzero(kept)])
        error = rigid - ft[kept]
        for j, name in enumerate(joints):
            rows.append({"bag": bag, "joint_name": name, "regime": part["probe_regime"].iloc[0],
                         "rmse": float(np.sqrt(np.mean(error[:, j] ** 2)))})
    out = pd.DataFrame(rows)
    out.to_csv(cache, index=False)
    return out


def _pooled(rmse: pd.Series) -> float:
    """RMS over bags of the per-bag RMSE (bags of one file have about the same length)."""
    return float(np.sqrt(np.mean(np.square(rmse)))) if len(rmse) else float("nan")


def _corr_log(est, truth) -> float:
    ok = np.isfinite(est) & np.isfinite(truth) & (est > 0) & (truth > 0)
    if ok.sum() < 3 or np.std(np.log(est[ok])) == 0 or np.std(np.log(truth[ok])) == 0:
        return float("nan")
    return float(np.corrcoef(np.log(est[ok]), np.log(truth[ok]))[0, 1])


def b4_tables(run: Path, settings: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """``(deployed, gated, other)``: the deployed seed per job and platform, and 04 B4 per joint-regime.

    Per job (its deployed seed, on the production test bags; the teacher J1 with its privileged evaluation):
    the escape flag at the selected epoch (the final stage's ``escape_at_best``), corr(log K_hat_0, log K_0) and
    median |f_n_hat / f_n - 1| on the observable bags, and the RMSE against R (J11) and 0.9 x rigid.
    """
    deployed = pd.DataFrame(deployed_checkpoints(run).values())
    if not len(deployed):
        return deployed, pd.DataFrame(), pd.DataFrame()
    escape = {}
    for _, entry in deployed.iterrows():
        for stage in FINAL_STAGES:
            done = run / "jobs" / entry["job"] / entry["platform"] / f"{entry['seed']}_{stage}.done.json"
            if done.exists():
                row = json.loads(done.read_text())
                value = row.get("escape_at_best")
                try:
                    escape[(entry["job"], entry["platform"])] = json.loads(value) if isinstance(value, str) else value
                except json.JSONDecodeError:
                    escape[(entry["job"], entry["platform"])] = None
                deployed.loc[_, "stage"] = stage
                deployed.loc[_, "stopped_at"] = row.get("stopped_at")
                break
    chosen = set(deployed["checkpoint"])

    def per_bag(job, platform, privileged=False):
        tag = f"{platform}-production" + ("-privileged" if privileged else "")
        frame = _read(run / "eval" / job / f"{tag}.per_bag.csv")
        return frame[frame["checkpoint"].isin(chosen)] if len(frame) and "checkpoint" in frame else frame

    rows = []
    for platform in settings.get("platforms", {}):
        reference = per_bag("J11", platform)
        try:
            rigid = rigid_per_bag(settings, platform)
        except Exception as error:  # noqa: BLE001  (reported in the table, the summary still writes)
            print(f"rigid per bag {platform}: {type(error).__name__}: {error}", file=sys.stderr)
            rigid = pd.DataFrame(columns=["bag", "joint_name", "regime", "rmse"])
        for job in B4_JOBS:
            frame = per_bag(job, platform, privileged=job == "J1")
            if not len(frame):
                continue
            flagged = escape.get((job, platform))
            for (joint, regime), part in frame.groupby(["joint_name", "regime"], dropna=False):
                index = int(part["joint"].iloc[0])
                observable = part[part["observable"] > 0.5] if "observable" in part else part.iloc[:0]
                ref = reference[(reference["joint_name"] == joint) & (reference["regime"] == regime)] \
                    if len(reference) else reference
                rig = rigid[(rigid["joint_name"] == joint) & (rigid["regime"] == regime)]
                rmse, r_rmse, rigid_rmse = _pooled(part["rmse"]), _pooled(ref["rmse"]) if len(ref) else float("nan"), \
                    _pooled(rig["rmse"])
                corr = _corr_log(observable["K_hat"].to_numpy(float), observable["K_true"].to_numpy(float)) \
                    if "K_hat" in observable else float("nan")
                f_n = float(np.nanmedian(np.abs(observable["f_n_hat"] / observable["f_n_true"] - 1))) \
                    if "f_n_hat" in observable and len(observable) else float("nan")
                rows.append({"platform": platform, "joint": joint, "regime": regime, "job": job,
                             "model": JOB_NAMES.get(job, job), "bags": len(part), "observable_bags": len(observable),
                             "escape_at_selected": None if flagged is None else index in flagged,
                             "corr_logK": corr, "med_fn_relerr": f_n, "rmse": rmse, "rmse_R": r_rmse,
                             "rmse_rigid": rigid_rmse, "vs_R": rmse / r_rmse if r_rmse == r_rmse else float("nan"),
                             "vs_rigid": rmse / rigid_rmse if rigid_rmse == rigid_rmse else float("nan"),
                             "B4_ident": bool(corr >= B4_CORR and f_n <= B4_FN),
                             "B4_accuracy": bool(rmse <= r_rmse and rmse <= B4_RIGID * rigid_rmse),
                             "gated": (platform, joint, regime) in GATED})
    table = pd.DataFrame(rows)
    if not len(table):
        return deployed, table, table
    table["B4_no_escape"] = table["escape_at_selected"].eq(False)
    gated, other = table[table["gated"]].drop(columns="gated"), table[~table["gated"]].drop(columns="gated")
    return deployed, gated, other


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", default="r8-campaign")
    parser.add_argument("--config", default=None, help="the campaign settings (default: the run id's, if any)")
    args = parser.parse_args(argv)
    global RESULTS
    settings, config = campaign_settings(args.run_id, args.config)
    if settings.get("results_dir"):
        RESULTS = SIM / settings["results_dir"]
    run = SIM / "runs" / args.run_id
    figures = RESULTS / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    lines = ["# 2M-ID weekend campaign: summary (descriptive)", "",
             "Generated by `scripts/r8_summary.py`; the owner decides what goes to `final_results/` and the paper.",
             "Every model row is the job's deployed seed: its best validation RMSE (the test RMSE plays no part).", ""]

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

    if settings.get("filter"):
        deployed, gated, other = b4_tables(run, settings)
        lines += ["", "## r9: deployed seed per job (validation selection) and 04 B4", "",
                  f"Settings `{config.relative_to(SIM) if config else '?'}`. The deployed seed is the final stage's "
                  "best validation RMSE; `stopped_at` marks a stage ended by the hard stop (09 Sec 2).", ""]
        lines.append(markdown(deployed.round(4)) if len(deployed) else "pending")
        lines += ["", "### B4 on the gated joint-regimes (06 Sec 1.3: iiwa A2 `full`; UR10 lift `full`, `real_like`; "
                      "UR10 elbow `full`)", "",
                  f"Production test bags. Identification on the observable bags: corr(log K_hat_0, log K_0) >= {B4_CORR}"
                  f", median |f_n_hat / f_n - 1| <= {B4_FN:.0%}. Accuracy: RMSE <= R (J11) and <= {B4_RIGID} x rigid "
                  "(the bare URDF, per bag). Escape: the flag at the selected epoch. J1 is scored with the true K_0 "
                  "(privileged).", ""]
        lines.append(markdown(gated.round(3)) if len(gated) else "pending")
        lines += ["", "### The non-gated joint-regimes (reported, not judged)", ""]
        lines.append(markdown(other.round(3)) if len(other) else "pending")

    sim = simulated_table(run)
    lines += ["", "## 2. Simulated (test / gain-shift / out-of-prior / modal files)", ""]
    lines.append(markdown(sim.round(4)) if len(sim) else "pending")

    table, ident = identification_table(run)
    lines += ["", "## 3. Identification (K_hat vs truth per held-out robot)", "",
              "`logR2_Keff` is against the bag's effective linear pair (the identifiable target of a nonlinear spring); "
              "corr(delta_hat, delta) is secondary (00_plan D-4).", ""]
    lines.append(markdown(ident.round(3)) if len(ident) else "pending")
    if len(table):
        from_consistency = table[np.isfinite(table["K_true"])].groupby(["job", "file", "tier", "joint_name"])["K_hat"]
        spread = (from_consistency.max() / from_consistency.min() - 1).groupby(["job", "file", "joint_name"]).median()
        lines += ["", "Consistency of K_hat across a test robot's bags (median max/min - 1):", "",
                  markdown(spread.rename("K_hat_spread").reset_index().round(3))]

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
            frame = frame[frame["checkpoint"].isin(deployed_checkpoints(run))]  # 04 C1: one row per job, its deployed seed
            cols = [c for c in ("checkpoint", "input", "exclude_head_s", "filter_time_step", "rmse") if c in frame] \
                + [c for c in frame.columns if c.startswith("rmse_joint_")]
            lines += [f"### {csv.stem}", "", markdown(frame[cols].round(4)), ""]
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
