#!/usr/bin/env python3
"""identified_filter/03 F3: why the oracle fit fails, on a handful of ``full`` bags per platform.

For each robot's bag 0, two builds and two questions:

* the **plant** as configured, and a **control**: the same bags with a linear
  spring, no transmission error, no ripple and no noise.  A fit that fails on
  the control is a bug; a fit that passes there and fails on the plant means the
  linear-K target is ambiguous on this plant;
* the two-mass filter with the true ``tau_r`` and each candidate pair (sampled,
  ``K_eff``, the oracle fit): the lowest **in-band** ``tau_s`` RMSE is the target
  the dynamics support.  ``K_eff`` and the fit see a bag's first half and are
  scored on its second: on its own rows a fit wins by construction.

Privileged simulator truth throughout; nothing here touches a real file.

    uv run python tools/r8_oracle_diagnosis.py --bags 10
    uv run python tools/r8_oracle_diagnosis.py --platform ur10 --bags 2 --no-report   # look, write nothing
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
_REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [os.fspath(_REPO / "src"), os.fspath(_REPO / "scripts")]

import numpy as np
import pandas as pd
import yaml

from diagnose_controller_modes import _load_asset
from elastic_sim.dataset import assign_splits, generate, load_config
from elastic_sim.dataset_config import differentiation_policy
from elastic_sim.round7 import transmission_priors
from elastic_sim.round8 import SG_BAND_FRACTION, fit_usable, target_rmse, verify_bags

RESULTS = _REPO / "REFACTOR_SPECS" / "identified_filter" / "results"
CAMPAIGN = _REPO / "config" / "campaigns" / "r8_campaign.yaml"


def control_config(config):
    """Same robots, seeds and trajectories; a linear spring, no transmission error or ripple, no noise."""
    extras = replace(config.plant_extras, stiffness_factors=(), stiffness_breakpoints=(),
                     stiffness_breakpoints_torque=(), stiffness_breakpoints_effort_fraction=(),
                     transmission_error_amplitude=(), ripple_amplitude=0.0)
    return replace(config, plant_extras=extras, noise=replace(config.noise, enabled=False),
                   signals=replace(config.signals, target_source="clean"))


def diagnose(config, asset, bags: int, jobs: int, cache: dict) -> pd.DataFrame:
    """One row per bag, joint and plant."""
    labels = assign_splits(config.tiers, config.split)
    tiers = [t.name for t in config.tiers if not t.is_rigid and labels[t.name] == "train"][:bags]
    selection = replace(config, only_bags=tuple((t, 0) for t in tiers), rigid_reference=False)
    rows = []
    for plant, built in (("plant", selection), ("control", control_config(selection))):
        frame, manifest, _ = generate(built, asset, verbose=False, jobs=jobs, trajectory_cache=cache)
        contract = {"differentiation": differentiation_policy(built),
                    "transmission_priors": transmission_priors(manifest, asset,
                                                               burn_in_override_s=built.burn_in_override_s)}
        chunks = [group for _, group in frame.groupby("bag", sort=False)]
        heads = [group.iloc[:len(group) // 2] for group in chunks]
        middles = [float(group["t"].iloc[len(group) // 2] - group["t"].iloc[0]) for group in chunks]
        fixed = [[x] * len(chunks) for x in (manifest, contract, asset, built.round8)]
        with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as pool:
            fitted = pd.concat(pool.map(verify_bags, chunks, *fixed))
            held_out = pd.concat(pool.map(verify_bags, heads, *fixed)).to_dict("records")
            scored = list(pool.map(target_rmse, chunks, held_out, *fixed[:3], middles))
        names, f_sg = list(manifest["joint_names"]), SG_BAND_FRACTION / float(manifest["sample_time_step"])
        for record, head, scores in zip(fitted.to_dict("records"), held_out, scored):
            group = frame[frame["bag"] == record["bag"]]
            priors = contract["transmission_priors"]
            usable, usable_head = (fit_usable(row, group, names, priors, f_sg) for row in (record, head))
            for j, name in enumerate(names):
                sampled = float(group[f"stiffness__{name}"].iloc[0])
                ident, eff = record[f"ident_K__{name}"], record[f"K_eff__{name}"]
                damping = float(group[f"damping__{name}"].iloc[0])
                f_n = record[f"f_n_filter__{name}"]
                rows.append({"plant": plant, "bag": record["bag"], "joint": name, "K": sampled, "K_eff": eff,
                             "ident_K": ident, "f_n": f_n, "fitted": bool(f_n <= f_sg),
                             "D": damping, "ident_D": record[f"ident_D__{name}"], "D_eff": record[f"D_eff__{name}"],
                             "at_bound": bool(f_n <= f_sg and not usable[j]),
                             "held_out_at_bound": bool(f_n <= f_sg and not usable_head[j]),
                             "snr": record[f"elastic_snr__{name}"], **{f"rmse_{key}": scores[key][j] for key in scores}})
    return pd.DataFrame(rows)


def summarise(table: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Per plant and joint: the fit's error and each target's in-band RMSE; and the verdict."""
    fitted = table[table["fitted"]].copy()
    fitted["err_K"] = (fitted["ident_K"] / fitted["K"] - 1.0).abs()
    fitted["err_K_eff"] = (fitted["ident_K"] / fitted["K_eff"] - 1.0).abs()
    rmse = [c for c in fitted.columns if c.startswith("rmse_")]
    summary = (fitted.groupby(["plant", "joint"], sort=False)
               .agg(bags=("bag", "size"), at_bound=("at_bound", "mean"), err_K=("err_K", "median"),
                    err_K_eff=("err_K_eff", "median"), snr=("snr", "median"), **{c: (c, "median") for c in rmse})
               .reset_index())
    pass_rate = fitted.assign(ok=fitted["err_K_eff"] <= 0.10).groupby("plant")["ok"].mean().to_dict()
    # Median of the per-joint-bag ratio to the best target of that joint-bag, scale-free across joints.  On every
    # joint-bag the fit target is K_fit: the fit where usable, K_eff where it ran away.  On the usable subset it is
    # the fit itself (F3: ident_K "where not at a bound").
    on_plant = fitted[fitted["plant"] == "plant"]

    def ratios(part):
        return {k.removeprefix("rmse_"): v for k, v in part[rmse].div(part[rmse].min(axis=1), axis=0).median().items()}

    def means(part):
        """Scale-free mean of the per-joint-bag RMSE over the mean of the K_eff RMSE, per joint then across joints."""
        by_joint = part.groupby("joint")[rmse].mean()
        return {k.removeprefix("rmse_"): v for k, v in by_joint.div(by_joint["rmse_effective"], axis=0).mean().items()}

    usable = on_plant[~on_plant["held_out_at_bound"]]
    every, subset = means(on_plant), ratios(usable)
    return summary, {"control_pass_rate": pass_rate.get("control", float("nan")),
                     "plant_pass_rate": pass_rate.get("plant", float("nan")),
                     "at_bound_fraction": float(on_plant["at_bound"].mean()), "usable_joint_bags": int(len(usable)),
                     "joint_bags": int(len(on_plant)), "ratio_usable_subset": subset, "ratio_all": ratios(on_plant),
                     "mean_rmse_vs_effective_all": every, "best_target": min(every, key=every.get)}


def markdown(platform: str, table: pd.DataFrame, summary: pd.DataFrame, verdict: dict) -> list[str]:
    lines = [f"### {platform}", "",
             f"Fit within 10 % of K_eff: control **{verdict['control_pass_rate']:.2f}**, plant "
             f"**{verdict['plant_pass_rate']:.2f}** of the fitted joint-bags; {verdict['at_bound_fraction']:.2f} of the "
             f"plant's ran away.  Held-out in-band RMSE, mean per joint relative to K_eff, over all "
             f"{verdict['joint_bags']} joint-bags: "
             f"{', '.join(f'{k} {v:.3f}' for k, v in verdict['mean_rmse_vs_effective_all'].items())}; best: "
             f"**{verdict['best_target']}**.  Median ratio to the per-joint-bag best, all: "
             f"{', '.join(f'{k} {v:.3f}' for k, v in verdict['ratio_all'].items())}; on the "
             f"{verdict['usable_joint_bags']} whose held-out fit is usable: "
             f"{', '.join(f'{k} {v:.3f}' for k, v in verdict['ratio_usable_subset'].items())}.", "",
             "| plant | joint | bags | at bound | K err vs sampled | K err vs K_eff | SNR | "
             + " | ".join(c.removeprefix("rmse_") + " RMSE [Nm]" for c in summary.columns if c.startswith("rmse_")) + " |",
             "|---|---|---|---|---|---|---|" + "---|" * sum(c.startswith("rmse_") for c in summary.columns)]
    for _, row in summary.iterrows():
        rmse = " | ".join(f"{row[c]:.3g}" for c in summary.columns if c.startswith("rmse_"))
        lines.append(f"| {row['plant']} | {row['joint']} | {row['bags']} | {row['at_bound']:.2f} | {row['err_K']:.3g} | "
                     f"{row['err_K_eff']:.3g} | {row['snr']:.3g} | {rmse} |")
    lines += ["", "Per bag, plant (sampled K, K_eff, oracle fit, f_n, on a bound, SNR):", "",
              "| bag | joint | K | K_eff | ident_K | f_n [Hz] | at bound | SNR |", "|---|---|---|---|---|---|---|---|"]
    for _, row in table[(table["plant"] == "plant") & table["fitted"]].iterrows():
        lines.append(f"| {row['bag']} | {row['joint']} | {row['K']:.4g} | {row['K_eff']:.4g} | {row['ident_K']:.4g} | "
                     f"{row['f_n']:.3g} | {row['at_bound']} | {row['snr']:.3g} |")
    return lines + [""]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--platform", nargs="+", default=None, help="default: every platform of the campaign")
    parser.add_argument("--bags", type=int, default=10)
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--no-report", action="store_true", help="print only; write nothing")
    args = parser.parse_args(argv)
    settings = yaml.safe_load(CAMPAIGN.read_text(encoding="utf-8"))
    out = ["# Oracle-fit diagnosis (identified_filter/03 F3)", ""]
    verdicts, tables = {}, []
    for platform in args.platform or settings["platforms"]:
        config = load_config(_REPO / settings["platforms"][platform]["config"])
        table = diagnose(config, _load_asset(config.asset), args.bags, args.jobs, {})
        summary, verdicts[platform] = summarise(table)
        tables.append(table.assign(platform=platform))
        section = markdown(platform, table, summary, verdicts[platform])
        print("\n".join(section), flush=True)
        out += section
    if not args.no_report:
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / "oracle_diagnosis.md").write_text("\n".join(out), encoding="utf-8")
        pd.concat(tables).to_csv(RESULTS / "oracle_diagnosis.csv", index=False)
        (RESULTS / "oracle_diagnosis.json").write_text(json.dumps(verdicts, indent=1), encoding="utf-8")
    print(json.dumps(verdicts, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
