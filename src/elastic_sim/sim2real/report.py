"""Run folder, report and comparisons (S2R_01 T-4.5, S2R_02 A-2/A-3, S2R_04 A-11).

A calibration run folder ``data/sim2real/<robot>/<run_id>/`` holds:

==========================  ==================================================
``config.yaml``             the resolved config (+ the command-line overrides)
``inputs.json``             runs (path, dataset sha256, plan digests), code
                            digest ``erd-src-digest/2``, prior overlay sha256,
                            optimizer, seed, budget, command line, versions
``registry.yaml``           the registry the run started from
``identifiability_theta0.*``  T-4.3 at the start point
``identifiability.*``       T-4.3 at the result (d_post per parameter) + A-12
``history.jsonl``           one line per evaluation (problem.py); each A-12
                            refit has its own ``history_refit<k>.jsonl``
``working.yaml``            A-3 working overlay, rewritten on each val improvement
``best.yaml``               the result as an overlay: the stage's full set,
                            ``calibrated: true | false``, d_post and d, the
                            partners of the not-calibrated ones (S2R_08)
``metrics.json``            A-2 suite per split at theta0 and at the best theta
``replay_<split>.parquet``  the best theta's replay (result.py columns)
``figures/``                ft real vs sim, history, friction signature
``report.md``               :func:`render_report`, rebuilt from the files above
==========================  ==================================================
"""

from __future__ import annotations

import importlib.util
import json
import math
import platform
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from .config import REPO_ROOT

DIGEST_SOURCE = REPO_ROOT / "src" / "ros2" / "erd_recording" / "erd_recording" / "code_digest.py"
#: A free parameter within 2 % of the box edge (|z| >= 0.98) is "at a bound":
#: its value is set by the prior's range, not by the data or the prior's mean.
AT_BOUND_Z = 0.98


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def code_digest() -> dict[str, Any]:
    """``erd-src-digest/2`` of the repository (loaded from its standard-library
    source file, so ``elastic_sim`` doesn't import the ROS package)."""
    if not DIGEST_SOURCE.is_file():
        return {"method": "erd-src-digest/2", "digest": None, "note": f"{DIGEST_SOURCE} missing"}
    spec = importlib.util.spec_from_file_location("_erd_code_digest", DIGEST_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {"method": module.METHOD, "digest": module.source_digest(REPO_ROOT)}


def run_inputs(config: Any, runs: Mapping[str, Any], **extra: Any) -> dict[str, Any]:
    from .real_runs import file_sha256

    out: dict[str, Any] = {"config": str(config.source), "runs": {}, "code": code_digest(),
                           "python": platform.python_version(), "versions": _versions()}
    for run_id, run in runs.items():
        plans = {}
        for key in ("segments", "standstill_segments", "identify_segments"):
            for entry in run.plan_manifest.get(key, []):
                plans[entry["segment_id"]] = entry["digest"]
        out["runs"][run_id] = {"path": str(run.path), "dataset_sha256": run.dataset_sha256, "plan_digests": plans,
                               "sidecar_key": (run.sidecar or {}).get("key")}
    prior = extra.get("prior_from")
    if prior:
        extra["prior_from_sha256"] = file_sha256(Path(prior))
    out.update({k: (str(v) if isinstance(v, Path) else v) for k, v in extra.items()})
    return out


def _versions() -> dict[str, str | None]:
    import importlib.metadata as metadata

    out = {}
    for name in ("numpy", "scipy", "pin", "mujoco", "cma", "scikit-optimize", "skrl", "torch"):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    tmp.replace(path)
    return path


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def overlay_annotations(identifiability: Mapping[str, Any], names: Sequence[str] | None = None,
                        not_calibrated: Mapping[str, str] | None = None, *, zeroed: Sequence[str] = (),
                        partners: Mapping[str, Sequence[Mapping[str, Any]]] | None = None
                        ) -> dict[str, dict[str, Any]]:
    """Per overlay parameter: ``calibrated`` (S2R_08 A-12') with the reason
    and the partners when not, and the A-11 shares (d_post, the data share
    since F-2; d, the diagonal diagnostic) where the identifiability block
    has them."""
    rows = {r["name"]: r for r in identifiability["parameters"]}
    frozen = {e["name"]: e["d_post"] for e in (identifiability.get("a12") or {}).get("frozen", [])}
    names = [r["name"] for r in identifiability["parameters"] if r["free"]] if names is None else names
    not_calibrated = dict(not_calibrated or {})
    out = {}
    for name in names:
        entry: dict[str, Any] = {"calibrated": name not in not_calibrated}
        if name in not_calibrated:
            entry["not_calibrated_reason"] = not_calibrated[name]
        if partners and name in partners:
            entry["partners"] = [{"name": e["name"], "corr": round(float(e["corr"]), 4)} for e in partners[name]]
        if name in frozen:
            entry["frozen_d_post"] = round(float(frozen[name]), 6)
        if name in zeroed:
            entry["calibrated_by"] = "session zero (S2R_06 A-13)"
        row = rows.get(name)
        if row is not None:
            entry["data_share"] = round(float(row["data_share"]), 6)
            if row["data_share_posterior"] is not None:
                entry["data_share_posterior"] = round(float(row["data_share_posterior"]), 6)
                entry["prior_determined"] = bool(row["prior_determined"])
        out[name] = entry
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def write_figures(folder: Path, results_by_split: Mapping[str, Sequence[Any]], history: Sequence[Mapping[str, Any]],
                  joint_names: Sequence[str]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = folder / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    written = []
    for split, results in results_by_split.items():
        dynamic = [r for r in results if r.kind != "hold"]
        if not dynamic:
            continue
        errors = [float(np.sqrt(np.mean(r.residual ** 2))) for r in dynamic]
        chosen = dynamic[int(np.argsort(errors)[len(errors) // 2])]
        fig, axes = plt.subplots(len(joint_names), 2, figsize=(13, 2.0 * len(joint_names)), sharex=True)
        for j, name in enumerate(joint_names):
            axes[j, 0].plot(chosen.t, chosen.ft[:, j], lw=0.8, label="real ft")
            axes[j, 0].plot(chosen.t, chosen.ft_sim[:, j], lw=0.8, label="sim ft")
            axes[j, 0].set_ylabel(f"{name}\nN m")
            axes[j, 1].plot(chosen.t, chosen.residual[:, j], lw=0.6, color="C3")
            axes[j, 1].set_ylabel("sim - real")
        axes[0, 0].legend(loc="upper right", fontsize=7)
        axes[0, 0].set_title(f"{split}: {chosen.run_id}:{chosen.bag} (median-RMSE bag)")
        axes[-1, 0].set_xlabel("t, s")
        axes[-1, 1].set_xlabel("t, s")
        fig.tight_layout()
        path = figures / f"ft_{split}.png"
        fig.savefig(path, dpi=90)
        plt.close(fig)
        written.append(path.name)
    ok = [h for h in history if h.get("error") is None]
    if ok:
        fig, ax = plt.subplots(figsize=(8, 4))
        index = [h["i"] for h in ok]
        ax.semilogy(index, np.minimum.accumulate([h["loss"] for h in ok]), label="train loss (best so far)")
        val = [h["val_loss"] for h in ok if h.get("val_loss") is not None]
        if val:
            ax.semilogy([h["i"] for h in ok if h.get("val_loss") is not None], np.minimum.accumulate(val),
                        label="val data loss (best so far)")
        ax.set_xlabel("evaluation")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures / "history.png", dpi=90)
        plt.close(fig)
        written.append("history.png")
    return written


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def render_report(folder: str | Path) -> str:
    """report.md from the run folder's files alone (``--report-only``)."""
    folder = Path(folder)
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    inputs = json.loads((folder / "inputs.json").read_text(encoding="utf-8"))
    metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    identifiability = json.loads((folder / "identifiability.json").read_text(encoding="utf-8"))
    summary = metrics.get("summary", {})
    joints = metrics.get("joint_names") or [f"J{j + 1}" for j in range(7)]
    lines = [f"# Calibration run `{folder.name}`", "",
             f"* config: `{inputs.get('config')}`; stage `{inputs.get('stage')}`; mode {config.get('replay', {}).get('mode')}",
             f"* optimizer: **{inputs.get('method')}**, seed {inputs.get('seed')}, budget {inputs.get('budget')}; "
             f"{summary.get('evaluations')} evaluations, result #{summary.get('best_index')} of "
             f"{summary.get('best_phase', 'the fit')} (selected on **{summary.get('selection', 'val')}**), "
             f"wall {summary.get('wall_s', 0):.1f} s",
             f"* code digest `{(inputs.get('code') or {}).get('digest')}` (erd-src-digest/2); prior from "
             f"`{inputs.get('prior_from') or 'base config'}`",
             f"* calibrated label: mode **{_mode(config)}** "
             f"(label: S2R_08 A-12', freeze_refit: S2R_06 A-12), d_post >= "
             f"{(config.get('identifiability') or {}).get('d_post_min', 0.9):g}", ""]
    lines += ["## Split (S2R_06 A-13, per-session zero)", "", "| split | patterns |", "|---|---|"]
    for split, patterns in (config.get("split") or {}).items():
        lines.append(f"| {split} | {', '.join(f'`{p}`' for p in patterns) or '–'} |")
    lines += ["", "Training never sees test-run data except the sensor zero: after the stage fit only the zero runs' "
              "`sensor.ft.*.offset@<run>` are fitted on the `zero` bags, everything else frozen.", ""]
    lines += _phases_section(metrics)
    lines += _zero_section(metrics, joints)
    loss = metrics.get("loss", {})
    if loss:
        scale = loss["scales"]["ft"]
        lines += ["## Loss normalisation (S2R_01 §6)", "",
                  "| | " + " | ".join(joints) + " |", "|---|" + "---|" * len(joints),
                  "| s_j, N m | " + " | ".join(f"{v:.3g}" for v in scale["scale"]) + " |",
                  "| set by | " + " | ".join(scale["binding"]) + " |", "",
                  f"lambda = {loss['prior_weight']:.3g} ("
                  f"{'auto x %g, factor/N' % loss.get('prior_weight_factor', 1.0) if loss['prior_weight_auto'] else 'set'}; "
                  f"N = {loss['n_residuals']} residual entries).", ""]
    lines += _torque_section(metrics, joints)
    lines += _slopes_section(metrics, joints)
    lines += _friction_section(metrics, joints)
    lines += _spectrum_section(metrics, joints)
    lines += _plausibility_section(metrics, identifiability, bound_hits(folder))
    lines += ["## Identifiability at the result", "",
              f"Condition number {identifiability['condition_number']:.3g}, collinearity index "
              f"{identifiability['collinearity_index']:.3g}; {identifiability['n_directions_data_determined']} of "
              f"{identifiability['n_free']} directions data-determined; free parameters with d_post < "
              f"{(identifiability.get('a12') or {}).get('d_post_min', 0.9):g}: "
              f"{identifiability['n_prior_determined_posterior']} (diagonal d, diagnostic: "
              f"{identifiability.get('n_prior_determined_diagonal', '–')}). Full table: `identifiability.md`.", ""]
    if (folder / "sensitivity.md").is_file():
        lines += [(folder / "sensitivity.md").read_text(encoding="utf-8"), ""]
    if identifiability.get("gain_mass_pairs"):
        lines += ["Sensor gain vs link mass (S2R_04 R-3):", "", "| gain | most collinear mass | cos | d gain | d mass |",
                  "|---|---|---|---|---|"]
        share = {r["name"]: r["data_share"] for r in identifiability["parameters"]}
        for p in identifiability["gain_mass_pairs"]:
            lines.append(f"| `{p['gain']}` | `{p['mass']}` | {p['cos']:+.3f} | {share[p['gain']]:.3f} | "
                         f"{share[p['mass']]:.3f} |")
        lines.append("")
    lines += ["## Files", "", "`best.yaml` (the stage's full parameter set, each `calibrated: true | false`, with d_post "
              "and d), `working.yaml`, `metrics.json`, `replay_<split>.parquet`, `history.jsonl` (+ "
              "`history_refit<k>.jsonl`), `identifiability{,_theta0}.{json,md}`, `figures/`.", ""]
    return "\n".join(lines)


def _mode(config: Mapping[str, Any]) -> str:
    block = config.get("identifiability") or {}
    return block.get("mode") or ("freeze_refit" if block.get("freeze_then_refit") else "none")


def _fmt(value: Any, spec: str = ".3f") -> str:
    return "–" if value is None else format(value, spec)


def _torque_section(metrics: Mapping[str, Any], joints: Sequence[str]) -> list[str]:
    lines = ["## (a) Torque error, N m (FR_05-comparable)", "",
             "Median over bags of the per-bag RMSE over joints and kept rows; contract baselines are each run's "
             "own (all 10 excitations, 1 kHz). R0 rows: decimated per the config. theta0 is the stage's start "
             "with the same session zero as the result (A-13).", "",
             "| split | theta0 | best | contract rigid | contract state-fit (in-sample) | contract tau-fit (in-sample) |",
             "|---|---|---|---|---|---|"]
    for split in ("train", "val", "test"):
        before = metrics.get("theta0", {}).get(split, {}).get("torque")
        after = metrics.get("best", {}).get(split, {}).get("torque")
        if not after:
            continue
        baselines = metrics["best"][split].get("baselines") or {}
        rigid = "/".join(_fmt(b.get("baseline_rms_rigid"), ".2f") for b in baselines.values()) or "–"
        state = "/".join(_fmt(b.get("baseline_rms_state"), ".2f") for b in baselines.values()) or "–"
        tau = "/".join(_fmt(b.get("baseline_rms_tau"), ".2f") for b in baselines.values()) or "–"
        lines.append(f"| {split} | {_fmt(before and before['rmse_median_over_bags'])} | "
                     f"**{_fmt(after['rmse_median_over_bags'])}** | {rigid} | {state} | {tau} |")
    if metrics.get("best_all_rows"):
        lines += ["", "On **every** row (the stage fitted a row subset; these are the FR_05-comparable numbers):", "",
                  "| split | theta0 | best |", "|---|---|---|"]
        for split in ("train", "val", "test"):
            before = (metrics.get("theta0_all_rows", {}).get(split) or {}).get("torque")
            after = (metrics["best_all_rows"].get(split) or {}).get("torque")
            if after:
                lines.append(f"| {split} | {_fmt(before and before['rmse_median_over_bags'])} | "
                             f"**{_fmt(after['rmse_median_over_bags'])}** |")
    strict = (metrics.get("best_strict") or {}).get("test", {}).get("torque")
    if strict:
        zeroed = (metrics["best"].get("test") or {}).get("torque") or {}
        lines += ["", f"Test with the session zero against the **strict** variant (the test run's offsets = run "
                  f"{metrics['best_strict']['offsets_from']}'s: session drift counted as error, S2R_06 A-13):", "",
                  "| | zeroed | strict |", "|---|---|---|",
                  f"| test RMSE (median over bags) | {_fmt(zeroed.get('rmse_median_over_bags'))} | "
                  f"{strict['rmse_median_over_bags']:.3f} |",
                  "| per joint | " + " ".join(f"{v:.3f}" for v in zeroed.get("per_joint_rmse", [])) + " | "
                  + " ".join(f"{v:.3f}" for v in strict["per_joint_rmse"]) + " |"]
    other = metrics.get("val_selected")
    if other:
        lines += ["", f"The val-selected point (evaluation #{other['index']}) against this end point (F-6: S0 is a "
                  "warm start):", "", "| split | end point (used) | val-selected |", "|---|---|---|"]
        for split in ("train", "val", "test"):
            a = (metrics["best"].get(split) or {}).get("torque")
            b = (other.get(split) or {}).get("torque")
            if a and b:
                lines.append(f"| {split} | {a['rmse_median_over_bags']:.3f} | {b['rmse_median_over_bags']:.3f} |")
    converged = metrics.get("converged")
    if converged:
        lines += ["", f"The optimizer's end point (lowest train loss, evaluation #{converged['index']}) against the "
                  "val-selected theta:", "", "| split | selected (val) | converged (train) |", "|---|---|---|"]
        for split in ("train", "val", "test"):
            a = (metrics["best"].get(split) or {}).get("torque")
            b = (converged.get(split) or {}).get("torque")
            if a and b:
                lines.append(f"| {split} | {a['rmse_median_over_bags']:.3f} | {b['rmse_median_over_bags']:.3f} |")
    lines += ["", "Per joint, RMSE (NRMSE), at the best theta:", "",
              "| split | " + " | ".join(joints) + " |", "|---|" + "---|" * len(joints)]
    for split in ("train", "val", "test"):
        after = metrics.get("best", {}).get(split, {}).get("torque")
        before = metrics.get("theta0", {}).get(split, {}).get("torque")
        if not after:
            continue
        lines.append(f"| {split} | " + " | ".join(f"{r:.3f} ({n:.2f})" for r, n in
                                                  zip(after["per_joint_rmse"], after["per_joint_nrmse"])) + " |")
        if before:
            lines.append(f"| {split}, theta0 | " + " | ".join(f"{r:.3f}" for r in before["per_joint_rmse"]) + " |")
    return lines + [""]


def _slopes_section(metrics: Mapping[str, Any], joints: Sequence[str]) -> list[str]:
    lines = ["## (b) Hold gravity slopes (target 1.00)", ""]
    slopes0 = (metrics.get("theta0", {}).get("holds") or {}).get("hold_slopes") or {}
    slopes1 = (metrics.get("best", {}).get("holds") or {}).get("hold_slopes") or {}
    if not slopes1:
        return lines + ["No holds in this run.", ""]
    other = metrics.get("converged") or metrics.get("val_selected") or {}
    slopes2 = (other.get("holds") or {}).get("hold_slopes") or {}
    used = (metrics.get("summary") or {}).get("selection", "val")
    label = "converged (train)" if metrics.get("converged") else "val-selected"
    lines += [slopes1.get("statistic", ""), "",
              f"| run | joint | theta0 | result ({used}) | r2 (result) | {label} |", "|---|---|---|---|---|---|"]
    for run, block in slopes1.items():
        if not isinstance(block, Mapping) or "slope" not in block:
            continue
        before = (slopes0.get(run) or {}).get("slope", {})
        after = (slopes2.get(run) or {}).get("slope", {})
        for j, value in block["slope"].items():
            lines.append(f"| {run} | {joints[int(j)]} | {_fmt(before.get(j) if isinstance(before, Mapping) else None, '.4f')}"
                         f" | **{value:.4f}** | {block['r2'][j]:.4f} | "
                         f"{_fmt(after.get(j) if isinstance(after, Mapping) else None, '.4f')} |")
    return lines + [""]


def _friction_section(metrics: Mapping[str, Any], joints: Sequence[str]) -> list[str]:
    lines = ["## (c) Friction signature", "",
             "Largest |mean residual| over the velocity bins (N m), and the odd (Coulomb-like) component, per joint.",
             "", "| split | | " + " | ".join(joints) + " |", "|---|---|" + "---|" * len(joints)]
    for split in ("train", "val", "test"):
        for label, key in (("theta0", "theta0"), ("best", "best")):
            block = metrics.get(key, {}).get(split, {}).get("friction_signature")
            if not block:
                continue
            lines.append(f"| {split} | {label} max | " + " | ".join(_fmt(v) for v in block["max_abs_bin_mean"]) + " |")
            lines.append(f"| {split} | {label} odd | " + " | ".join(_fmt(v, "+.3f") for v in block["odd_component"])
                         + " |")
    return lines + [""]


def _spectrum_section(metrics: Mapping[str, Any], joints: Sequence[str]) -> list[str]:
    lines = ["## (d) Residual spectrum at the best theta", "",
             "Mean-square residual per band, N m^2 (fraction of the joint's total).", "",
             "| split | band, Hz | " + " | ".join(joints) + " |", "|---|---|" + "---|" * len(joints)]
    for split in ("train", "val", "test"):
        block = metrics.get("best", {}).get(split, {}).get("spectrum")
        if not block:
            continue
        if not block.get("bags"):
            lines.append(f"| {split} | – | no uniformly sampled bag (row filter) |" + " |" * (len(joints) - 1))
            continue
        for (lo, hi), power, fraction in zip(block["bands_hz"], block["power_nm2"], block["fraction"]):
            band = f"{lo:g}–{hi:g}" if hi is not None else f"> {lo:g}"
            lines.append(f"| {split} | {band} | " + " | ".join(f"{p:.3g} ({_fmt(f, '.2f')})"
                                                                 for p, f in zip(power, fraction)) + " |")
    return lines + [""]


def _plausibility_section(metrics: Mapping[str, Any], identifiability: Mapping[str, Any],
                          hits: Sequence[Mapping[str, Any]] = ()) -> list[str]:
    block = metrics.get("best", {}).get("plausibility")
    if not block:
        return []
    at_bound = {h["name"] for h in hits}
    rows = block["parameters"]
    calibrated = [r for r in rows if r.get("calibrated", True)]
    lines = ["## (f) Plausibility: calibrated vs not calibrated", "",
             f"Physically consistent: **{block['physical']['consistent']}** {block['physical']['problems'] or ''}. "
             f"{block['rule']}. **{len(calibrated)} of {len(rows)} parameters calibrated**; "
             f"{block.get('n_not_calibrated', 0)} **not calibrated** (listed second, with the reason and, for those "
             "the data do not separate, the partners to change them with); "
             f"{block['n_prior_determined']} calibrated ones with d_post below the threshold; "
             f"{len(at_bound & {r['name'] for r in calibrated})} calibrated ones **at a bound** of their box "
             f"(|z| >= {AT_BOUND_Z}).", "",
             "| parameter | calibrated | value | prior mean | dist. (sigma) | d_post | d (diag.) | | partners |",
             "|---|---|---|---|---|---|---|---|---|"]
    for row in sorted(rows, key=lambda r: not r.get("calibrated", True)):
        flag = ", ".join(f for f, on in (("prior-determined", row["prior_determined"] and row.get("calibrated", True)),
                                         ("**at bound**", row["name"] in at_bound)) if on)
        if not row.get("calibrated", True):
            flag = row.get("not_calibrated_reason") or "not calibrated"
        lines.append(f"| `{row['name']}` | {'yes' if row.get('calibrated', True) else '**no**'} | {row['value']:.6g} | "
                     f"{row['mean']:.6g} | {_fmt(row['prior_distance_sigma'], '+.2f')} | "
                     f"{_fmt(row['data_share_posterior'])} | {_fmt(row['data_share'])} | {flag} | "
                     f"{_partners(row.get('partners')) if not row.get('calibrated', True) else ''} |")
    lines += ["", "Inertial moves against the asset URDF:", "", "| link | mass, kg | Δmass | Δcom, mm | inertia scale |",
              "|---|---|---|---|---|"]
    for link, move in block["inertial_moves_vs_asset"].items():
        lines.append(f"| {link} | {move['mass_kg']:.3f} | {move['delta_mass_pct']:+.1f} % | "
                     f"{', '.join(f'{1e3 * v:+.1f}' for v in move['delta_com_m'])} | {move['inertia_scale']:.3f} |")
    return lines + [""]


def _partners(partners: Sequence[Mapping[str, Any]] | None) -> str:
    from .identifiability import format_partners

    return format_partners(partners) if partners is not None else ""


def _phases_section(metrics: Mapping[str, Any]) -> list[str]:
    phases = (metrics.get("summary") or {}).get("phases") or []
    if len(phases) <= 1 and not (metrics.get("summary") or {}).get("n_frozen_a12"):
        return []
    lines = ["## Fit and refits (S2R_06 A-12, freeze_refit mode)", "", "| phase | free parameters | evaluations | selected | train loss | "
             "val data loss | wall s |", "|---|---|---|---|---|---|---|"]
    for p in phases:
        lines.append(f"| {p['phase']} | {p['n_free']} | {p['evaluations']} | #{p['selected_index']} | "
                     f"{_fmt(p['train_loss'], '.5g')} | {_fmt(p['val_loss'], '.5g')} | {p['wall_s']:.1f} |")
    return lines + ["", "The frozen parameters and the d_post that froze each: `identifiability.md`.", ""]


def _zero_section(metrics: Mapping[str, Any], joints: Sequence[str]) -> list[str]:
    zero = metrics.get("session_zero") or {}
    if not zero.get("names"):
        return []
    lines = ["## Session zero (S2R_06 A-13)", "",
             f"Fitted on {', '.join(f'`{b}`' for b in zero['bags'])} ({zero['nfev']} residual evaluations), every other "
             "parameter at the stage result:", "",
             "| | " + " | ".join(joints) + " |", "|---|" + "---|" * len(joints),
             "| offset before, N m | " + " | ".join(f"{v:+.3f}" for v in zero["before"].values()) + " |",
             "| offset after, N m | " + " | ".join(f"{v:+.3f}" for v in zero["after"].values()) + " |",
             "| hold RMS before | " + " | ".join(f"{v:.3f}" for v in zero["hold_rms_before"]) + " |",
             "| hold RMS after | " + " | ".join(f"{v:.3f}" for v in zero["hold_rms_after"]) + " |"]
    if zero.get("at_bound"):
        lines += ["", "**At a bound:** " + ", ".join(f"`{n}`" for n in zero["at_bound"])]
    return lines + [""]


def bound_hits(folder: Path, theta: Mapping[str, float] | None = None) -> list[dict[str, Any]]:
    """Free parameters of a run at a bound of their box (from ``registry.yaml``
    and the best theta of ``metrics.json``)."""
    from .parameters import Registry

    registry = Registry.from_yaml((folder / "registry.yaml").read_text(encoding="utf-8"))
    if theta is None:
        theta = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))["best_theta"]
    hits = []
    for name in registry.free_names:
        spec = registry.specs[name]
        z = spec.to_unit(float(theta[name]))
        if abs(z) >= AT_BOUND_Z:
            hits.append({"name": name, "value": float(theta[name]), "bound": spec.upper if z > 0 else spec.lower,
                         "z": z})
    return hits


def read_history(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in rows if "i" in r]


# ---------------------------------------------------------------------------
# Optimizer comparison (A-2)
# ---------------------------------------------------------------------------


def compare_runs(folders: Sequence[str | Path], *, split: str = "test") -> tuple[str, list[dict[str, Any]]]:
    """Rank finished runs on the A-2 metrics on ``split``; sample efficiency
    (evaluations to reach 1 % of the run's final best val loss) is secondary."""
    rows = []
    for folder in map(Path, folders):
        metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
        inputs = json.loads((folder / "inputs.json").read_text(encoding="utf-8"))
        history = read_history(folder / "history.jsonl")
        best = metrics["best"]
        torque = best.get(split, {}).get("torque") or {}
        slopes = (best.get("holds") or {}).get("hold_slopes") or {}
        plaus = best.get("plausibility") or {}
        val = [h["val_loss"] for h in history if h.get("val_loss") is not None and h.get("error") is None]
        running = np.minimum.accumulate(val) if val else np.array([])
        reach = int(np.argmax(running <= running[-1] * 1.01)) + 1 if len(running) else None
        distances = [abs(r["prior_distance_sigma"]) for r in plaus.get("parameters", [])
                     if r["prior_distance_sigma"] is not None]
        hits = bound_hits(folder, metrics["best_theta"])
        rows.append({"run": folder.name, "method": inputs.get("method"), "seed": inputs.get("seed"),
                     "at_bound": [h["name"] for h in hits],
                     "evaluations": metrics.get("summary", {}).get("evaluations"),
                     "rmse": torque.get("rmse_median_over_bags"), "per_joint_rmse": torque.get("per_joint_rmse"),
                     "slopes": {run: b.get("slope") for run, b in slopes.items() if isinstance(b, Mapping)},
                     "val_loss": float(running[-1]) if len(running) else None, "evals_to_1pct": reach,
                     "max_prior_distance_sigma": max(distances) if distances else None,
                     "n_prior_determined": plaus.get("n_prior_determined"),
                     "n_calibrated": sum(1 for p in plaus.get("parameters", []) if p.get("calibrated", True)),
                     "strict_rmse": (((metrics.get("best_strict") or {}).get(split) or {}).get("torque") or {})
                     .get("rmse_median_over_bags"),
                     "consistent": (plaus.get("physical") or {}).get("consistent"),
                     "friction_odd_max": _max_abs((best.get(split, {}).get("friction_signature") or {})
                                                  .get("odd_component")),
                     "wall_s": metrics.get("summary", {}).get("wall_s")})
    rows.sort(key=lambda r: (r["rmse"] if r["rmse"] is not None else math.inf))
    lines = [f"| rank | run | method | seed | {split} RMSE | strict | per-joint {split} RMSE | hold slopes (joint: slope, per run) | "
             "max |prior dist.| σ | calibrated | prior-det. | at bound | max odd friction | val loss | evals to 1 % | evals | wall s |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for rank, r in enumerate(rows, start=1):
        slopes = "; ".join(f"{run}: " + ", ".join(f"A{int(j) + 1} {v:.3f}" for j, v in s.items())
                           for run, s in r["slopes"].items() if s)
        per_joint = " ".join(f"{v:.2f}" for v in (r["per_joint_rmse"] or []))
        lines.append(f"| {rank} | `{r['run']}` | {r['method']} | {r['seed']} | {_fmt(r['rmse'])} | "
                     f"{_fmt(r['strict_rmse'])} | {per_joint} | "
                     f"{slopes} | {_fmt(r['max_prior_distance_sigma'], '.2f')} | {r['n_calibrated']} | "
                     f"{r['n_prior_determined']} | "
                     f"{len(r['at_bound'])} | "
                     f"{_fmt(r['friction_odd_max'])} | {_fmt(r['val_loss'], '.4g')} | {r['evals_to_1pct']} | "
                     f"{r['evaluations']} | {_fmt(r['wall_s'], '.0f')} |")
    methods: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        methods.setdefault(str(r["method"]), []).append(r)
    lines += ["", f"Per method ({split} RMSE over seeds; evaluations and wall time summed):", "",
              "| method | runs | mean | min | max | evaluations | wall s |", "|---|---|---|---|---|---|---|"]
    for method, group in methods.items():
        values = [g["rmse"] for g in group if g["rmse"] is not None]
        lines.append(f"| {method} | {len(group)} | {np.mean(values):.3f} | {min(values):.3f} | {max(values):.3f} | "
                     f"{sum(g['evaluations'] or 0 for g in group)} | {sum(g['wall_s'] or 0 for g in group):.0f} |")
    return "\n".join(lines) + "\n", rows


# ---------------------------------------------------------------------------
# lambda sensitivity (S2R_06 F-1)
# ---------------------------------------------------------------------------

#: A conclusion "changes across the decade" when ...
SENSITIVITY_RMSE_REL = 0.10      # the test RMSE moves by more than 10 %,
SLOPE_TARGET = 0.02              # a hold slope crosses the 1.00 +- 0.02 target,
SENSITIVITY_SD = 2.0             # a parameter calibrated in both moves > 2 posterior sd (of the stage run),
#                                  or the calibrated set changes.


def sensitivity(main: str | Path, others: Mapping[float, str | Path]) -> tuple[str, dict[str, Any]]:
    """The stage run against the same stage at other ``prior_weight_factor``
    values: test RMSE (zeroed and strict), hold slopes, calibrated set and
    values, with every conclusion that changes flagged."""
    main = Path(main)
    folders = {_factor(main): main, **{float(k): Path(v) for k, v in others.items()}}
    data = {}
    for factor, folder in folders.items():
        metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
        ident = json.loads((folder / "identifiability.json").read_text(encoding="utf-8"))
        registry = _registry(folder)
        not_calibrated = metrics.get("not_calibrated", {})
        theta = metrics["best_theta"]
        data[factor] = {
            "folder": str(folder), "test": ((metrics["best"].get("test") or {}).get("torque") or {}),
            "strict": (((metrics.get("best_strict") or {}).get("test") or {}).get("torque") or {}),
            "slopes": {run: b.get("slope") for run, b in ((metrics["best"].get("holds") or {}).get("hold_slopes")
                                                          or {}).items() if isinstance(b, Mapping)},
            "calibrated": sorted(n for n in theta if n not in not_calibrated),
            "distance": {n: registry.specs[n].prior_distance(theta[n]) for n in theta if n in registry.specs},
            "sd": {r["name"]: r["posterior_sd_sigma"] for r in ident["parameters"] if r["posterior_sd_sigma"]},
            "at_bound": [h["name"] for h in bound_hits(folder, theta) if h["name"] not in not_calibrated]}
    reference = data[_factor(main)]
    flags = []
    for factor, block in data.items():
        if block is reference:
            continue
        a, b = reference["test"].get("rmse_median_over_bags"), block["test"].get("rmse_median_over_bags")
        if a and b and abs(b - a) > SENSITIVITY_RMSE_REL * a:
            flags.append(f"x{factor:g}: test RMSE {a:.3f} -> {b:.3f} (> {100 * SENSITIVITY_RMSE_REL:g} %)")
        for run, slopes in reference["slopes"].items():
            for j, value in (slopes or {}).items():
                other = ((block["slopes"].get(run) or {}).get(j))
                if other is not None and (abs(value - 1.0) <= SLOPE_TARGET) != (abs(other - 1.0) <= SLOPE_TARGET):
                    flags.append(f"x{factor:g}: hold slope {run} A{int(j) + 1} {value:.3f} -> {other:.3f} crosses "
                                 f"1.00 +- {SLOPE_TARGET}")
        gained = sorted(set(block["calibrated"]) - set(reference["calibrated"]))
        lost = sorted(set(reference["calibrated"]) - set(block["calibrated"]))
        if gained or lost:
            flags.append(f"x{factor:g}: calibrated set changes (+{len(gained)}: {', '.join(gained) or '–'}; "
                         f"-{len(lost)}: {', '.join(lost) or '–'})")
        for name in sorted(set(block["calibrated"]) & set(reference["calibrated"])):
            sd = reference["sd"].get(name)
            da, db = reference["distance"].get(name), block["distance"].get(name)
            if sd and da is not None and db is not None and abs(db - da) > SENSITIVITY_SD * sd:
                flags.append(f"x{factor:g}: `{name}` moves {db - da:+.3f} sigma (> {SENSITIVITY_SD:g} posterior sd "
                             f"{sd:.3f})")
    factors = sorted(data)
    lines = ["## lambda sensitivity (S2R_06 F-1)", "",
             "The same stage at other `prior_weight_factor` values (prior_from unchanged); flagged: a test RMSE change "
             f"> {100 * SENSITIVITY_RMSE_REL:g} %, a hold slope crossing 1.00 +- {SLOPE_TARGET}, a change of the "
             f"calibrated set, a parameter calibrated in both moving > {SENSITIVITY_SD:g} posterior sd.", "",
             "| | " + " | ".join(f"x{f:g}" + (" (this run)" if data[f] is reference else "") for f in factors) + " |",
             "|---|" + "---|" * len(factors),
             "| test RMSE (zeroed) | " + " | ".join(_fmt(data[f]["test"].get("rmse_median_over_bags")) for f in factors)
             + " |",
             "| test RMSE (strict) | " + " | ".join(_fmt(data[f]["strict"].get("rmse_median_over_bags"))
                                                   for f in factors) + " |",
             "| calibrated parameters | " + " | ".join(str(len(data[f]["calibrated"])) for f in factors) + " |",
             "| calibrated at a bound | " + " | ".join(str(len(data[f]["at_bound"])) for f in factors) + " |"]
    for run, slopes in reference["slopes"].items():
        for j in (slopes or {}):
            lines.append(f"| hold slope {run} A{int(j) + 1} | " + " | ".join(
                _fmt((data[f]["slopes"].get(run) or {}).get(j), ".4f") for f in factors) + " |")
    lines += ["", f"**Flags ({len(flags)}):**" if flags else "**No conclusion changes across the decade.**", ""]
    lines += [f"* {f}" for f in flags]
    return "\n".join(lines) + "\n", {"factors": factors, "runs": {str(f): data[f]["folder"] for f in factors},
                                     "flags": flags}


def _factor(folder: Path) -> float:
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    return float((config.get("loss") or {}).get("prior_weight_factor", 1.0))


def _registry(folder: Path):
    from .parameters import Registry

    return Registry.from_yaml((folder / "registry.yaml").read_text(encoding="utf-8"))


def write_candidate(folder: str | Path) -> Path:
    """S2R_06 P4a-5: the run's ``best.yaml`` as a **candidate** overlay,
    ``<run>/candidate_calibrated.yaml`` (under data/sim2real: never the
    published ``config/sim2real/calibrated/``; publishing waits for the
    architect's review, A-3). The stage's full set, ``calibrated`` per entry."""
    from datetime import datetime, timezone

    from .parameters import check_write_target

    folder = Path(folder).resolve()
    target = folder / "candidate_calibrated.yaml"
    check_write_target(target)
    overlay = yaml.safe_load((folder / "best.yaml").read_text(encoding="utf-8"))
    inputs = json.loads((folder / "inputs.json").read_text(encoding="utf-8"))
    metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    parameters = overlay.pop("parameters")
    header = {**overlay, "kind": "candidate", "status": "candidate: publish only after the architect's review "
              "(S2R_06 P4a-5, A-3)", "source_run": str(folder), "code_digest": (inputs.get("code") or {}).get("digest"),
              "runs": {k: v.get("dataset_sha256") for k, v in (inputs.get("runs") or {}).items()},
              "test_rmse": ((metrics["best"].get("test") or {}).get("torque") or {}).get("rmse_median_over_bags"),
              "n_calibrated": sum(1 for e in parameters.values() if e.get("calibrated", True)),
              "n_parameters": len(parameters), "written": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if overlay.get("identifiability_mode") == "label":
        header["status"] += "; S2R_08 A-12' labels"
    tmp = target.with_suffix(".yaml.tmp")
    tmp.write_text(yaml.safe_dump({**header, "parameters": parameters}, sort_keys=False), encoding="utf-8")
    tmp.replace(target)
    return target


def _max_abs(values: Sequence[float | None] | None) -> float | None:
    clean = [abs(v) for v in (values or []) if v is not None and math.isfinite(v)]
    return max(clean) if clean else None


__all__ = ["sensitivity", "write_candidate", "AT_BOUND_Z", "bound_hits", "code_digest", "compare_runs", "overlay_annotations", "read_history", "render_report", "run_inputs",
           "write_figures", "write_json"]
