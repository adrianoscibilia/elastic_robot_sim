"""One calibration run, start to finish (S2R_01 T-4.5; R0 in P-3, S2R_06 P-4a).

:func:`calibrate` resolves the stage config, builds the registry (with the
previous stage's overlay as prior, ``prior_from``; offsets of a run with no
training bag excluded, A-13), the problem and the optimizer, and then:

1. **fit** within the budget, selecting on ``selection`` (``val``: lowest
   val data loss; ``end_point``: the optimizer's own end point, F-6);
2. **the calibrated label** (``identifiability.mode``):

   * ``label`` (S2R_08 A-12', the default): d_post **once, on the full free
     set**, at the selected point; ``calibrated: true`` iff d_post >=
     ``d_post_min``, else ``false`` with the reason and the **partners**
     (|posterior correlation| >= ``partner_min``). No refit, no value
     changes;
   * ``freeze_refit`` (S2R_06 A-12, the owner's overrule): the free parameter
     with the lowest d_post is frozen at its freeze value (its prior mean:
     the previous stage's value when ``prior_from`` gave one, else the
     nominal), d_post is recomputed, until every free parameter has
     ``d_post >= d_post_min``; then a refit on that set, from the fit's
     values (``history_refit<k>.jsonl``). If the refit's point leaves a
     parameter below the threshold, the round repeats (``max_rounds``);
3. **session zero** (A-13, ``split.zero``): only the zero runs' sensor
   offsets are fitted (least squares, no prior) on their holds, everything
   else frozen at the stage result; theta0 gets the same zero, so the
   baseline column is "nominal plant, zeroed sensor";
4. scores theta0, the result and (A-13's strict column) the result with the
   test run's offsets = the training run's; writes the run folder described
   in :mod:`.report`. ``best.yaml`` carries the stage's full parameter set,
   each entry marked ``calibrated: true | false``.

The working overlay (``working.yaml``, S2R_02 A-3) is rewritten at each
improvement while the optimizer runs. Nothing is published: that is a
separate step after the architect's review.

:func:`report_only` rebuilds ``report.md`` (and the figures) from a run
folder alone.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from . import identifiability as ida
from .config import Sim2RealConfig
from .metrics import metrics, plausibility
from .optimizers import LeastSquaresOptimizer, make_optimizer
from .parameters import Registry, check_write_target, unseen_offsets
from .plant import PlantBase
from .problem import CalibrationProblem, build_engines, load_runs
from .real_runs import RealRun
from .report import (compare_runs, overlay_annotations, read_history, render_report, run_inputs, write_figures,
                     write_json)
from .result import read_replay, write_replay

SPLITS = ("train", "val", "test", "zero")
#: S2R_08 A-12': the overlay header of a label-mode run.
LABEL_NOTE = ("not-calibrated values are nominal plus the minimum-norm share of a data-determined combination; "
              "change them only together with their partners")


class Interrupted(RuntimeError):
    """Raised by ``stop_after`` (tests): the run stops as if killed."""


def analysed_names(registry: Registry, base: PlantBase) -> list[str]:
    """The free parameters plus, for the record, the frozen stiffnesses
    (R0 sees K only through the static deflection: T-4.3 shows it)."""
    extra = [f"transmission.{j}.stiffness" for j in base.joint_names
             if f"transmission.{j}.stiffness" in registry.specs and registry.specs[f"transmission.{j}.stiffness"].sigma]
    return list(registry.free_names) + [n for n in extra if n not in registry.free_names]


def identifiability_at(problem: CalibrationProblem, theta: Mapping[str, float], names: list[str], *,
                       partner_min: float = ida.PARTNER_MIN) -> dict[str, Any]:
    jacobian = problem.data_jacobian_u(theta, names)
    return ida.analyse(jacobian, names, prior_weight=problem.loss.prior_weight, free=problem.registry.free_names,
                       partner_min=partner_min)


def evaluate_theta(problem: CalibrationProblem, theta: Mapping[str, float], contracts: Mapping[str, Any], *,
                   plaus: Mapping[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, list]]:
    """The A-2 suite per split, plus the hold slopes over every hold."""
    results = {split: problem.replay(theta, split) for split in SPLITS if split in problem.engines}
    out: dict[str, Any] = {}
    for split, rows in results.items():
        out[split] = metrics(rows, split, contracts=contracts)
    everything = [r for rows in results.values() for r in rows]
    out["holds"] = {"hold_slopes": metrics(everything, None).get("hold_slopes")}
    if plaus is not None:
        out["plausibility"] = dict(plaus)
    return out, results


def calibrate(config: Sim2RealConfig, *, method: str | None = None, seed: int | None = None,
              budget: int | None = None, run_id: str | None = None, resume: bool = False,
              runs: Mapping[str, RealRun] | None = None, base: PlantBase | None = None, figures: bool = True,
              stop_after: int | None = None, verbose: bool = False, command: str | None = None,
              mode: str | None = None, reduced_from: str | Path | None = None) -> Path:
    """``mode`` overrides ``identifiability.mode`` (``label`` |
    ``freeze_refit``); ``reduced_from`` (a run folder or its ``best.yaml``)
    fixes the free set to that run's calibrated parameters, the rest frozen
    at its fitted values (S2R_06/S2R_08 P4a-4; freeze-then-refit is then
    off, the label still applies)."""
    settings = dataclasses.replace(config.optimizer, method=method or config.optimizer.method,
                                   seed=config.optimizer.seed if seed is None else int(seed),
                                   budget=config.optimizer.budget if budget is None else int(budget))
    stage = config.source.stem
    run_id = run_id or stage
    folder = (config.output / run_id).resolve()
    check_write_target(folder / "history.jsonl")
    if (folder / "history.jsonl").is_file() and not resume:
        raise FileExistsError(f"{folder} already has a history: pass resume=True (--resume) or another run id")
    base = PlantBase.from_config(config) if base is None else base
    registry = Registry.from_config(config, base)
    excluded = unseen_offsets(Registry.from_config(dataclasses.replace(config, split=None), base), config)
    reduced = None
    if reduced_from is not None:
        registry, reduced = _reduced_registry(registry, Path(reduced_from))
    mode = config.identifiability.mode if mode is None else mode
    if mode not in ("label", "freeze_refit"):
        raise ValueError(f"identifiability mode {mode!r}: label or freeze_refit")
    a12 = mode == "freeze_refit" and reduced is None
    d_post_min = config.identifiability.d_post_min
    partner_min = config.identifiability.partner_min
    runs = load_runs(config) if runs is None else runs
    contracts = {run_id_: run.contract for run_id_, run in runs.items()}
    folder.mkdir(parents=True, exist_ok=True)

    resolved = dict(config.resolved)
    resolved["optimizer"] = {**dict(resolved.get("optimizer") or {}), "method": settings.method,
                             "seed": settings.seed, "budget": settings.budget}
    resolved["identifiability"] = {**{k: v for k, v in dict(resolved.get("identifiability") or {}).items()
                                      if k != "freeze_then_refit"},
                                   "mode": mode, "d_post_min": d_post_min, "partner_min": partner_min}
    if reduced is not None:
        resolved["reduced_from"] = reduced
    (folder / "config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    write_json(folder / "inputs.json", run_inputs(config, runs, stage=stage, method=settings.method,
                                                  seed=settings.seed, budget=settings.budget,
                                                  prior_from=config.prior_from, command=command or " ".join(sys.argv),
                                                  reduced_from=(reduced or {}).get("overlay")))
    (folder / "registry.yaml").write_text(registry.to_yaml(), encoding="utf-8")

    # The overlay's parameter set: the stage's free set plus the excluded
    # (A-13) offsets, so best.yaml is a complete robot for these runs.
    overlay_registry = registry.unfreeze(excluded + [n for n in (reduced or {}).get("frozen", [])])
    working = folder / "working.yaml"
    not_calibrated: dict[str, str] = {n: "A-13: another session's sensor offset (no session zero applied)"
                                      for n in excluded}
    not_calibrated.update({n: "fixed by the reduced set (not calibrated in the source run)"
                           for n in (reduced or {}).get("frozen", [])})
    header_base = dict(stage=stage, run_id=run_id, selected_on=config.selection, identifiability_mode=mode)
    if mode == "label":
        header_base["note"] = LABEL_NOTE

    def on_improvement(evaluation, theta):
        overlay_registry.write_overlay(working, theta, evaluation=evaluation.index, val_loss=evaluation.val_loss,
                                       train_loss=evaluation.loss, **header_base)

    problem = CalibrationProblem.from_config(config, registry, base, runs, history_path=folder / "history.jsonl",
                                             resume=resume, on_improvement=on_improvement)
    theta0 = registry.values()
    names = analysed_names(registry, base)
    ident0 = identifiability_at(problem, theta0, names, partner_min=partner_min)
    write_json(folder / "identifiability_theta0.json", ident0)
    (folder / "identifiability_theta0.md").write_text(ida.to_markdown(ident0, title="Identifiability at theta0"),
                                                      encoding="utf-8")

    calls = 0

    def run_optimizer(current: CalibrationProblem) -> tuple[Any, float]:
        optimizer = make_optimizer(settings)
        if isinstance(optimizer, LeastSquaresOptimizer):
            optimizer.jacobian = current.jacobian
            function = current.residuals
        else:
            function = current.objective

        def guarded(z):
            nonlocal calls
            calls += 1
            if stop_after is not None and calls > stop_after:
                raise Interrupted(f"stopped after {stop_after} evaluations")
            return function(z)

        start = time.perf_counter()
        if current.registry.n_free:
            optimizer.minimize(guarded, [(-1.0, 1.0)] * current.registry.n_free, current.registry.z0(),
                               max_evals=settings.budget, verbose=verbose)
        else:
            current.evaluate(np.zeros(0))
        return optimizer, time.perf_counter() - start

    least_squares = settings.method == "least_squares"
    optimizer, wall = run_optimizer(problem)
    selected = _select(problem, config.selection, least_squares)
    phases = [_phase_record(problem, selected, "fit", "history.jsonl", wall)]
    a12_frozen: list[dict[str, Any]] = []
    freeze_value = {n: registry.specs[n].mean for n in registry.free_names}
    if a12:
        for round_ in range(1, config.identifiability.max_rounds + 1):
            theta_sel = problem.full_theta(selected.theta)
            frozen, theta_frozen = ida.freeze_plan(
                lambda t, n, p=problem: p.data_jacobian_u(t, n), theta_sel, problem.registry.free_names,
                freeze_value, prior_weight=problem.loss.prior_weight, d_post_min=d_post_min)
            if not frozen:
                break
            for entry in frozen:
                entry["round"] = round_
            a12_frozen += frozen
            keep = [n for n in problem.registry.free_names if n not in {e["name"] for e in frozen}]
            refit_registry = problem.registry.freeze({e["name"]: e["frozen_at"] for e in frozen}) \
                .with_values({n: theta_sel[n] for n in keep})
            history = f"history_refit{round_}.jsonl"
            problem = CalibrationProblem(refit_registry, base, problem.engines, problem.loss,
                                         history_path=folder / history, resume=resume,
                                         on_improvement=on_improvement, runs=problem.runs, jobs=problem.jobs)
            optimizer, wall = run_optimizer(problem)
            selected = _select(problem, config.selection, least_squares)
            phases.append(_phase_record(problem, selected, f"refit {round_}", history, wall))
    not_calibrated.update({e["name"]: f"A-12: not seen by these data (d_post {e['d_post']:.3f} < {d_post_min:g} "
                                      f"when frozen, round {e['round']})" for e in a12_frozen})
    if selected is None:
        raise RuntimeError("no successful evaluation")

    theta = problem.full_theta(selected.theta)
    final_free = list(problem.registry.free_names)
    names = final_free + [e["name"] for e in a12_frozen] + \
        [n for n in analysed_names(registry, base) if n not in registry.free_names]
    ident = ida.analyse(problem.data_jacobian_u(theta, names), names, prior_weight=problem.loss.prior_weight,
                        free=final_free, partner_min=partner_min)
    ident["a12"] = {"mode": mode, "enabled": a12, "d_post_min": d_post_min, "frozen": a12_frozen, "free": final_free,
                    "below_threshold_after_refit": [r["name"] for r in ident["parameters"]
                                                    if r["free"] and r["data_share_posterior"] < d_post_min]}
    partners: dict[str, list] = {}
    if mode == "label":
        # S2R_08 A-12': one d_post on the full free set labels; nothing moves.
        labelled = ida.labels(ident, d_post_min=d_post_min)
        unseen = {n: e for n, e in labelled.items() if not e["calibrated"]}
        not_calibrated.update({n: e["reason"] for n, e in unseen.items()})
        partners = {n: e["partners"] for n, e in unseen.items()}
        ident["a12"].update({"partner_min": partner_min, "below_threshold_after_refit": [],
                             "labelled": {n: {"d_post": e["d_post"], "partners": e["partners"]}
                                          for n, e in unseen.items()},
                             "n_calibrated": sum(1 for e in labelled.values() if e["calibrated"])})
    write_json(folder / "identifiability.json", ident)
    (folder / "identifiability.md").write_text(ida.to_markdown(ident, title="Identifiability at the result")
                                               + _a12_markdown(ident["a12"]), encoding="utf-8")

    # A-13: the session zero, for the result and for theta0.
    zero = problem.engines.get("zero")
    zero_record: dict[str, Any] = {}
    if zero is not None:
        zero_names = [n for n in overlay_registry.specs if ".offset@" in n
                      and n.rsplit("@", 1)[1] in zero.run_ids and n in overlay_registry.free_names]
        theta, zero_record = session_zero(problem, zero, theta, zero_names)
        theta0, zero_record["theta0"] = session_zero(problem, zero, theta0, zero_names)
        for n in zero_names:
            not_calibrated.pop(n, None)
    share = {r["name"]: r["data_share"] for r in ident["parameters"]}
    share_post = {r["name"]: r["data_share_posterior"] for r in ident["parameters"]}
    plant = problem.plant(theta)
    plaus = plausibility(overlay_registry, theta, plant, data_share=share, data_share_posterior=share_post,
                         not_calibrated=not_calibrated, partners=partners)
    metrics0, _ = evaluate_theta(problem, theta0, contracts,
                                 plaus=plausibility(overlay_registry, theta0, problem.plant(theta0)))
    metrics1, results = evaluate_theta(problem, theta, contracts, plaus=plaus)
    extra: dict[str, Any] = {}
    strict = _strict_theta(theta, config, problem, zero)
    if strict is not None and "test" in problem.engines:
        extra["best_strict"] = {"test": metrics(problem.replay(strict[0], "test"), "test", contracts=contracts),
                                "offsets_from": strict[1]}
    end_point = problem.best_train("squared" if least_squares else "loss")
    if end_point is not None and end_point.index != selected.index:
        # The other selection rule's choice next to this one, so the report
        # shows what selection cost or bought.
        theta_c = problem.full_theta(end_point.theta)
        if zero is not None:
            theta_c, _ = session_zero(problem, zero, theta_c, zero_record["names"])
        extra["converged"], _ = evaluate_theta(problem, theta_c, contracts,
                                               plaus=plausibility(overlay_registry, theta_c, problem.plant(theta_c)))
        extra["converged"]["index"] = end_point.index
    elif config.selection == "end_point" and problem.best is not None and problem.best.index != selected.index:
        theta_v = problem.full_theta(problem.best.theta)
        if zero is not None:
            theta_v, _ = session_zero(problem, zero, theta_v, zero_record["names"])
        extra["val_selected"], _ = evaluate_theta(problem, theta_v, contracts)
        extra["val_selected"]["index"] = problem.best.index
    if config.replay.r0.rows != "all":
        # A row-filtered stage (S0) is also scored on every row: FR_05-comparable.
        full = dataclasses.replace(config, replay=dataclasses.replace(
            config.replay, r0=dataclasses.replace(config.replay.r0, rows="all")))
        engines = build_engines(full, base, runs)
        for label, values in (("theta0_all_rows", theta0), ("best_all_rows", theta)):
            plant_ = problem.plant(values)
            extra[label] = {split: metrics(engine.replay(plant_), split, contracts=contracts)
                            for split, engine in engines.items()}
    best_theta = {n: float(theta[n]) for n in overlay_registry.free_names}
    summary = {"evaluations": sum(p["evaluations"] for p in phases), "calls_this_session": calls,
               "best_index": selected.index, "best_phase": phases[-1]["phase"], "best_train_loss": selected.loss,
               "best_val_loss": selected.val_loss, "wall_s": sum(p["wall_s"] for p in phases),
               "method": settings.method, "seed": settings.seed, "budget": settings.budget,
               "selection": config.selection, "phases": phases,
               "n_free_stage": registry.n_free, "n_free_final": len(final_free), "n_frozen_a12": len(a12_frozen),
               "identifiability_mode": mode, "n_not_calibrated_label": len(partners) if mode == "label" else None,
               "jacobian_evaluations": getattr(optimizer, "jacobian_evaluations", None)}
    write_json(folder / "metrics.json", {"joint_names": list(base.joint_names), "summary": summary,
                                         "loss": problem.loss.describe(), "theta0": metrics0, "best": metrics1,
                                         "best_theta": best_theta, "not_calibrated": not_calibrated,
                                         "session_zero": zero_record, **extra})
    for split, rows in results.items():
        write_replay(rows, folder / f"replay_{split}.parquet")
    annotations = overlay_annotations(ident, overlay_registry.free_names, not_calibrated,
                                      zeroed=zero_record.get("names", ()), partners=partners)
    header = dict(evaluation=selected.index, val_loss=selected.val_loss, train_loss=selected.loss, **header_base)
    overlay_registry.write_overlay(folder / "best.yaml", best_theta, annotations=annotations, **header)
    overlay_registry.write_overlay(working, best_theta, annotations=annotations, **header)
    if figures:
        write_figures(folder, results, read_history(folder / "history.jsonl"), base.joint_names)
    (folder / "report.md").write_text(render_report(folder), encoding="utf-8")
    return folder


def _select(problem: CalibrationProblem, selection: str, least_squares: bool):
    """F-6: ``val`` = lowest val data loss; ``end_point`` = the optimizer's own
    end point (lowest 0.5|r|^2 for least squares, else the lowest loss)."""
    if selection == "end_point":
        return problem.best_train("squared" if least_squares else "loss")
    return problem.best


def _phase_record(problem: CalibrationProblem, selected, phase: str, history: str, wall: float) -> dict[str, Any]:
    return {"phase": phase, "history": history, "n_free": problem.registry.n_free,
            "evaluations": len(problem.evaluations), "selected_index": None if selected is None else selected.index,
            "train_loss": None if selected is None else selected.loss,
            "val_loss": None if selected is None else selected.val_loss, "wall_s": wall}


def _reduced_registry(registry: Registry, source: Path) -> tuple[Registry, dict[str, Any]]:
    overlay = source / "best.yaml" if source.is_dir() else source
    entries = (yaml.safe_load(overlay.read_text(encoding="utf-8")) or {}).get("parameters") or {}
    keep = [n for n in registry.free_names if n in entries and entries[n].get("calibrated", True)]
    frozen = {n: float(entries[n]["value"]) for n in registry.free_names if n not in keep and n in entries}
    missing = [n for n in registry.free_names if n not in entries]
    if missing:
        raise ValueError(f"reduced set from {overlay}: free parameters missing there: {', '.join(missing)}")
    return registry.freeze(frozen), {"overlay": str(overlay), "free": keep, "frozen": list(frozen)}


def session_zero(problem: CalibrationProblem, engine: Any, theta: Mapping[str, float], names: Sequence[str]
                 ) -> tuple[dict[str, float], dict[str, Any]]:
    """S2R_06 A-13: least squares on ``names`` (the zero runs' offsets) over
    the zero bags (holds), every other parameter at ``theta``; no prior (a
    sensor zero is a measurement). Rows weighted as the loss weighs them."""
    from scipy.optimize import least_squares as scipy_least_squares

    theta = dict(theta)
    names = list(names)
    if not names:
        return theta, {"names": []}
    specs = [problem.registry.specs[n] for n in names]
    lower = np.array([s.lower for s in specs])
    upper = np.array([s.upper for s in specs])

    def plant_at(x):
        values = dict(theta)
        values.update({n: float(v) for n, v in zip(names, x)})
        return values, problem.plant(values)

    def residual(x):
        _, plant = plant_at(x)
        results = engine.replay(plant)
        return problem.loss.residual_vector(results)

    def jacobian(x):
        values, plant = plant_at(x)
        weights = problem.loss.row_weights(engine.results(plant, *engine.simulate(plant)))
        return weights[:, None] * engine.jacobian(plant, names, theta=values)

    x0 = np.clip([float(theta[n]) for n in names], lower, upper)
    before = engine.replay(problem.plant(theta))
    fit = scipy_least_squares(residual, x0, jac=jacobian, bounds=(lower, upper), method="trf", x_scale="jac",
                              xtol=1e-12, ftol=1e-12, gtol=1e-12)
    values, plant = plant_at(fit.x)
    after = engine.replay(plant)

    def rms(results):
        return np.sqrt(np.mean(np.vstack([r.residual for r in results]) ** 2, axis=0)).tolist()

    return values, {"names": names, "before": {n: float(theta[n]) for n in names},
                    "after": {n: float(v) for n, v in zip(names, fit.x)},
                    "at_bound": [n for n, v, lo, hi in zip(names, fit.x, lower, upper) if v <= lo or v >= hi],
                    "hold_rms_before": rms(before), "hold_rms_after": rms(after), "nfev": int(fit.nfev),
                    "bags": [f"{b.run_id}:{b.bag}" for b in engine.bags]}


def _strict_theta(theta: Mapping[str, float], config: Sim2RealConfig, problem: CalibrationProblem,
                  zero: Any) -> tuple[dict[str, float], str] | None:
    """A-13's strict column: the test run's offsets = the (single) training
    run's, session drift counted as error."""
    trained = config.split.runs_of("train", problem.runs)
    tested = config.split.runs_of("test", problem.runs)
    if len(trained) != 1 or not tested:
        return None
    strict = dict(theta)
    for name in theta:
        if ".offset@" in name:
            joint_name, run = name.rsplit("@", 1)
            if run in tested and run not in trained:
                strict[name] = float(theta[f"{joint_name}@{trained[0]}"])
    return strict, trained[0]


def _a12_markdown(block: Mapping[str, Any]) -> str:
    if block.get("mode") == "label":
        labelled = block.get("labelled") or {}
        lines = ["", f"## Calibrated label (S2R_08 A-12'): {block.get('n_calibrated')} calibrated, {len(labelled)} "
                 f"not separated by these data", "",
                 f"d_post once, on the full free set (threshold {block['d_post_min']:g}); no refit, no value changes. "
                 f"Partners: free parameters with |posterior correlation| >= {block['partner_min']:g}. "
                 f"{LABEL_NOTE[0].upper() + LABEL_NOTE[1:]}.", "",
                 "| parameter | d_post | partners (posterior corr) |", "|---|---|---|"]
        lines += [f"| `{n}` | {e['d_post']:.3f} | {ida.format_partners(e['partners'], limit=8)} |"
                  for n, e in sorted(labelled.items(), key=lambda kv: kv[1]["d_post"])]
        return "\n".join(lines) + "\n"
    if not block.get("enabled"):
        return "\nFreeze-then-refit (S2R_06 A-12): off for this run.\n"
    lines = ["", f"## Freeze-then-refit (S2R_06 A-12): {len(block['frozen'])} frozen, {len(block['free'])} calibrated",
             "", f"Frozen in order of d_post (threshold {block['d_post_min']:g}), each at its freeze value: "
             "**not calibrated: not seen by these data**.", "",
             "| round | parameter | d_post when frozen | fitted value | frozen at |", "|---|---|---|---|---|"]
    lines += [f"| {e['round']} | `{e['name']}` | {e['d_post']:.3f} | {e['fitted']:.6g} | {e['frozen_at']:.6g} |"
              for e in block["frozen"]]
    if block["below_threshold_after_refit"]:
        lines += ["", "**Below the threshold after the last refit:** "
                  + ", ".join(f"`{n}`" for n in block["below_threshold_after_refit"])]
    return "\n".join(lines) + "\n"


def report_only(folder: str | Path, *, figures: bool = True) -> Path:
    """Rebuild report.md (and figures) from the run folder's files."""
    folder = Path(folder).resolve()
    check_write_target(folder / "report.md")
    if figures:
        metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
        results = {split: read_replay(folder / f"replay_{split}.parquet") for split in SPLITS
                   if (folder / f"replay_{split}.parquet").is_file()}
        write_figures(folder, results, read_history(folder / "history.jsonl"), metrics["joint_names"])
    (folder / "report.md").write_text(render_report(folder), encoding="utf-8")
    return folder / "report.md"


__all__ = ["Interrupted", "analysed_names", "calibrate", "compare_runs", "evaluate_theta", "identifiability_at",
           "report_only", "session_zero"]
