#!/usr/bin/env python3
"""Small, testable pieces of ``scripts/round5_weekend.sh`` (`R5_10` T-8).

    launch_helpers.py stem CONFIG                  # dataset file stem the build writes
    launch_helpers.py probe CONFIG TAG [ACCEPTED]  # PASS/FAIL line; exit 1 on FAIL
    launch_helpers.py bags DIR                     # bags a data stage built (manifests)
    launch_helpers.py rows DATASET [SPLIT]         # rows of one split of a written dataset
    launch_helpers.py timeouts PREFLIGHT_RUN PLAN_JSON > timeouts.tsv
    launch_helpers.py fullbags CONFIG [ABLATIONS]   # round 6: bags a full build of CONFIG makes
    launch_helpers.py bounds CONFIG TAG             # round 6: Sec 2.2 / Sec 3 bounds, PASS/FAIL
    launch_helpers.py timeouts6 PREFLIGHT_RUN PLAN_JSON > timeouts.tsv   # round 6 (scripts/round6_run.sh)

FR_03 retrain (``scripts/round6_retrain.sh``):

    launch_helpers.py retrain_chain SRC_YAML OUT_YAML DATASET EPOCHS PATIENCE SEED...
    launch_helpers.py merge_summaries OUT_CSV IN_CSV...
    launch_helpers.py retrain_plan PREFLIGHT_RUN PLAN_JSON [PARALLEL] > timeouts.tsv  # + projection.json
    launch_helpers.py gpu_fit PREFLIGHT_RUN PARALLEL         # PASS/FAIL line; exit 1 on FAIL

Timeouts come from the preflight's measured rates, times three (never a flat
18 h, `R5_09` F-1): a data stage is budgeted per bag, an NN stage per
epoch-row (one training epoch over one row), each with a floor.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(_REPO / "src"))

SAFETY = 3.0
FLOOR_S = 20 * 60


def stem(config_path: str) -> str:
    from elastic_sim.dataset import load_config

    return Path(load_config(config_path).output).stem


def probe(config_path: str, tag: str, accepted: str = "") -> int:
    """FAIL when the probe band cannot survive the consumer's derivative.

    The bound is the Savitzky-Golay floor, ``0.09 x rate``
    (``differentiation.probe_resolvable``).  ``accepted`` lists platforms
    whose unresolvable probe an architect decision accepted (`R5_09` D-2:
    the UR10 at 125 Hz), which are reported but do not fail.
    """
    import warnings

    from elastic_sim.dataset import differentiation_policy, load_config

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        config = load_config(config_path)
    policy = differentiation_policy(config)
    top, rate = policy["probe_top_hz"], policy["sample_rate_hz"]
    line = f"{tag}: probe top {top:g} Hz vs 0.09 x {rate:g} Hz = {0.09 * rate:g} Hz"
    if policy["probe_resolvable"]:
        print(f"PASS {line}")
        return 0
    if tag in {t for t in accepted.replace(",", " ").split() if t}:
        print(f"ACCEPTED {line} -- unresolvable, accepted by R5_09 D-2")
        return 0
    print(f"FAIL {line} -- cap probe_harmonics at {int(0.09 * rate / config.excitation.base_frequency)} "
          "(R5_09 D-2) before building")
    return 1


def bags(directory: str) -> int:
    total = 0
    for manifest in Path(directory).glob("*.manifest.json"):
        data = json.loads(manifest.read_text(encoding="utf-8"))
        total += int(data.get("n_bags", 0)) + len(data.get("unstable_bags") or [])
    return total


def rows(dataset: str, split: str | None = None) -> int:
    import pyarrow.parquet as pq

    if split is None:
        return pq.ParquetFile(dataset).metadata.num_rows
    column = pq.read_table(dataset, columns=["split"]).column("split").to_pylist()
    return sum(1 for value in column if value == split)


def _stage_seconds(run: Path) -> dict[str, float]:
    seconds = {}
    timing = run / "timing.tsv"
    if timing.is_file():
        for line in timing.read_text().splitlines():
            name, value = line.split("\t")[:2]
            seconds[name] = float(value)
    return seconds


def timeouts(preflight_run: str, plan_path: str) -> int:
    """One ``<stage>\\t<seconds>`` line per stage of the planned full run.

    ``plan`` (JSON) holds the preflight's and the full run's sizes::

        {"preflight": {"trials":..,"screen":..,"top":..,"refine":..,"epochs":..,"models":..,"robots":..,"traj":..},
         "full": {...same keys...}, "screen_rows": .., "platforms": {"ur10": {"data_dir": .., "datasets": {label: path}}}}
    """
    run = Path(preflight_run)
    plan = json.loads(Path(plan_path).read_text())
    measured = _stage_seconds(run)
    pre, full = plan["preflight"], plan["full"]
    screen_rows = int(plan.get("screen_rows", 100_000))
    out = []

    # A tiny preflight's NN stages are dominated by fixed start-up cost
    # (interpreter, CUDA, dataset load), so their per-epoch-row rate
    # overestimates; no stage may outlive the run's own deadline.
    cap = int(plan.get("cap_s", 10**9))

    def budget(stage: str, pre_units: float, full_units: float) -> None:
        if stage not in measured or pre_units <= 0:
            return
        rate = measured[stage] / pre_units
        out.append((stage, min(cap, max(FLOOR_S, int(math.ceil(SAFETY * rate * full_units))))))

    for tag, platform in plan["platforms"].items():
        pre_bags = bags(platform["data_dir"])
        # Bags scale with (robots + rigid) x trajectories; modes and extras
        # rows are the same in both runs.
        scale = (full["robots"] + 1) * full["traj"] / max(1, (pre["robots"] + 1) * pre["traj"])
        budget(f"10-data-{tag}", pre_bags, pre_bags * scale)
        for label, dataset in platform["datasets"].items():
            train = rows(dataset, "train")
            # Full-size train rows: same rows per bag, more bags, and the
            # holdout takes a similar share (the split is by robot count).
            full_train = train * scale

            def epoch_rows(size, n_train):
                return size["models"] * (size["trials"] * size["screen"] * min(n_train, screen_rows)
                                         + size["top"] * size["refine"] * n_train)

            budget(f"20-optuna-{label}", epoch_rows(pre, train), epoch_rows(full, full_train))
            budget(f"40-train-{label}", pre["models"] * pre["epochs"] * train,
                   full["models"] * full["epochs"] * full_train)
            test = rows(dataset, "test") or 1
            budget(f"50-eval-{label}", pre["models"] * test, full["models"] * test * scale)
    for stage, seconds in out:
        print(f"{stage}\t{seconds}")
    return 0


def fullbags(config_path: str, ablations: str = "0") -> int:
    """Production + gain-shift (+ both ablations) bags of a full round-6 build."""
    import warnings

    from elastic_sim.dataset import load_config

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        config = load_config(config_path, check_bounds=False)
    trajectories = int(config.n_trajectories)
    production = (int(config.transmission.robots) + (1 if config.rigid_reference else 0)) * trajectories
    gainshift = int(config.split.test_robots) * trajectories
    return production + gainshift + (2 * production if ablations not in ("", "0", "false") else 0)


def bounds(config_path: str, tag: str) -> int:
    """Round 6's doctor line: the config loads (which enforces both bounds) and their values.

    ``load_config`` refuses a schema-3 config whose Sec 2.2 gain bound or Sec 3
    explicit-term bound is violated; this prints what they are, so the doctor
    shows the margin and not only the verdict.
    """
    import warnings

    from elastic_sim.assets import AssetRegistry
    from elastic_sim.dataset import load_config
    from elastic_sim.dataset_bounds import dof_totals, explicit_term_table
    from elastic_sim.dataset_config import differentiation_policy, gain_bound

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            config = load_config(config_path)
    except ValueError as error:
        print(f"FAIL {tag}: {str(error).splitlines()[0]} ...")
        return 1
    asset = AssetRegistry.for_repository(_REPO).load(config.asset)
    worst = max(dof_totals(explicit_term_table(config, asset)).values())
    gain = gain_bound(config)
    policy = differentiation_policy(config)
    ok = gain["ok"] and worst <= 1.0 and policy["probe_resolvable"]
    print(f"{'PASS' if ok else 'FAIL'} {tag}: {config.controller.location} loop, gain bound {gain['value']:.3f} <= 0.5, "
          f"explicit terms max {worst:.2f} <= 1 (h = {config.max_time_step:.3g} / {config.rigid_time_step:.3g} s), "
          f"probe top {policy['probe_top_hz']:g} Hz resolvable {policy['probe_resolvable']}")
    return 0 if ok else 1


def timeouts6(preflight_run: str, plan_path: str) -> int:
    """Round 6's stage timeouts from a preflight that ran the NN stages at two epoch counts.

    ``plan`` (JSON)::

        {"preflight": {"robots", "traj", "trials", "screen", "top", "refine", "models"},
         "epochs": [e_low, e_high], "full": {..., "epochs"}, "screen_rows", "cap_s",
         "variants": {tag: {"data_dir", "full_bags", "dataset"}}}

    A data stage is budgeted per bag (``full_bags`` from the full config).  An
    NN stage ran twice in the preflight, at ``e_low`` and ``e_high`` epochs;
    the difference is the per-epoch-row rate without the fixed start-up cost
    (interpreter, CUDA, dataset load), and the low run's time minus its epochs
    is that fixed cost.  Full = SAFETY x (fixed + rate x full epoch-rows).
    """
    run = Path(preflight_run)
    plan = json.loads(Path(plan_path).read_text())
    measured = _stage_seconds(run)
    pre, full = plan["preflight"], plan["full"]
    low, high = plan["epochs"]
    screen_rows = int(plan.get("screen_rows", 100_000))
    cap = int(plan.get("cap_s", 10**9))
    out = []

    def clamp(seconds: float) -> int:
        return min(cap, max(FLOOR_S, int(math.ceil(SAFETY * seconds))))

    for tag, variant in plan["variants"].items():
        name = f"10-data-{tag}"
        pre_bags = bags(variant["data_dir"])
        if name in measured and pre_bags:
            out.append((name, clamp(measured[name] / pre_bags * float(variant["full_bags"]))))
        scale = float(variant["full_bags"]) / max(pre_bags, 1)
        train = rows(variant["dataset"], "train") if Path(variant["dataset"]).is_file() else 0
        full_train = train * scale

        def optuna_units(size: dict, screen: float, refine: float, n: float) -> float:
            return size["models"] * (size["trials"] * screen * min(n, screen_rows) + size["top"] * refine * n)

        # Epoch-rows of each NN stage: the preflight at e epochs (screen = refine = e), and the full run.
        stages = {
            "20-optuna": (lambda e: optuna_units(pre, e, e, train),
                          optuna_units(full, full["screen"], full["refine"], full_train)),
            "40-train": (lambda e: pre["models"] * e * train, full["models"] * full["epochs"] * full_train),
        }
        for stage, (pre_units, full_units) in stages.items():
            a, b = measured.get(f"{stage}-{tag}-e{low}"), measured.get(f"{stage}-{tag}-e{high}")
            if a is None or b is None or not train:
                continue
            ua, ub = pre_units(low), pre_units(high)
            rate = max((b - a) / max(ub - ua, 1e-9), 0.0)
            fixed = max(a - rate * ua, 0.0)
            out.append((f"{stage}-{tag}", clamp(fixed + rate * full_units)))
        # Export and evaluation: the high-epoch preflight time of each stage, scaled by the data.
        suffix = f"-e{high}"
        for key, seconds in measured.items():
            if (key.startswith(f"30-export-{tag}-") or key.startswith(f"50-eval-{tag}-")) and key.endswith(suffix):
                out.append((key[: -len(suffix)], clamp(seconds * max(scale, 1.0))))
    for stage, seconds in out:
        print(f"{stage}\t{seconds}")
    return 0


# ------------------------------------------------------------ FR_03 retrain ---
#: The residual classes that get an input-matched twin reading tau_cmd (FR_03 T-4).
TAU_TWINS = ("lnn_tau_res", "kalnn_tau_res")


def retrain_chain(src: str, out: str, dataset: str, epochs: str, patience: str, *seeds: str) -> int:
    """``chain-<v>-retrain.yaml`` from a round-6 chain: same hyperparameters, val selection, seeds.

    The four source runs keep their class path, ``training_overrides`` and
    KAN kwargs; each residual run gets a ``<base>_tau`` twin with
    ``model_kwargs.residual_inputs: q_dq_tau`` (FR_03 T-4).
    """
    import yaml

    config = yaml.safe_load(Path(src).read_text(encoding="utf-8"))
    defaults = dict(config["defaults"])
    defaults.update({"dataset_file": str(Path(dataset).resolve()), "epochs": int(epochs),
                     "early_stopping_patience": int(patience), "selection": "val",
                     "seeds": [int(v) for v in seeds], "stop_on_error": False})
    runs = [dict(run) for run in config["runs"]]
    for run in list(runs):
        if run["base_name"] in TAU_TWINS:
            twin = {key: (dict(value) if isinstance(value, dict) else value) for key, value in run.items()}
            twin["base_name"] = f"{run['base_name']}_tau"
            twin["model_kwargs"] = {**(run.get("model_kwargs") or {}), "residual_inputs": "q_dq_tau"}
            twin["_provenance"] = {**(run.get("_provenance") or {}), "parent": run["base_name"],
                                   "note": "parent's hyperparameters reused, no Optuna (FR_03 T-4)"}
            runs.append(twin)
    missing = [f"{b}_tau" for b in TAU_TWINS if f"{b}_tau" not in {r["base_name"] for r in runs}]
    if missing:
        raise SystemExit(f"{src}: no parent run for {missing}")
    header = f"# generated by launch_helpers.py retrain_chain from {src} (FR_03 T-5) -- do not hand-edit\n"
    Path(out).write_text(header + yaml.safe_dump({"defaults": defaults, "runs": runs}, sort_keys=False),
                         encoding="utf-8")
    print(f"{out}: {len(runs)} runs x seeds {list(defaults['seeds'])}, selection val, "
          f"epochs {epochs}, patience {patience}")
    return 0


def merge_summaries(out: str, *inputs: str) -> int:
    """Concatenate per-seed chain summaries (one launcher stage per seed) into one CSV."""
    import pandas as pd

    frames = [pd.read_csv(path) for path in inputs if Path(path).is_file() and Path(path).stat().st_size]
    if not frames:
        raise SystemExit(f"no summaries among {inputs}")
    pd.concat(frames, ignore_index=True).to_csv(out, index=False)
    print(f"{out}: {sum(len(f) for f in frames)} rows from {len(frames)} file(s)")
    return 0


def _summary_rates(path: Path) -> dict[str, dict]:
    import pandas as pd

    if not path.is_file():
        return {}
    frame = pd.read_csv(path)
    return {row.base_name: {"s_epoch": float(row.seconds_per_epoch),
                            "fixed": max(float(row.train_seconds) - float(row.seconds_per_epoch) * float(row.epochs_run), 0.0),
                            "cuda_peak_mb": None if pd.isna(getattr(row, "cuda_peak_mb", None)) else float(row.cuda_peak_mb)}
            for row in frame.itertuples(index=False) if pd.notna(row.seconds_per_epoch)}


def _round6_epoch_rate(run: Path, tag: str) -> float | None:
    """Round 6's measured seconds per training epoch of one variant (40-train time / epochs run)."""
    import pandas as pd

    seconds = _stage_seconds(run).get(f"40-train-{tag}")
    summary = run / f"chain-{tag}.summary.csv"
    if seconds is None or not summary.is_file():
        return None
    return seconds / max(float(pd.read_csv(summary)["epochs_run"].sum()), 1.0)


def retrain_plan(preflight_run: str, plan_path: str, parallel: str = "1") -> int:
    """Stage timeouts for the full retrain from its preflight, and the wall-time projection.

    The preflight trained every class of its variants for a few epochs, once
    alone (``40-train-<v>-s0``) and once as three concurrent seeds
    (``40-train-<v>-p3-s<k>``); each summary row carries the median epoch
    time and the fixed cost.  A variant the preflight did not train takes a
    measured variant's rates scaled by round 6's own per-epoch ratio between
    the two (``reference`` in the plan); its evaluation stages scale with
    its test rows (``datasets`` in the plan).  Full stage = SAFETY x (fixed + rate x
    epochs) per run, summed over the classes; projection.json (beside the
    preflight's ticket) holds the per-(variant, class) rates and the
    projected wall time for PARALLEL_SEEDS = 1 and 3, every run at the epoch cap.
    """
    run = Path(preflight_run)
    plan = json.loads(Path(plan_path).read_text())
    epochs, seeds = int(plan["epochs"]), list(plan["seeds"])
    cap = int(plan.get("cap_s", 10**9))
    measured = _stage_seconds(run)

    def clamp(seconds: float) -> int:
        return min(cap, max(FLOOR_S, int(math.ceil(SAFETY * seconds))))

    rates: dict[str, dict] = {}
    for tag in plan["preflight_variants"]:
        seq = _summary_rates(run / f"chain-{tag}-retrain-s0.summary.csv")
        par = {}
        for k in range(3):
            for name, value in _summary_rates(run / f"chain-{tag}-retrain-p3-s{k}.summary.csv").items():
                par.setdefault(name, []).append(value)
        rates[tag] = {name: {"s_epoch_p1": v["s_epoch"], "fixed_s": v["fixed"], "cuda_peak_mb": v["cuda_peak_mb"],
                             "s_epoch_p3": max((p["s_epoch"] for p in par.get(name, [])), default=None),
                             "source": "preflight"}
                      for name, v in seq.items()}
    for tag, ref in (plan.get("reference") or {}).items():
        if tag in rates or ref.get("from") not in rates:
            continue
        r6 = Path(ref["round6_run"])
        a, b = _round6_epoch_rate(r6, tag), _round6_epoch_rate(Path(ref["round6_from_run"]), ref["from"])
        factor = (a / b) if a and b else float(ref.get("fallback_factor", 2.0))
        rates[tag] = {name: {**v, "s_epoch_p1": v["s_epoch_p1"] * factor,
                             "s_epoch_p3": None if v["s_epoch_p3"] is None else v["s_epoch_p3"] * factor,
                             "source": f"{ref['from']} x {factor:.2f} (round-6 per-epoch ratio)"}
                     for name, v in rates[ref["from"]].items()}

    p = int(parallel)
    out, projection = [], {"epochs": epochs, "seeds": seeds, "rates": rates, "wall_s": {}}
    for which in (1, 3):
        total = 0.0
        for tag in plan["variants"]:
            classes = rates.get(tag) or {}
            key = "s_epoch_p1" if which == 1 else "s_epoch_p3"
            per_seed = sum(v["fixed_s"] + (v[key] or v["s_epoch_p1"] * 3) * epochs for v in classes.values())
            total += per_seed * (len(seeds) if which == 1 else math.ceil(len(seeds) / 3))
            if which == p:
                for seed in seeds:
                    out.append((f"40-train-{tag}-s{seed}", clamp(per_seed)))
        projection["wall_s"][f"parallel_{which}"] = total
    # Relabel, collect, evaluation and report: the preflight's own times, scaled by the data where it grows.
    for tag in plan["variants"]:
        ref = tag if tag in plan["preflight_variants"] else (plan.get("reference") or {}).get(tag, {}).get("from")
        if ref is None:
            continue
        datasets = plan.get("datasets") or {}
        scale = 1.0
        if ref != tag and tag in datasets and ref in datasets:
            scale = rows(datasets[tag], "test") / max(rows(datasets[ref], "test"), 1)
        for stage in ("15-relabel", "35-chain", "45-collect", "50-eval"):
            for key, seconds in measured.items():
                if key.startswith(f"{stage}-{ref}"):
                    out.append((f"{stage}-{tag}{key[len(stage) + 1 + len(ref):]}",
                                clamp(seconds * scale * (len(seeds) if stage == "50-eval" else 1))))
    if "60-report" in measured:
        out.append(("60-report", clamp(measured["60-report"] * 4)))
    projection["wall_s"]["other_stages_upper"] = sum(s for n, s in out if not n.startswith("40-train")) / SAFETY
    (run / "projection.json").write_text(json.dumps(projection, indent=2), encoding="utf-8")
    for stage, seconds in out:
        print(f"{stage}\t{seconds}")
    return 0


GPU_CONTEXT_MB = 700.0      # a CUDA context per process, not in max_memory_allocated
UNMEASURED_SCALE = 1.5      # a variant the preflight did not train (iiwa: 7 dof vs 6, (7/6)^2 = 1.36)


def gpu_fit(preflight_run: str, parallel: str) -> int:
    """Doctor line: PARALLEL concurrent trainings fit in the free GPU memory (FR_03 T-5)."""
    import subprocess

    p = int(parallel)
    peaks = []
    for path in Path(preflight_run).glob("chain-*-retrain*.summary.csv"):
        peaks += [v["cuda_peak_mb"] for v in _summary_rates(path).values() if v["cuda_peak_mb"]]
    if not peaks:
        print(f"FAIL no measured CUDA peak in {preflight_run}: run the preflight first")
        return 1
    try:
        free = float(subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                                    capture_output=True, text=True, check=True).stdout.split()[0])
    except (OSError, subprocess.CalledProcessError, IndexError, ValueError) as error:
        print(f"FAIL cannot read free GPU memory: {error}")
        return 1
    need = p * (max(peaks) * UNMEASURED_SCALE + GPU_CONTEXT_MB)
    ok = need <= free
    print(f"{'PASS' if ok else 'FAIL'} PARALLEL_SEEDS={p}: need {need:.0f} MB = {p} x (largest measured peak "
          f"{max(peaks):.0f} MB x {UNMEASURED_SCALE} + {GPU_CONTEXT_MB:.0f} MB context), free {free:.0f} MB")
    return 0 if ok else 1


def main(argv: list[str]) -> int:
    command, *args = argv
    if command == "stem":
        print(stem(args[0]))
        return 0
    if command == "probe":
        return probe(*args)
    if command == "bags":
        print(bags(args[0]))
        return 0
    if command == "rows":
        print(rows(*args))
        return 0
    if command == "timeouts":
        return timeouts(*args)
    if command == "bounds":
        return bounds(*args)
    if command == "fullbags":
        print(fullbags(*args))
        return 0
    if command == "timeouts6":
        return timeouts6(*args)
    if command == "retrain_chain":
        return retrain_chain(*args)
    if command == "merge_summaries":
        return merge_summaries(*args)
    if command == "retrain_plan":
        return retrain_plan(*args)
    if command == "gpu_fit":
        return gpu_fit(*args)
    raise SystemExit(f"unknown command {command!r}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
