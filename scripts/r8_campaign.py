#!/usr/bin/env python3
"""The identified-filter weekend campaign (``REFACTOR_SPECS/identified_filter/01_weekend_campaign.md``).

    python scripts/r8_campaign.py data              # Sec 1: the r8 datasets, both platforms in parallel
    python scripts/r8_campaign.py plan              # Sec 4: 2-epoch smoke on 5 % of r8 train -> campaign_plan.json
    python scripts/r8_campaign.py launch            # Sec 8: `all`, detached (systemd-run --user, sleep inhibited)
    python scripts/r8_campaign.py launch --only-run # the queue only (the r8 data and the plan already exist)
    python scripts/r8_campaign.py all               # data -> datasets.md -> plan -> run, in the foreground
    python scripts/r8_campaign.py run               # the queue in the foreground
    python scripts/r8_campaign.py status            # done / failed / pending per stage
    python scripts/r8_campaign.py chains --job J3   # write a job's chain configs, run nothing (inspection)

Everything is resumable: a stage whose ``done.json`` exists is skipped and a killed training resumes
from its last epoch (``training_overrides.resume``).  Per-stage timeout: 1.5 x the plan's estimate;
a failed or timed-out stage is retried once, then logged and the queue goes on.  A P2/P3 stage that would
start after the deadline is skipped (``02_review_round2`` R2); a P1 stage always runs.  Before each job:
>= 50 GB free disk; ``nvidia-smi`` every 10 min; one line per finished job in ``results/progress.md``.
Never writes outside the two repositories and ``REFACTOR_SPECS/identified_filter/``.  No commits.

Settings live in ``config/campaigns/r8_campaign.yaml`` (CLI flags override).  r9 (``identified_filter/07``):
``config/campaigns/r9_campaign.yaml`` with ``--config``; its ``filter:`` block switches the identified classes to the
step-2 filter (tangent spring, URDF inertia x alpha, zeta damping), and ``deadline: null`` means the plan's time +
``deadline_hours``.
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import yaml

SIM = Path(__file__).resolve().parents[1]
NN = (SIM / ".." / "dynamic_model_nn").resolve()
CONFIG = SIM / "config" / "campaigns" / "r8_campaign.yaml"
RESULTS = SIM / "REFACTOR_SPECS" / "identified_filter" / "results"
SIM_PY = ["uv", "run", "--project", str(SIM), "python"]
NN_PY = ["uv", "run", "--project", str(NN), "python"]
#: dynamic_model_nn's ``model_workflow``: a stage aborted at the zero predictor, and that predictor's share (04 C2).
STAGE_ABORT_EXIT = 3
ZERO_PREDICTOR_SHARE = 0.95


def load_settings(path: Path = CONFIG) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def log(run: Path, message: str) -> None:
    line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}  {message}"
    print(line, flush=True)
    with (run / "campaign.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


PROGRESS_HEADER = "| finished | job | platform | seeds | test RMSE (per seed) | wall [h] |\n|---|---|---|---|---|---|\n"


def progress(step: str, platform: str = "", seeds: str = "", result: str = "", hours: float | None = None) -> None:
    """One row of ``results/progress.md`` (every job, and every step of ``all``)."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "progress.md"
    if not path.exists():
        path.write_text(PROGRESS_HEADER, encoding="utf-8")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"| {dt.datetime.now():%a %H:%M} | {step} | {platform} | {seeds} | {result} | "
                     f"{'' if hours is None else f'{hours:.2f}'} |\n")


def deadline_of(settings: dict) -> dt.datetime:
    """The owner's deadline, or (``deadline: null``, 07 Sec 2) the plan's creation + ``deadline_hours``."""
    if settings.get("deadline"):
        return dt.datetime.fromisoformat(settings["deadline"])
    path = RESULTS / "campaign_plan.json"
    if path.exists():
        return dt.datetime.fromisoformat(json.loads(path.read_text())["deadline"])
    return dt.datetime.now() + dt.timedelta(hours=float(settings.get("deadline_hours", 36.0)))


# ------------------------------------------------------------------------- data ---

def platform_paths(settings: dict, platform: str) -> dict:
    spec = settings["platforms"][platform]
    root = SIM / settings["data_root"] / platform
    stem = spec["stem"]
    return {"root": root, "stem": stem, "config": SIM / spec["config"],
            **{name: root / f"{stem}{suffix}.parquet" for name, suffix in
               (("production", ""), ("gainshift", "_gainshift"), ("oop", "_oop"), ("modal", "_modal"))}}


def build_platform(settings: dict, platform: str, jobs: int, run: Path) -> int:
    """``build_round6.py --taufree --round8``; on exit 4 one escalation level higher, up to 2 (Sec 1.1.3)."""
    paths = platform_paths(settings, platform)
    paths["root"].mkdir(parents=True, exist_ok=True)
    for escalation in range(3):
        command = SIM_PY + [str(SIM / "scripts/build_round6.py"), "--config", str(paths["config"]),
                            "--out-dir", str(paths["root"]), "--taufree", "--round8", "--quiet", "--jobs", str(jobs),
                            "--coverage-escalation", str(escalation)]
        log(run, f"data {platform}: {' '.join(command[4:])}")
        with (run / f"data-{platform}-e{escalation}.log").open("w") as out:
            code = subprocess.call(command, cwd=SIM, stdout=out, stderr=subprocess.STDOUT)
        log(run, f"data {platform}: exit {code} at escalation {escalation}")
        if code != 4:
            return code
    log(run, f"data {platform}: the coverage gate still fails at escalation 2; the bags stay flagged")
    return 4


def datasets_report(settings: dict) -> None:
    """``results/datasets.md``: per platform the files, SHA-256, coverage gate and observable fractions."""
    lines = ["# r8 datasets (identified_filter/01 Sec 1)", ""]
    for platform in settings["platforms"]:
        paths = platform_paths(settings, platform)
        lines += [f"## {platform}", ""]
        for name in ("production", "gainshift", "oop", "modal"):
            contract = paths[name].with_suffix(".contract.json")
            if not contract.exists():
                lines.append(f"* {name}: **missing** ({paths[name].name})")
                continue
            c = json.loads(contract.read_text(encoding="utf-8"))
            lines.append(f"* {name}: `{paths[name].relative_to(SIM)}` sha256 `{c.get('dataset_sha256')}`")
        contract = paths["production"].with_suffix(".contract.json")
        if contract.exists():
            c = json.loads(contract.read_text(encoding="utf-8"))
            gate = (c.get("modal_coverage") or {}).get("gate") or {}
            lines += ["", f"coverage gate: **{'OK' if gate.get('gate_ok') else 'FAILED'}**; gate joints "
                          f"{gate.get('gate_joints')}; raise {gate.get('raise_joints')}; regenerated "
                          f"{len(gate.get('regenerated') or {})}; still failing {len(gate.get('still_failing') or [])}"
                          + (f"; {gate['regeneration_stopped']}" if gate.get("regeneration_stopped") else ""), "",
                      "| regime | joint | resolvable bags | observable | median K rel. err. | median SNR |",
                      "|---|---|---|---|---|---|"]
            for regime, joints in (gate.get("per_regime") or {}).items():
                for joint, row in joints.items():
                    lines.append(f"| {regime} | {joint} | {row['resolvable_bags']} | {row['observable_fraction']:.2f} | "
                                 f"{row['median_K_relerr']:.3g} | {row['median_snr']:.3g} |")
            lines += ["", "band (modal_coverage.band):", "",
                      "| joint | f_n min | f_n max | f_lo | f_hi | resolvable |", "|---|---|---|---|---|---|"]
            for row in (c.get("modal_coverage") or {}).get("band", []):
                lines.append(f"| {row['joint']} | {row['f_n_min_hz']:.2f} | {row['f_n_max_hz']:.1f} | "
                             f"{row['f_lo_hz']:.2f} | {row['f_hi_hz']:.1f} | {row['resolvable']} |")
        lines.append("")
    # 03 F3: the oracle-fit diagnosis and the target it chose (tools/r8_oracle_diagnosis.py; bags off the old build).
    lines += [(RESULTS / name).read_text(encoding="utf-8") for name in ("oracle_verdict.md", "oracle_diagnosis.md")
              if (RESULTS / name).exists()]
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "datasets.md").write_text("\n".join(lines), encoding="utf-8")


def cmd_data(settings: dict, args) -> int:
    run = run_dir(settings)
    jobs = max(1, (os.cpu_count() or 2) // len(settings["platforms"]) - 1)
    threads, codes = [], {}
    for platform in settings["platforms"]:
        thread = threading.Thread(target=lambda p=platform: codes.__setitem__(p, build_platform(settings, p, jobs, run)))
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
    datasets_report(settings)
    log(run, f"data done: {codes}; report {RESULTS / 'datasets.md'}")
    return 0 if all(code in (0, 3, 4) for code in codes.values()) else 1


# ----------------------------------------------------------------------- chains ---

def r7_runs(settings: dict, platform: str) -> dict:
    chain = yaml.safe_load((SIM / settings["platforms"][platform]["r7_chain"]).read_text(encoding="utf-8"))
    return {run["base_name"]: run for run in chain["runs"]}, chain["defaults"]


def contract_kwargs(settings: dict, platform: str) -> dict:
    """The two-mass constructor kwargs from the r8 production contract (Sec 2.3)."""
    contract = json.loads(platform_paths(settings, platform)["production"].with_suffix(".contract.json")
                          .read_text(encoding="utf-8"))
    priors, joints = contract["transmission_priors"], contract["transmission_priors"]["joints"]
    f_link = max(j["mode_frequency_hz_range"][1] for j in joints)
    sg = float(contract["differentiation"]["sg_cutoff_hz"])
    return {"time_step": priors["sample_time_step"], "burn_in_s": priors["burn_in_s"],
            "stiffness_range": [j["stiffness_range"] for j in joints],
            "damping_ratio_range": [j["damping_ratio_range"] for j in joints],
            "reference_inertia": [j["link_inertia_home"] for j in joints],
            "damping_range": [j["damping_range"] for j in joints],
            # Sec 2.3: f_c = min(SG cutoff, 2 x the highest f_link of the contract prior); fixed, never tuned.
            "input_lowpass_hz": min(sg, 2.0 * f_link),
            "zeta_max": [max(1.0, 1.5 * j["damping_ratio_range"][1]) for j in joints]}


_STEP2_CACHE: dict = {}


def step2_kwargs(settings: dict, platform: str) -> dict:
    """The step-2 filter's constructor kwargs (07 Sec 2) from the settings' ``filter:`` block and the platform's
    config and asset; ``{}`` without the block (r8).

    The catalogue curve the plant uses (``spring_factors`` per joint, ``spring_knees`` in torque), the warm-start
    cutoff 0.5 x the lowest main-joint f_n of the band table (07 S1 (iii)) and the bare URDF's chain table.
    """
    block = settings.get("filter")
    if not block:
        return {}
    if platform not in _STEP2_CACHE:
        sys.path[:0] = [str(SIM / "src"), str(SIM / "scripts")]
        import numpy as np

        from diagnose_controller_modes import _load_asset
        from elastic_sim.dataset import load_config, torque_knees
        from elastic_sim.inertia_table import chain_table

        config = load_config(platform_paths(settings, platform)["config"])
        asset = _load_asset(config.asset)
        n = len(asset.joint_names)
        effort = np.asarray([np.inf if j.effort is None else j.effort for j in asset.resolve_active_joints()])
        factors = np.atleast_2d(np.asarray(config.plant_extras.stiffness_factors, dtype=float))
        knees = np.atleast_2d(torque_knees(config.plant_extras, effort))
        main = set(settings["platforms"][platform]["main_joints"])
        cutoff = float(block.get("warm_cutoff_ratio", 0.5)) * min(
            float(row["f_n_min_hz"]) for row in config.round8.band if row["joint"] in main)
        table = chain_table(asset, check_samples=0)
        table.pop("check")
        _STEP2_CACHE[platform] = {
            "spring": block.get("spring", "tangent"), "inertia": "urdf", "damping": "zeta",
            "spring_factors": np.broadcast_to(factors, (n, factors.shape[1])).tolist(),
            "spring_knees": np.broadcast_to(knees, (n, knees.shape[1])).tolist(),
            "spring_warm_cutoff_hz": cutoff, "urdf_chain": table,
            "alpha_range": list(block.get("alpha_range", [0.7, 1.4])),
            "stiffness_widening": float(block.get("stiffness_widening", 1.0)),
            "inertia_prefit_steps": int(block.get("inertia_prefit_steps", 500)), "lambda_inertia": None}
    return copy.deepcopy(_STEP2_CACHE[platform])


def privileged_source(settings: dict, platform: str) -> str:
    import pyarrow.parquet as pq

    names = pq.ParquetFile(platform_paths(settings, platform)["production"]).schema.names
    wanted = settings.get("privileged_source", "effective")
    return wanted if wanted == "sampled" or any(n.startswith("K_eff__") for n in names) else "sampled"


def stage_runs(settings: dict, job: dict, platform: str) -> list[dict]:
    """One dict per stage of ``job`` on ``platform``: class, kwargs, overrides, init (by stage name)."""
    runs, defaults = r7_runs(settings, platform)
    kw = contract_kwargs(settings, platform)
    spec = settings["platforms"][platform]
    kan = {k: v for k, v in (runs["kalnn_twomass"].get("model_kwargs") or {}).items() if k.startswith("kan_")}
    decimation = int(spec.get("decimation", 1))
    epochs, patience = settings["epochs"], settings["patience"]
    student_epochs, student_patience = settings["student_epochs"], settings["student_patience"]

    def phys(filter_param, backbone="kalnn", **extra):
        base = "kalnn_twomass" if backbone == "kalnn" else "lnn_twomass"
        kwargs = {key: kw[key] for key in ("time_step", "burn_in_s", "stiffness_range", "damping_ratio_range",
                                           "reference_inertia", "damping_range", "input_lowpass_hz")}
        kwargs.update(coupling=settings["coupling"], integrator="trapezoidal", filter_param=filter_param,
                      decimation=decimation, **(kan if backbone == "kalnn" else {}))
        kwargs.update(step2_kwargs(settings, platform))
        kwargs.update(extra)
        overrides = dict(runs[base]["training_overrides"], epochs=epochs, early_stopping_patience=patience,
                         selection="val", resume=True, privileged_source=privileged_source(settings, platform),
                         decimation_p=float(spec.get("decimation_p", 0.0)) if decimation > 1 else 0.0,
                         grad_clip_norm=settings.get("grad_clip_norm"), fail_fast_epochs=settings.get("fail_fast_epochs"))
        # 04 C2: the identified classes only; the r7 comparison classes keep their r7 training.
        overrides["learning_rate"] = float(overrides.get("learning_rate", defaults["learning_rate"])) \
            * float(spec.get("identified_lr_scale", 1.0))
        return {"class_path": f"{base}_id.LagrangianNN", "model_kwargs": kwargs, "overrides": overrides}

    def student(teacher_stage, backbone="kalnn", s_selection="val_param", **extra):
        s = phys("encoder", backbone, **extra)
        s["overrides"].update(stage="S", selection=s_selection, epochs=student_epochs,
                              early_stopping_patience=student_patience)
        s.update(name="S", init=teacher_stage)
        j = copy.deepcopy(phys("encoder", backbone, **extra))
        j["overrides"].update(stage="J", selection="val", epochs=student_epochs, early_stopping_patience=student_patience)
        j.update(name="J", init="S")
        return [s, j]

    kind = job["kind"]
    if kind == "teacher":
        return [dict(phys("privileged", lambda_param=0.0), name="T")]
    if kind == "phys":
        return [dict(phys("KD", lambda_param=0.0), name="P")]
    if kind == "student":
        extra = dict(job.get("model_kwargs") or {})
        teacher = job.get("teacher", "J1")
        stages = []
        if job.get("own_teacher"):
            backbone = extra.get("backbone", "kalnn")
            teacher_kwargs = {k: v for k, v in extra.items() if k in ("residual_inputs",)}
            stages.append(dict(phys("privileged", extra.get("backbone", "kalnn"), lambda_param=0.0, **teacher_kwargs),
                               name="T"))
            teacher = "T"
        backbone = extra.pop("backbone", "kalnn")
        s_selection = "val" if float(extra.get("lambda_param", 1.0)) == 0.0 else "val_param"
        return stages + student(teacher, backbone, s_selection, **extra)
    if kind == "r7_residual":
        run = runs["kalnn_tau_res"]
        kwargs = dict(run.get("model_kwargs") or {}, **(job.get("model_kwargs") or {}))
        overrides = dict(run["training_overrides"], epochs=epochs, early_stopping_patience=patience, selection="val",
                         resume=True)
        return [{"name": "R", "class_path": "kalnn_tau_res.LagrangianNN", "model_kwargs": kwargs,
                 "overrides": overrides}]
    if kind == "r7_twomass":
        run = runs["kalnn_twomass"]
        kwargs = dict(run.get("model_kwargs") or {})
        kwargs.update({k: kw[k] for k in ("time_step", "burn_in_s", "stiffness_range", "damping_ratio_range",
                                          "zeta_max")}, coupling=settings["coupling"], integrator="trapezoidal")
        overrides = dict(run["training_overrides"], epochs=epochs, early_stopping_patience=patience, selection="val",
                         resume=True)
        return [{"name": "R", "class_path": "kalnn_twomass.LagrangianNN", "model_kwargs": kwargs,
                 "overrides": overrides}]
    raise ValueError(f"unknown job kind {kind!r}")


def model_path(settings: dict, job: str, platform: str, seed: int, stage: str) -> Path:
    # The file name carries job and platform: dynamic_model_nn keys its loss files on the basename.
    return NN / "models" / settings["run_id"] / job / platform / f"{job}_{platform}_s{seed}_{stage}.pt"


def teacher_of(job_name: str, stage: dict) -> tuple[str, str] | None:
    """``(job, stage)`` of the teacher a stage initialises from; ``None`` for a stage continuing its own student."""
    init = stage.get("init")
    if init is None or init == "S":
        return None
    return (job_name, "T") if init == "T" else (init, "T")


def teacher_failure(run: Path, job_name: str, platform: str, seed: int, stage: dict) -> str | None:
    """04 C3: why this stage's teacher can't initialise a student, or ``None``.

    A teacher fails when it left no result, or its best val RMSE stayed within 5 % of the zero predictor's.
    A run predating ``val_zero_rmse`` is not judged on the second count.
    """
    teacher = teacher_of(job_name, stage)
    if teacher is None:
        return None
    done = stage_done(run, teacher[0], platform, seed, teacher[1])
    if not done.exists():
        return f"teacher {teacher[0]} s{seed} left no result"
    row = json.loads(done.read_text())
    try:
        best, zero = float(row["best_val_rmse"]), float(row["val_zero_rmse"])
    except (KeyError, TypeError, ValueError):
        return None
    if best >= ZERO_PREDICTOR_SHARE * zero:
        return f"teacher {teacher[0]} s{seed} stayed at the zero predictor (val {best:.4g} vs {zero:.4g})"
    return None


def checkpoint_of(path: Path) -> Path:
    return path.with_name(path.stem + "_checkpoint.pt")


def write_chain(settings: dict, job_name: str, job: dict, platform: str, seed: int, stage: dict,
                dataset: Path, run: Path) -> Path:
    _, defaults = r7_runs(settings, platform)
    out = model_path(settings, job_name, platform, seed, stage["name"])
    out.parent.mkdir(parents=True, exist_ok=True)
    overrides = dict(stage["overrides"])
    if stage.get("init"):
        teacher_job = job_name if stage["init"] in ("S", "T") else stage["init"]
        teacher_stage = stage["init"] if stage["init"] in ("S", "T") else "T"
        overrides["init_from"] = str(checkpoint_of(model_path(settings, teacher_job, platform, seed, teacher_stage)))
    chain = {"defaults": {**{k: v for k, v in defaults.items() if k not in ("seeds", "dataset_file")},
                          "dataset_file": str(dataset), "summary_dir": str(run / "log"), "stop_on_error": True},
             "runs": [{"base_name": f"{job_name}_{stage['name']}", "class_path": stage["class_path"],
                       "model_kwargs": stage["model_kwargs"], "training_overrides": overrides,
                       "output_model_path": str(out)}]}
    path = run / "chains" / job_name / platform / f"s{seed}_{stage['name']}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# generated by scripts/r8_campaign.py -- do not hand-edit\n" + yaml.safe_dump(chain, sort_keys=False),
                    encoding="utf-8")
    return path


# ------------------------------------------------------------------------ queue ---

def run_dir(settings: dict) -> Path:
    run = SIM / "runs" / settings["run_id"]
    run.mkdir(parents=True, exist_ok=True)
    return run


def stage_done(run: Path, job: str, platform: str, seed: int, stage: str) -> Path:
    return run / "jobs" / job / platform / f"s{seed}_{stage}.done.json"


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 2 ** 30


def nvidia_logger(run: Path, stop: threading.Event) -> None:
    while not stop.is_set():
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=timestamp,utilization.gpu,memory.used,memory.total,"
                                  "temperature.gpu", "--format=csv,noheader"], capture_output=True, text=True,
                                 timeout=30).stdout
            with (run / "nvidia-smi.log").open("a") as handle:
                handle.write(out)
        except (OSError, subprocess.SubprocessError):
            pass
        stop.wait(600)


def stage_timeout_s(settings: dict, estimate_h: float) -> float:
    """Per-stage timeout.  The plan's estimate is one seed alone on the GPU; the queue runs ``parallel_seeds``
    at once, so scale by it, and never below ``min_stage_timeout_h`` (default 12 h): the epoch cap, not the
    timeout, bounds a healthy training (2026-10-31: 1.5 x the one-seed estimate killed all three J1 seeds)."""
    scaled = 1.5 * float(estimate_h) * max(1, int(settings.get("parallel_seeds", 1)))
    return 3600.0 * max(scaled, float(settings.get("min_stage_timeout_h", 12.0)))


def run_stage(settings: dict, run: Path, job_name: str, job: dict, platform: str, seed: int, stage: dict,
              dataset: Path, timeout_s: float, epochs: int | None = None) -> dict | None:
    done = stage_done(run, job_name, platform, seed, stage["name"])
    if done.exists() and epochs is None:
        return json.loads(done.read_text())
    chain = write_chain(settings, job_name, job, platform, seed, stage, dataset, run)
    summary = chain.with_suffix(".summary.csv")
    command = NN_PY + [str(NN / "train_chain_from_config.py"), "--config", str(chain), "--seeds", str(seed),
                       "--summary-out", str(summary)] + (["--epochs", str(epochs)] if epochs else [])
    logfile = chain.with_suffix(".log")
    for attempt in (1, 2):
        started = time.time()
        with logfile.open("a") as out:
            out.write(f"\n=== attempt {attempt} {dt.datetime.now()}\n")
            out.flush()
            # Own process group, so a timeout kills uv *and* the python trainer under it (2026-10-31: killing
            # only uv left the trainers running beside the retry); unbuffered, so the log shows the epochs.
            proc = subprocess.Popen(command, cwd=NN, stdout=out, stderr=subprocess.STDOUT, start_new_session=True,
                                    env={**os.environ, "PYTHONUNBUFFERED": "1"})
            try:
                code = proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                code = "timeout"
                for sig, grace in ((signal.SIGTERM, 60), (signal.SIGKILL, 30)):
                    try:
                        os.killpg(proc.pid, sig)
                    except ProcessLookupError:
                        break
                    try:
                        proc.wait(timeout=grace)
                        break
                    except subprocess.TimeoutExpired:
                        continue
        if code == STAGE_ABORT_EXIT:
            log(run, f"FAIL  {job_name} {platform} s{seed} {stage['name']}: at the zero predictor, not retried ({logfile})")
            break
        if code == 0 and summary.exists():
            row = next(csv.DictReader(summary.open()))
            row.update(wall_s=time.time() - started, attempt=attempt)
            if epochs is None:
                done.parent.mkdir(parents=True, exist_ok=True)
                done.write_text(json.dumps(row, indent=1))
            return row
        log(run, f"FAIL  {job_name} {platform} s{seed} {stage['name']} attempt {attempt}: {code} ({logfile})")
    done.parent.mkdir(parents=True, exist_ok=True)   # a job's first stage may never have finished
    (done.with_suffix(".failed")).write_text(str(code))
    return None


def evaluate_job(settings: dict, run: Path, job_name: str, platform: str, rows: dict) -> None:
    """Sec 6.1 after each job: every seed's final stage on test, gain-shift, oop and modal (teacher also privileged)."""
    paths = platform_paths(settings, platform)
    head = contract_kwargs(settings, platform)["burn_in_s"]
    out_dir = run / "eval" / job_name
    out_dir.mkdir(parents=True, exist_ok=True)
    for seed, row in rows.items():
        checkpoint = row["checkpoint_path"]
        for name in ("production", "gainshift", "oop", "modal"):
            if not paths[name].exists():
                continue
            # 04 C4: the chirp file has no verification, so no K_eff, D_eff for the privileged path to read.
            for privileged in ([False, True] if row.get("base_name", "").endswith("_T") and name != "modal" else [False]):
                tag = f"{platform}-{name}" + ("-privileged" if privileged else "")
                marker = out_dir / f"{tag}-s{seed}.done"
                if marker.exists():
                    continue
                command = NN_PY + [str(NN / "evaluate_on.py"), "--checkpoint", checkpoint, "--dataset", str(paths[name]),
                                   "--split", "test", "--exclude-head-s", str(head), "--out", str(out_dir / f"{tag}.csv"),
                                   "--per-bag-out", str(out_dir / f"{tag}.per_bag.csv")]
                if privileged:
                    command.append("--privileged")
                with (out_dir / f"{tag}-s{seed}.log").open("w") as log_out:
                    code = subprocess.call(command, cwd=NN, stdout=log_out, stderr=subprocess.STDOUT, timeout=4 * 3600)
                if code == 0:
                    marker.touch()
                else:
                    log(run, f"FAIL  eval {job_name} {tag} s{seed}: exit {code}")


def plan_estimates(run: Path) -> dict:
    path = RESULTS / "campaign_plan.json"
    if not path.exists():
        return {}
    plan = json.loads(path.read_text())
    return {(j["job"], j["platform"], j["stage"]): j for j in plan.get("stages", [])}


def cmd_run(settings: dict, args) -> int:
    run = run_dir(settings)
    estimates = plan_estimates(run)
    plan = json.loads((RESULTS / "campaign_plan.json").read_text()) if (RESULTS / "campaign_plan.json").exists() else {}
    dropped, reasons = set(plan.get("dropped", [])), plan.get("dropped_reasons", {})
    stop = threading.Event()
    threading.Thread(target=nvidia_logger, args=(run, stop), daemon=True).start()
    deadline = deadline_of(settings)
    log(run, f"queue start: run {settings['run_id']}, parallel seeds {settings['parallel_seeds']}, deadline {deadline}")
    try:
        for job_name, job in sorted(settings["jobs"].items(), key=lambda kv: (kv[1]["priority"], kv[0])):
            if args.job and job_name not in args.job:
                continue
            for platform in settings["platforms"]:
                if f"{job_name}:{platform}" in dropped or job_name in dropped:
                    reason = reasons.get(f"{job_name}:{platform}", "past the deadline")
                    log(run, f"SKIP  {job_name} {platform}: dropped by the plan (Sec 4): {reason}")
                    continue
                if free_gb(SIM) < settings["min_free_gb"]:
                    log(run, f"STOP  {free_gb(SIM):.0f} GB free < {settings['min_free_gb']} GB")
                    return 2
                dataset = platform_paths(settings, platform)["production"]
                stages = stage_runs(settings, job, platform)
                seeds = job.get("seeds", settings["seeds"])
                started = time.time()
                results: dict = {}

                def one_seed(seed):
                    row = None
                    for stage in stages:
                        # R2: a P2/P3 stage that would start after the deadline is skipped; P1 always runs.
                        if job["priority"] > 1 and dt.datetime.now() >= deadline and \
                                not stage_done(run, job_name, platform, seed, stage["name"]).exists():
                            log(run, f"SKIP  {job_name} {platform} s{seed} {stage['name']}: would start after the "
                                     f"deadline {deadline}")
                            return
                        failure = teacher_failure(run, job_name, platform, seed, stage)
                        if failure:
                            log(run, f"SKIP  {job_name} {platform} s{seed} {stage['name']}: {failure}")
                            return
                        estimate = estimates.get((job_name, platform, stage["name"]), {}).get("hours", 12.0)
                        row = run_stage(settings, run, job_name, job, platform, seed, stage, dataset,
                                        timeout_s=stage_timeout_s(settings, estimate))
                        if row is None:
                            return
                    results[seed] = row

                pool = []
                for index in range(0, len(seeds), settings["parallel_seeds"]):
                    batch = [threading.Thread(target=one_seed, args=(s,)) for s in seeds[index:index + settings["parallel_seeds"]]]
                    for thread in batch:
                        thread.start()
                    for thread in batch:
                        thread.join()
                    pool += batch
                if results:
                    evaluate_job(settings, run, job_name, platform, results)
                rmse = ", ".join(f"s{s}: {float(r['rmse']):.4g}" for s, r in sorted(results.items()))
                progress(job_name, platform, f"{sorted(results)}/{seeds}", rmse or "failed or skipped",
                         (time.time() - started) / 3600)
                log(run, f"DONE  {job_name} {platform}: {rmse or 'no seed finished'}")
        # Sec 6: the final sweep: Phase 0c again on r8 (Sec 3), the real files whatever exists by then
        # (idempotent), then the Sec 7 summary.
        if not args.job:
            for platform in settings["platforms"]:
                paths = platform_paths(settings, platform)
                files = [str(paths[k]) for k in ("production", "gainshift") if paths[k].exists()]
                if files:
                    code = subprocess.call(NN_PY + [str(NN / "phase0_diagnostics.py"), "0c", "--dataset", *files,
                                                    "--tag", f"r8_{platform}", "--out", str(RESULTS / "phase0")], cwd=NN)
                    log(run, f"final sweep phase0 0c r8 {platform}: exit {code}")
            for script in (["bash", str(SIM / "scripts/run_real_eval.sh")],
                           SIM_PY + [str(SIM / "scripts/r8_summary.py"), "--run-id", settings["run_id"]]):
                code = subprocess.call(script, cwd=SIM, env={**os.environ, "RUN_ID": settings["run_id"]})
                log(run, f"final sweep {' '.join(script[-3:])}: exit {code}")
    finally:
        stop.set()
    log(run, "queue finished")
    return 0


# ------------------------------------------------------------------------ plan ----

def smoke_subset(settings: dict, platform: str, run: Path, fraction: float = 0.05) -> Path:
    """5 % of the train bags (>= 2), 5 % of val (>= 1) and one test bag, with the contract beside it."""
    import numpy as np
    import pandas as pd

    source = platform_paths(settings, platform)["production"]
    out = run / "smoke" / platform / source.name
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(source)
    bags = frame.groupby("bag", sort=False)["split"].first()
    rng = np.random.default_rng(0)
    keep = []
    for split, minimum, share in (("train", 2, fraction), ("val", 1, fraction), ("test", 1, 0.0)):
        names = bags[bags == split].index.to_numpy()
        count = max(minimum, int(round(share * len(names))))
        keep += list(rng.choice(names, size=min(count, len(names)), replace=False))
    frame[frame["bag"].isin(keep)].to_parquet(out, index=False)
    for suffix in (".contract.json", ".manifest.json"):
        shutil.copy(source.with_suffix(suffix), out.with_suffix(suffix))
    return out


def schedule(stages_out: list[dict], start: dt.datetime, deadline: dt.datetime, run_id: str) -> tuple:
    """``(dropped, reasons, total_hours)`` of the plan; sets each kept stage's cumulative hours and expected end.

    R1: a job with a stage whose smoke test failed is dropped with the reason (the queue runs the rest).
    Sec 4: past the deadline, jobs are dropped from the tail (P3 first, then P2), never P1.
    """
    jobs: dict = {}
    for stage in stages_out:
        jobs.setdefault((stage["priority"], stage["job"], stage["platform"]), []).append(stage)
    order = sorted(jobs)
    name = lambda key: f"{key[1]}:{key[2]}"  # noqa: E731
    hours = lambda key: sum(s["wall_hours"] for s in jobs[key] if s["wall_hours"] == s["wall_hours"])  # noqa: E731
    dropped, reasons = [], {}
    for key in order:
        failed = [s["stage"] for s in jobs[key] if not s["smoke_ok"]]
        if failed:
            dropped.append(name(key))
            reasons[name(key)] = f"smoke test failed on stage(s) {', '.join(failed)} (runs/{run_id}/smoke)"
    total = sum(hours(key) for key in order if name(key) not in dropped)
    for key in reversed(order):
        if start + dt.timedelta(hours=total) <= deadline or key[0] == 1 or name(key) in dropped:
            continue
        total -= hours(key)
        dropped.append(name(key))
        reasons[name(key)] = "past the deadline (Sec 4)"
    cumulative = 0.0
    for key in order:
        if name(key) in dropped:
            continue
        cumulative += hours(key)
        for stage in jobs[key]:
            stage["cumulative_hours"] = cumulative
            stage["expected_end"] = (start + dt.timedelta(hours=cumulative)).isoformat(timespec="minutes")
    return dropped, reasons, total


def cmd_plan(settings: dict, args) -> int:
    """Sec 4: each distinct stage for 2 epochs on 5 % of r8 train -> s/epoch -> hours; drop the tail past the deadline."""
    import pandas as pd

    run = run_dir(settings)
    stages_out = []
    for job_name, job in sorted(settings["jobs"].items(), key=lambda kv: (kv[1]["priority"], kv[0])):
        for platform in settings["platforms"]:
            subset = smoke_subset(settings, platform, run)
            full_train = pd.read_parquet(platform_paths(settings, platform)["production"], columns=["bag", "split"]) \
                .groupby("bag")["split"].first().eq("train").sum()
            smoke_train = pd.read_parquet(subset, columns=["bag", "split"]).groupby("bag")["split"].first().eq("train").sum()
            r7 = settings["platforms"][platform]["r7_epochs"]
            for stage in stage_runs(settings, job, platform):
                smoke_settings = copy.deepcopy(settings)
                smoke_settings["run_id"] = settings["run_id"] + "-smoke"
                row = run_stage(smoke_settings, run / "smoke", job_name, job, platform, 0, stage, subset,
                                timeout_s=3600, epochs=2)
                seconds = float(row["seconds_per_epoch"]) if row and row.get("seconds_per_epoch") else float("nan")
                # R2: the r7 epoch counts, capped by the r8 budget.
                expected = min(r7.get(stage["class_path"].split(".")[0], r7["default"]), settings["epochs"]) \
                    if stage["name"] in ("T", "P", "R") else settings["student_epochs"] * 0.7
                hours = seconds * full_train / max(1, smoke_train) * expected / 3600.0
                seeds = len(job.get("seeds", settings["seeds"]))
                passes = -(-seeds // settings["parallel_seeds"])
                stages_out.append({"job": job_name, "priority": job["priority"], "platform": platform,
                                   "stage": stage["name"], "smoke_seconds_per_epoch": seconds,
                                   "expected_epochs": expected, "hours": hours, "seeds": seeds,
                                   "wall_hours": hours * passes, "smoke_ok": row is not None})
    start = dt.datetime.now()
    deadline = dt.datetime.fromisoformat(settings["deadline"]) if settings.get("deadline") \
        else start + dt.timedelta(hours=float(settings.get("deadline_hours", 36.0)))
    dropped, reasons, total = schedule(stages_out, start, deadline, settings["run_id"])
    plan = {"created": start.isoformat(timespec="minutes"), "deadline": deadline.isoformat(timespec="minutes"),
            "deadline_source": "owner" if settings.get("deadline") else f"plan + {settings.get('deadline_hours', 36)} h",
            "parallel_seeds": settings["parallel_seeds"], "total_hours": total, "dropped": dropped,
            "dropped_reasons": reasons,
            "fits": start + dt.timedelta(hours=total) <= deadline, "stages": stages_out}
    if settings.get("filter"):
        plan["selector_cost"] = selector_cost(settings, run, stages_out)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "campaign_plan.json").write_text(json.dumps(plan, indent=1), encoding="utf-8")
    print(f"plan: {total:.1f} h at {settings['parallel_seeds']} parallel seeds; dropped {dropped or 'none'}; "
          f"{RESULTS / 'campaign_plan.json'}")
    return 0 if all(s["smoke_ok"] for s in stages_out) else 1


def selector_cost(settings: dict, run: Path, stages_out: list[dict]) -> dict:
    """07 S1.4: per platform, the teacher's smoke s/epoch with the fixed-point selector against one pass."""
    out = {}
    for job_name, job in settings["jobs"].items():
        if job["kind"] != "teacher":
            continue
        for platform in settings["platforms"]:
            stage = copy.deepcopy(stage_runs(settings, job, platform)[0])
            stage["model_kwargs"]["spring_passes"] = 1
            stage["name"] = "T1pass"
            smoke_settings = copy.deepcopy(settings)
            smoke_settings["run_id"] = settings["run_id"] + "-smoke"
            row = run_stage(smoke_settings, run / "smoke", job_name, job, platform, 0, stage,
                            smoke_subset(settings, platform, run), timeout_s=3600, epochs=2)
            one = float(row["seconds_per_epoch"]) if row and row.get("seconds_per_epoch") else float("nan")
            fixed = next((s["smoke_seconds_per_epoch"] for s in stages_out
                          if s["job"] == job_name and s["platform"] == platform and s["stage"] == "T"), float("nan"))
            out[platform] = {"job": job_name, "fixed_point_seconds_per_epoch": fixed, "one_pass_seconds_per_epoch": one,
                             "ratio": fixed / one if one == one and one > 0 else float("nan")}
    return out


# -------------------------------------------------------------------------- all ---

def cmd_all(settings: dict, args) -> int:
    """R1: data -> datasets.md -> plan -> run, unattended.  It stops only when no platform has a contract."""
    run = run_dir(settings)
    steps = (("data", cmd_data), ("plan", cmd_plan), ("run", cmd_run))
    for name, step in steps:
        started = time.time()
        log(run, f"all: {name} start")
        try:
            code = step(settings, args)
        except Exception as error:  # noqa: BLE001  (logged; the decision below says whether to go on)
            code = f"{type(error).__name__}: {error}"
        hours = (time.time() - started) / 3600
        log(run, f"all: {name} exit {code} ({hours:.2f} h)")
        progress(f"all: {name}", result=f"exit {code}", hours=hours)
        if name == "data":
            # 03 F5: train what exists.  A platform whose build failed is logged and skipped.
            built = {p: spec for p, spec in settings["platforms"].items()
                     if platform_paths(settings, p)["production"].with_suffix(".contract.json").exists()}
            missing = sorted(set(settings["platforms"]) - set(built))
            if not built:
                log(run, f"STOP  all: no r8 production contract for {missing}; nothing to train")
                return 1
            if missing:
                log(run, f"SKIP  all: no r8 production contract for {missing}; training on {sorted(built)}")
                progress("all: skipped", platform=" ".join(missing), result="no production contract")
            settings = {**settings, "platforms": built}
        if name == "plan" and code != 0:
            # The plan's smoke failures are in campaign_plan.json `dropped`; the queue runs the rest.
            log(run, "all: plan reported smoke failures (dropped in campaign_plan.json); going on")
        if name == "run":
            return code if isinstance(code, int) else 1
    return 0


# ----------------------------------------------------------------------- launch ---

SLEEP_TARGETS = ("sleep.target", "suspend.target", "hibernate.target", "hybrid-sleep.target")
INHIBIT = ["systemd-inhibit", "--what=sleep:idle:handle-lid-switch", "--who=r8-campaign",
           "--why=r8 identified-filter weekend campaign", "--mode=block"]


def _quiet(command: list[str]) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None


def sleep_guard() -> tuple[list[str] | None, str]:
    """R3: ``(prefix, message)``.  ``[]`` when the sleep targets are masked (the only reliable guard on peepo),
    ``None`` (refuse) otherwise.  ``systemd-inhibit`` is not used: polkit allows it from the desktop terminal
    but denies it inside a ``systemd-run --user`` unit, which has no session (2026-10-30: the unit failed
    with "Failed to inhibit: Access denied")."""
    states = {}
    for target in SLEEP_TARGETS:
        out = _quiet(["systemctl", "show", "-p", "LoadState", "--value", target])
        states[target] = out.stdout.strip() if out is not None else "unknown"
    if all(state == "masked" for state in states.values()):
        return [], "sleep: the sleep targets are masked: OK"
    return None, (f"sleep: the sleep targets are not all masked ({states}); systemd-inhibit cannot be used "
                  f"inside the --user unit.\n  mask them with:  sudo systemctl mask {' '.join(SLEEP_TARGETS)}")


def linger_check() -> str:
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or ""
    out = _quiet(["loginctl", "show-user", user, "-p", "Linger"])
    value = out.stdout.strip() if out is not None and out.returncode == 0 else "Linger=unknown"
    if value == "Linger=yes":
        return "linger: yes"
    return (f"WARNING linger: {value}: a --user unit dies at logout; stay logged in or run "
            f"`loginctl enable-linger {user}`")


def cmd_launch(settings: dict, args) -> int:
    unit = settings.get("unit", "r8-campaign")
    prefix, sleep_message = sleep_guard()
    print(sleep_message)
    print(linger_check())
    step = "run" if args.only_run else "all"
    uv = shutil.which("uv") or "uv"
    command = ["systemd-run", "--user", f"--unit={unit}", "--collect", f"--working-directory={SIM}",
               "--property=KillSignal=SIGTERM", f"--setenv=PATH={os.environ.get('PATH', '')}",
               *(prefix or []), uv, *SIM_PY[1:], str(Path(__file__).resolve()), step,
               "--config", str(Path(args.config).resolve())]
    print(" ".join(command))
    print(f"unit: {unit}\nwatch: journalctl --user -u {unit} -f\n       python scripts/r8_campaign.py status\n"
          f"       tail -f runs/{settings['run_id']}/campaign.log")
    if prefix is None:
        print("refusing to launch: the machine may suspend mid-run (see above)" +
              (" [dry run: nothing launched]" if args.dry_run else ""))
        return 0 if args.dry_run else 1
    if args.dry_run:
        return 0
    return subprocess.call(command)


def cmd_status(settings: dict, args) -> int:
    run = run_dir(settings)
    for job_name, job in sorted(settings["jobs"].items(), key=lambda kv: (kv[1]["priority"], kv[0])):
        for platform in settings["platforms"]:
            states = []
            for seed in job.get("seeds", settings["seeds"]):
                for stage in ("T", "P", "R", "S", "J"):
                    done = stage_done(run, job_name, platform, seed, stage)
                    if done.exists():
                        states.append(f"s{seed}{stage}:done")
                    elif done.with_suffix(".failed").exists():
                        states.append(f"s{seed}{stage}:FAILED")
            print(f"P{job['priority']} {job_name:4s} {platform:10s} {' '.join(states) or 'pending'}")
    return 0


def cmd_chains(settings: dict, args) -> int:
    run = run_dir(settings)
    for job_name in args.job or settings["jobs"]:
        job = settings["jobs"][job_name]
        for platform in settings["platforms"]:
            for seed in job.get("seeds", settings["seeds"]):
                for stage in stage_runs(settings, job, platform):
                    print(write_chain(settings, job_name, job, platform, seed, stage,
                                      platform_paths(settings, platform)["production"], run))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("data", "plan", "run", "all", "launch", "status", "chains"))
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--job", nargs="+", default=None, help="only these jobs (run, chains)")
    parser.add_argument("--parallel-seeds", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="launch: print the checks and the systemd-run command only")
    parser.add_argument("--only-run", action="store_true", help="launch: the queue only (`run`), not `all`")
    args = parser.parse_args(argv)
    settings = load_settings(Path(args.config))
    global RESULTS
    if settings.get("results_dir"):
        RESULTS = Path(settings["results_dir"]) if Path(settings["results_dir"]).is_absolute() \
            else SIM / settings["results_dir"]
    if args.parallel_seeds:
        settings["parallel_seeds"] = args.parallel_seeds
    return {"data": cmd_data, "plan": cmd_plan, "run": cmd_run, "all": cmd_all, "launch": cmd_launch,
            "status": cmd_status, "chains": cmd_chains}[args.command](settings, args)


if __name__ == "__main__":
    raise SystemExit(main())
