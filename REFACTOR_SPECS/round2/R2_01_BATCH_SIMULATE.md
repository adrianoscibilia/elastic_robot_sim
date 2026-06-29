# R2_01 — Batch "simulate every JSON in a recordings folder"

Goal: one command that takes a folder of recordings, simulates each baked trajectory with a
given model, writes a sim parquet next to each recording, and (optionally) reports the
sim-vs-real error per recording. This is steps 1–2 of the sim2real loop in batch form and
the input to manual inspection / calibration.

## 1. Shared loader (F5) — `src/elastic_sim/recordings.py` (NEW)

Extract the recordings-pairing logic currently duplicated in `run_calibration.py` so both
the batch script and the calibrator use one implementation.

```python
"""Discover and load real-robot recordings as (RolloutResult, TrajectoryConfig, id) triples."""
from __future__ import annotations
import glob, os
from dataclasses import dataclass
import pandas as pd
from .rollout import RolloutResult, RolloutStore
from .trajectory import TrajectoryConfig

@dataclass
class Recording:
    rec_id: str                 # timestamp (flat) or traj_id (structured)
    traj_config: TrajectoryConfig
    real: RolloutResult | None  # None if no real.parquet yet
    real_path: str | None
    base_dir: str               # directory the files live in

def iter_flat_recordings(recordings_dir: str, *, require_real: bool = True):
    """Flat layout: trajectory_<ts>.json (+ real_<ts>.parquet) from collect_dataset.py."""
    for traj_path in sorted(glob.glob(os.path.join(recordings_dir, "trajectory_*.json"))):
        ts = os.path.basename(traj_path)[len("trajectory_"):-len(".json")]
        real_path = os.path.join(recordings_dir, f"real_{ts}.parquet")
        has_real = os.path.exists(real_path)
        if require_real and not has_real:
            continue
        cfg = TrajectoryConfig.load(traj_path)
        real = RolloutResult.from_dataframe(pd.read_parquet(real_path)) if has_real else None
        yield Recording(ts, cfg, real, real_path if has_real else None, recordings_dir)

def iter_structured_recordings(rollouts_dir: str, *, require_real: bool = True):
    """Structured layout: <traj_id>/trajectory.json (+ real.parquet) from record_rollouts.py."""
    store = RolloutStore(rollouts_dir)
    for traj_id in store.list_traj_ids():
        has_real = store.has_real(traj_id)
        if require_real and not has_real:
            continue
        cfg = store.load_trajectory(traj_id)
        real = store.load_real(traj_id) if has_real else None
        yield Recording(traj_id, cfg, real, None, os.path.join(store.base_dir, traj_id))
```

> `run_calibration.py`'s `_load_rollouts_flat` / `_load_rollouts` should be **refactored to
> call these** (return the same `(real, config)` pairs + ids). See `R2_04` Task 7.

## 2. New script — `scripts/simulate_recordings.py`

### Behavior
For each recording in the chosen folder: rebuild the baked trajectory, run each requested
backend with the given params, and save `<backend>_sim_<id>.parquet` next to the recording.
If `--compare` and a `real_*.parquet` exists, run `compare()` and accumulate a summary.

### CLI
```
python scripts/simulate_recordings.py \
    --recordings-dir data/recordings/session_01 \      # flat (collect_dataset.py) — OR
    --rollouts-dir   data/rollouts \                    # structured (record_rollouts.py)
    --backends newton mujoco \                          # default: both
    --params config/settings.yaml \                     # model params; default settings.yaml
    --time-step 0.01 \                                  # default from simulation.time_step
    --cut-off-time 0.0 \                                # for --compare only (see F3); default 0
    --noise \                                           # off by default
    --compare \                                         # also compute sim-vs-real error
    --summary-csv data/recordings/session_01/sim_summary.csv   # optional, with --compare
```
- Exactly one of `--recordings-dir` / `--rollouts-dir` is required (error if both/neither).
- `--params` accepts `settings.yaml` or a `calibrated_<backend>.yaml`
  (`RobotParams.from_yaml(path)`), so the same script verifies a calibration result.

### Output naming
- Flat layout: `<backend>_sim_<ts>.parquet` next to `real_<ts>.parquet`.
  (Mirrors run_calibration's `<backend>_calibrated_<ts>.parquet`; `_sim_` = pre-calibration
  / arbitrary params.)
- Structured layout: reuse `RolloutStore.save_sim(traj_id, rollout, backend=backend)` →
  `<backend>.parquet` in the `<traj_id>/` dir.

### Reference implementation (pseudocode)
```python
# bootstrap sys.path like the other scripts (src + scripts)
import argparse, os, sys, yaml, numpy as np, pandas as pd
from elastic_sim.params import RobotParams
from elastic_sim.rollout import RolloutStore
from elastic_sim.recordings import iter_flat_recordings, iter_structured_recordings
from elastic_sim.trajectory import _trajectory_from_config
from elastic_sim.compare import compare

def _build_runner(backend, params, time_step):
    if backend == "newton":
        from elastic_sim.sim_runner import build_model, run_rollout
        model, dof_map, _ = build_model(params)
        return lambda traj, noise, cot: run_rollout(model, dof_map, traj,
                    noise=noise, cut_off_time=cot, time_step=time_step)
    else:
        from elastic_sim.mujoco_runner import build_model, run_rollout
        model, data, dof_map, act_map = build_model(params, time_step=time_step)
        return lambda traj, noise, cot: run_rollout(model, data, dof_map, act_map, traj,
                    noise=noise, cut_off_time=cot)

def main():
    args = _parse_args()
    settings = yaml.safe_load(open(args.params or "config/settings.yaml"))
    time_step = args.time_step or settings.get("simulation", {}).get("time_step", 0.01)
    params = RobotParams.from_yaml(args.params)   # default settings.yaml inside

    if args.recordings_dir:
        recs = list(iter_flat_recordings(args.recordings_dir, require_real=False))
        flat = True
    else:
        recs = list(iter_structured_recordings(args.rollouts_dir, require_real=False))
        flat = False
    if not recs:
        print("ERROR: no recordings found."); sys.exit(1)

    summary_rows = []
    for backend in args.backends:
        runner = _build_runner(backend, params, time_step)   # build model ONCE per backend
        for rec in recs:
            traj = _trajectory_from_config(rec.traj_config)
            sim = runner(traj, args.noise, 0.0)              # keep full record (F3)
            # save
            if flat:
                out = os.path.join(rec.base_dir, f"{backend}_sim_{rec.rec_id}.parquet")
                sim.to_dataframe().to_parquet(out, index=False)
            else:
                RolloutStore(os.path.dirname(rec.base_dir)).save_sim(rec.rec_id, sim, backend=backend)
            print(f"[{backend}] {rec.rec_id}: saved")
            # compare
            if args.compare and rec.real is not None:
                res = compare(sim, rec.real, cut_off_time=args.cut_off_time)
                phys = res.get("per_signal_phys", {})        # added in R2_03
                print(f"    metric={res['metric']:.4f}  pos={phys.get('position'):.4g}m "
                      f"vel={phys.get('velocity'):.4g}m/s force={phys.get('force'):.4g}N")
                summary_rows.append({"backend": backend, "id": rec.rec_id,
                                     "metric": res["metric"], **{f"phys_{k}": v for k,v in phys.items()}})
    if args.compare and args.summary_csv and summary_rows:
        pd.DataFrame(summary_rows).to_csv(args.summary_csv, index=False)
        print(f"Summary written: {args.summary_csv}")
```

### Notes
- Build each backend model **once**, reuse across all recordings (don't rebuild per file).
- Keep the full sim record (`cut_off_time=0`) and let `compare()` trim — addresses F3 and
  lets bias-baselining use the pre-motion window (`R2_02`).
- The script must not require `--compare`; without it, it just produces sim parquet files
  (useful when real recordings aren't present yet, e.g. sim-only studies).

## 3. Acceptance
- Running with `--recordings-dir` on a folder of N recordings writes N `<backend>_sim_<ts>.parquet`
  files per backend, each with the full Round-1 schema.
- With `--compare`, prints per-recording physical errors and (if requested) a summary CSV.
- Re-runnable and idempotent (overwrites its own `_sim_` outputs; never touches
  `real_*.parquet` or `trajectory_*.json`).
