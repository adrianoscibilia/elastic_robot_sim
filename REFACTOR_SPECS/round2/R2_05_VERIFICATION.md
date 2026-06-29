# R2_05 — Verification (round 2)

Add to `tests/`. Run `pytest tests/ -q`. Tests that need a simulator backend
(newton/mujoco) or ROS should `pytest.importorskip(...)` so the suite still runs without
heavy deps.

## A. compare(): physical errors present and consistent

`tests/test_compare_phys.py`
```python
import numpy as np
from elastic_sim.rollout import RolloutResult
from elastic_sim.compare import compare

def _roll(t, q, dq, tau):
    z = np.zeros_like(q)
    return RolloutResult(time=t, ref_pos=q, ref_vel=dq, q_motor=q, q_link=q,
                         dq_motor=dq, dq_link=dq, tau_motor=tau, tau_link=tau, metadata={})

def test_phys_keys_and_zero_error():
    t = np.linspace(0, 5, 200)
    q = np.column_stack([np.sin(t), np.cos(t), 0.5*np.sin(t)])
    dq = np.gradient(q, t, axis=0)
    tau = 10*q
    r = _roll(t, q, dq, tau)
    res = compare(r, r, cut_off_time=0.0)
    assert {"position","velocity","force"} <= set(res["per_signal_phys"])
    assert res["per_signal_phys"]["position"] < 1e-6   # identical → ~0
```

## B. Force bias removal cancels a constant offset

`tests/test_force_bias.py`
```python
import numpy as np
from elastic_sim.compare import compare
# build sim and real identical except real force has +7 N constant offset on all axes.
# with force_opts={"remove_bias":True,"bias_window_s":[0,1]} the force phys error ≈ 0;
# with remove_bias False it ≈ 7.
```
Assert `phys_force_with_bias_removal < 0.1` and `phys_force_raw > 5`.

## C. Force sign alignment

`tests/test_force_sign.py`
```python
# real force = -1 * sim force (active axes). 
# sign="auto" → small force error; sign=+1 → large.
# also: an axis with span < sign_min_activity_n is NOT flipped.
```

## D. Per-axis aggregation = worst axis

`tests/test_phys_worstaxis.py` — give one axis a large error and others zero; assert
`per_signal_phys["position"]` equals that axis's RMSE (max, not mean).

## E. requirement_met logic (no sim needed)

`tests/test_requirement_met.py`
```python
# monkeypatch a SimCalibrationProblem-like object: set _best_phys manually and check
# requirement_met returns True only when ALL gated signals <= thresholds, and that
# a None threshold is ignored.
```
Construct a minimal object with `_best_phys` + the `requirement_met` method (or instantiate
the real class with a tiny synthetic rollout list and a stubbed `_run_sim`).

## F. Optimizer early stop honors should_stop

`tests/test_should_stop.py`
```python
import numpy as np
from elastic_sim.optimizers.cma_backend import CMAOptimizer
def test_cma_stops_early():
    calls = {"n": 0}
    def obj(x):
        calls["n"] += 1
        return float(np.sum(x**2))
    stop = lambda: calls["n"] >= 5
    opt = CMAOptimizer(sigma0=0.3, max_evals=1000)
    best, hist = opt.minimize(obj, [(-1,1)]*3, x0=np.zeros(3), max_evals=1000, should_stop=stop)
    assert len(hist) < 50         # stopped far before budget
```
Repeat for BO (`importorskip` its dep). skrl: best-effort (`importorskip`).

## G. recordings loaders

`tests/test_recordings.py` — write a temp flat dir with one `trajectory_<ts>.json` (via
`make_ptp_trajectory(...).config.save`) + a synthetic `real_<ts>.parquet`
(`RolloutResult.to_dataframe().to_parquet`); assert `iter_flat_recordings` yields one
`Recording` with `traj_config.mode==2` and `real is not None`. `importorskip("pyarrow")`.

## H. simulate_recordings.py smoke (sim backend available)

`tests/test_simulate_recordings.py` (`importorskip` newton or mujoco):
- Build a temp flat dir with 2 recordings (trajectory json + synthetic real parquet).
- Invoke the script's `main()` (or a factored `run(...)`) with `--recordings-dir tmp
  --backends mujoco --compare --summary-csv tmp/s.csv`.
- Assert 2 `mujoco_sim_<ts>.parquet` files exist with the full 25-column schema, the summary
  CSV has 2 rows with `phys_position/velocity/force` columns, and the originals are untouched.

## I. End-to-end calibration stop (sim backend available)

`tests/test_calibration_stop.py` (`importorskip`):
- 1–2 synthetic recordings. Run a short calibration with very loose thresholds → assert it
  reports `requirement MET` and stops before `max_evals`.
- Run with impossible thresholds (`force_rmse_n: 0`) and `max_evals` small → assert it
  reports `NOT met` and the process would exit non-zero (call the inner function and check
  its return/raised SystemExit code == 2).

## J. Non-regression: raw parquet untouched

Hash a `real_*.parquet` before and after a compare/calibration/simulate run; assert
identical bytes.

---

## Manual / hardware checks (document in PR)
- `python scripts/simulate_recordings.py --recordings-dir data/recordings/<session> --backends newton mujoco --compare`
  → one `<backend>_sim_<ts>.parquet` per recording per backend + printed per-signal errors.
- `python scripts/run_calibration.py --recordings-dir data/recordings/<session>`
  → prints per-signal PASS/FAIL and `requirement MET/NOT met`; writes `calibrated_<backend>.yaml`.
- Confirm the force block behaves: toggle `force.remove_bias` and observe the force error and
  resulting calibrated stiffness change sensibly.

## Definition of done
- [ ] `pytest tests/ -q` green (A–J, with importorskip where needed).
- [ ] R2-INV-1…4 hold.
- [ ] `simulate_recordings.py` and the calibration report work on a real session folder.
- [ ] All new thresholds/options live in `calibration.yaml`; no hard-coded targets in `.py`.
- [ ] Findings F1 (force) and F4 (phys reporting) resolved; F2 left as opt-in pending user
      confirmation; F3 folded into the single-cut-off restructure; F5 (shared loaders) done.
