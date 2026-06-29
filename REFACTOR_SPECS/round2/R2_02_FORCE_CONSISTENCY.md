# R2_02 — Force physical consistency (bias removal + sign alignment)

Addresses Finding **F1 [H]** and the project question *"do I have to remove bias? is the
end-effector sensor force ok?"*. Decision (user): **remove bias + verify/align sign**,
configurable. The on-disk parquet is never mutated — preprocessing happens in the
comparison path so raw data stays intact.

## 1. The mismatch (why this matters)

| | Real (`tau_link`) | Sim (`tau_link`) |
|---|---|---|
| Source | raw `/ft_sensor/wrench` `fx,fy,fz` at the EE (`record_real_rollout.py` ~203–213) | elastic spring reaction `qfrc_passive` (`*_runner.py`) |
| Value at rest | non-zero (sensor bias + tool weight / preload) | ~0 |
| Sign | force environment→tool (sensor frame) | spring restoring force |

Comparing them raw injects a constant offset (bias) and possibly a global sign flip into the
force RMSE, which the optimizer then "fixes" by distorting stiffness/damping. Removing bias
and aligning sign makes the force term reflect true dynamic mismatch.

## 2. Config — `config/calibration.yaml` new `force:` block

```yaml
force:
  remove_bias: true          # subtract a baseline from the REAL force before comparing
  bias_window_s: [0.0, 2.0]  # [t0, t1] static pre-motion window for the baseline mean.
                             # Default = [0, cut_off_time]; the trajectory ramps from rest,
                             # so this window is quasi-static. Keep <= cut_off_time.
  sign: auto                 # auto | +1 | -1
                             #   auto: per-axis, flip real axis if it anti-correlates with sim
                             #   +1/-1: force a fixed global sign (use after you know the convention)
  sign_min_activity_n: 1.0   # auto-sign guard: only decide sign on an axis whose force
                             # spans > this (N); otherwise leave sign +1 (avoid noise-driven flips)
```

> **Recording-side option (recommended, separate from the loss):** also capture the
> baseline at acquisition. In `record_real_rollout.py`, before sending the trajectory,
> collect F/T for ~`bias_window_s[1]` seconds while the robot is still, store the mean in
> the parquet `metadata` (e.g. `ft_bias_xyz`). The comparison can then prefer the recorded
> bias over re-estimating from the window. This is optional; the in-`compare` estimation
> works on existing recordings. Spec'd as an optional task in `R2_04`.

## 3. `compare()` changes — bias + sign preprocessing

Add a `force_opts: dict | None = None` parameter to `compare()` (and thread it through
`SimCalibrationProblem`). Apply **only to the real force**, **before** resampling/RMSE.

```python
def _preprocess_force(real, sim, grid, *, force_opts, metadata):
    """Return real_tau (N,3) bias-removed and sign-aligned. Non-destructive."""
    real_tau = real.resample(grid).tau_link.copy()
    sim_tau  = sim.resample(grid).tau_link
    opts = force_opts or {}

    # 1) Bias removal -------------------------------------------------------
    if opts.get("remove_bias", True):
        recorded = (metadata or {}).get("ft_bias_xyz")   # prefer recorded baseline if present
        if recorded is not None:
            real_tau = real_tau - np.asarray(recorded).reshape(1, 3)
        else:
            t0, t1 = opts.get("bias_window_s", [0.0, 0.0])
            # window indices on `grid`; if empty (grid starts after t1), fall back to first sample
            mask = (grid >= t0) & (grid < t1)
            base = real_tau[mask].mean(axis=0) if mask.any() else real_tau[:1].mean(axis=0)
            real_tau = real_tau - base.reshape(1, 3)

    # 2) Sign alignment -----------------------------------------------------
    sign_cfg = opts.get("sign", "auto")
    if sign_cfg in (+1, -1, "+1", "-1"):
        real_tau = real_tau * int(sign_cfg)
    elif sign_cfg == "auto":
        guard = float(opts.get("sign_min_activity_n", 1.0))
        for i in range(3):
            span = real_tau[:, i].max() - real_tau[:, i].min()
            if span > guard:
                # flip axis if real anti-correlates with sim (dot < 0)
                if float(np.dot(real_tau[:, i], sim_tau[:, i])) < 0.0:
                    real_tau[:, i] = -real_tau[:, i]
    return real_tau
```

Then in `compare()`'s force loop, use the preprocessed `real_tau[:, i]` instead of
`r.tau_link[:, i]`. **Important caveats to encode as comments:**

- Bias removal applies to the **real** signal only (sim rest force is already ~0).
- The `bias_window_s` must lie inside the recorded data. Because Round-1 trajectories ramp
  from rest (`ramp = 1 - exp(-t/ramp_tau)`), `[0, cut_off_time]` is quasi-static and a good
  default baseline window. If `compare()` is called with `cut_off_time` trimming the grid
  before the window, the baseline can't be computed — that's why `simulate_recordings.py`
  and the calibrator should pass the **full** sim/real record into `compare()` and let
  `compare()` do the cut-off (Finding F3). Compute the baseline on the *untrimmed* real
  signal, then apply the cut-off for the RMSE window.
- Auto-sign is **per-axis** and guarded by `sign_min_activity_n` so a near-silent axis isn't
  flipped by noise. Once the true convention is known, set `sign: +1` or `-1` explicitly for
  determinism (auto-sign can in principle change between datasets).

## 4. Restructure: baseline before cut-off (ties F1 + F3 together)

`compare()` currently builds the grid with `t_start = max(..., cut_off_time)`. To support
bias-baselining over `[0, cut_off]`, split into two windows:

1. Build a **full** overlap grid from `t_start_full = max(sim.time[0], real.time[0])`
   (no cut-off) to `t_end`. Resample both onto it. Estimate the force baseline on this full
   grid (so the pre-motion window is available).
2. Apply `cut_off_time` as a **mask** on the full grid for the RMSE computation of all
   signals. (Equivalent result to today for position/velocity, but now the force baseline
   has access to the transient.)

This makes the cut-off single-sourced in `compare()`; callers pass `cut_off_time=0` into
`run_rollout` (see `R2_01` and `R2_04` Task 8 for the calibrator).

## 5. Acceptance
- With `remove_bias: true`, a real force with a constant offset added produces the **same**
  loss as without the offset (bias cancels).
- With `sign: auto`, a real force that is the exact negation of the sim force yields a small
  loss (sign corrected), while `sign: +1` on the same data yields a large loss.
- `sign_min_activity_n` prevents flipping an axis whose force span is below the guard.
- Raw `real_*.parquet` files are unchanged after a calibration/compare run.
