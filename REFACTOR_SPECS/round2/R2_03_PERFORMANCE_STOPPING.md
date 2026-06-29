# R2_03 — Step 4: iterate until a performance requirement is reached

Decision (user): the requirement is **physical per-signal thresholds**. The loop must stop
as soon as position, velocity, and force errors are each at/under their target (in real
units), and report pass/fail per signal. If the budget runs out first, report the best and
FAIL explicitly.

## 1. Physical per-signal errors — extend `compare()`

`compare()` already computes normalised RMSE. Add raw-unit RMSE aggregated per signal type
so thresholds are interpretable. Aggregate across axes with **max** (worst axis must pass)
— stricter and clearer than mean for an acceptance gate.

In `compare()`, after computing per-axis errors, also compute:

```python
# Raw-unit RMSE per axis (no normalisation), on the cut-off-masked grid
pos_axis = [ _rmse(s.q_link[m, i],  r_q[m, i])  for i in range(3) ]   # meters
vel_axis = [ _rmse(s.dq_link[m, i], r_dq[m, i]) for i in range(3) ]   # m/s
frc_axis = [ _rmse(s.tau_link[m, i], real_tau[m, i]) for i in range(3) ]  # N  (real_tau from R2_02)

per_signal_phys = {
    "position": float(np.max(pos_axis)),   # worst-axis RMSE (m)
    "velocity": float(np.max(vel_axis)),   # worst-axis RMSE (m/s)
    "force":    float(np.max(frc_axis)),   # worst-axis RMSE (N)
}
per_signal_phys_axis = {                   # keep per-axis for diagnostics
    "position": pos_axis, "velocity": vel_axis, "force": frc_axis,
}
```

Add both to the returned dict (alongside `metric`, `per_signal`, `per_axis`). Existing
callers keep working (new keys are additive).

> If F2's recommendation is adopted, position/velocity phys errors come from
> `q_motor`/`dq_motor` instead of `q_link`/`dq_link`. Keep the metric and the phys errors on
> the **same** signals so the gate matches the objective.

## 2. Config — `config/calibration.yaml` new `performance_requirements:` block

```yaml
performance_requirements:
  enabled: true
  # Worst-axis RMSE targets, real units. A signal with null is not gated.
  position_rmse_m: 0.005     # 5 mm
  velocity_rmse_ms: 0.02     # 2 cm/s
  force_rmse_n: 2.0          # 2 N
  evaluate_on: train         # train | validation | both  (which set gates the stop)
```

## 3. Optimizer interface — add `should_stop` callback

The optimizers currently stop only at `max_evals`. Add an optional early-stop hook to the
`Optimizer.minimize` signature in `optimizers/base.py` and implement it in all three
backends:

```python
def minimize(self, objective, bounds, x0=None, *, max_evals=200, verbose=False,
             should_stop: Callable[[], bool] | None = None):
    ...
```

- Call `should_stop()` **after each objective evaluation** (i.e., after the best-so-far is
  updated). If it returns `True`, stop and return `(best_theta, history)` immediately.
- **CMA** (`cma_backend.py`): check after each `tell` / each candidate eval; break the ask/
  tell loop when `should_stop()`.
- **BO** (`bo_backend.py`): check after each evaluated point (including initial design).
- **skrl** (`skrl_backend.py`): check at the end of each rollout/iteration. If awkward to
  interrupt, at minimum check between iterations.
- `should_stop=None` ⇒ behavior identical to today (no early stop).

> Implementation tip: the cleanest place to evaluate the requirement is a closure in
> `run_calibration.py` that reads `problem`'s best per-signal errors (next section), so the
> optimizers stay generic and only see a zero-arg `should_stop`.

## 4. `SimCalibrationProblem` — track best per-signal physical errors

Today `loss()` keeps `(theta, scalar)` history only. Extend it to remember the per-signal
physical errors of the **best** evaluation so the stop check is cheap.

```python
# in loss(): aggregate compare() results across rollouts
phys_list = [res["per_signal_phys"] for res in results]   # one per rollout
mean_phys = {k: float(np.mean([p[k] for p in phys_list])) for k in ("position","velocity","force")}
# store alongside the scalar; update best-phys when this eval is the new best
self._loss_history.append((theta.copy(), mean_loss))
if mean_loss <= self._best_loss:
    self._best_loss = mean_loss
    self._best_phys = mean_phys

def requirement_met(self, thresholds: dict) -> bool:
    """True if every gated signal's best-so-far phys error <= its threshold."""
    if not getattr(self, "_best_phys", None):
        return False
    for key, thr in thresholds.items():
        if thr is None:
            continue
        if self._best_phys.get(key, float("inf")) > thr:
            return False
    return True

def best_phys(self) -> dict | None:
    return getattr(self, "_best_phys", None)
```

Map yaml keys → signal keys: `position_rmse_m→position`, `velocity_rmse_ms→velocity`,
`force_rmse_n→force`.

> **Aggregation note:** `requirement_met` uses the **mean across rollouts** of the worst-axis
> phys error. Document this. (Worst-axis within a rollout, mean across rollouts — a
> reasonable acceptance gate; if the user later wants worst-across-all, change the mean to a
> max in one place.)

## 5. `run_calibration.py` — wire it together

1. Read `performance_requirements` from `cal`; build `thresholds = {"position":..., "velocity":..., "force":...}` (None where disabled / `enabled: false`).
2. Read `force` block; pass `force_opts` into `SimCalibrationProblem` (→ `compare`).
3. Build the stop callback and pass it to the optimizer:
   ```python
   reqs = cal.get("performance_requirements", {})
   thresholds = ( {"position": reqs.get("position_rmse_m"),
                   "velocity": reqs.get("velocity_rmse_ms"),
                   "force":    reqs.get("force_rmse_n")}
                  if reqs.get("enabled", False) else {} )
   should_stop = (lambda: problem.requirement_met(thresholds)) if thresholds else None
   best_theta, history = optimizer.minimize(problem.loss, bounds, x0=x0,
                                            max_evals=args.max_evals, verbose=args.verbose,
                                            should_stop=should_stop)
   ```
4. **Report** after optimisation:
   ```python
   phys = problem.best_phys() or {}
   print("\nPerformance requirement:")
   met_all = True
   for key, thr in thresholds.items():
       if thr is None:
           continue
       val = phys.get(key, float("inf"))
       ok = val <= thr; met_all &= ok
       print(f"  {key:8s}: {val:.4g} (target {thr:.4g})  {'PASS' if ok else 'FAIL'}")
   print(f"  => requirement {'MET' if met_all else 'NOT met'} "
         f"after {len(history)} evals")
   ```
5. Optionally also evaluate the requirement on the **validation** set if
   `evaluate_on in ('validation','both')` (reuse `_validate`, which already computes
   `compare` per val rollout — extend it to aggregate `per_signal_phys` and print PASS/FAIL).
6. **Exit code:** if `performance_requirements.enabled` and the requirement was **not** met,
   `sys.exit(2)` after saving the best params (so a batch driver can detect "needs more
   data / more evals"). Always save the best params first (a failed run is still useful).

## 6. (Out of scope, document only) Outer "iterate" beyond optimisation
The 4-step loop's "iterate" is the optimiser's eval loop with the threshold stop above.
A higher-level loop (collect more data → re-calibrate) is a workflow, not code here. If
desired later, a thin driver could: run calibration → if FAIL, prompt to collect more
recordings → repeat. Not implemented this round.

## 7. Acceptance
- With achievable thresholds, calibration stops **before** `max_evals` and prints
  `requirement MET`.
- With impossible thresholds (e.g. `force_rmse_n: 0.0`), it runs the full budget, prints
  per-signal `FAIL`, saves best params, and exits non-zero.
- `should_stop=None` path is unchanged from current behavior.
