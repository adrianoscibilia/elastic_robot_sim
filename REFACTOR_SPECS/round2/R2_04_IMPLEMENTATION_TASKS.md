# R2_04 — Implementation Tasks (ordered, atomic)

Execute in order. After each edited `.py`, run
`python -c "import ast; ast.parse(open(PATH).read())"`. Tests are in `R2_05`.
Cross-refs: `R2_01` (batch), `R2_02` (force), `R2_03` (stopping).

---

## TASK 1 — `config/calibration.yaml`: add `force:` and `performance_requirements:`

▶ ADD the `force:` block (`R2_02 §2`) and `performance_requirements:` block (`R2_03 §2`).
**Acceptance:** `yaml.safe_load` succeeds; `cal['performance_requirements']['enabled'] is True`.

---

## TASK 2 — `src/elastic_sim/compare.py`: physical errors + force preprocessing + single cut-off

▶ ADD `force_opts: dict | None = None` and `metadata: dict | None = None` params to
`compare(...)`.
▶ RESTRUCTURE the grid (R2_02 §4): build the **full** overlap grid (no cut-off), resample
both, compute the force baseline/sign on the full grid, then build a boolean `mask` for
`grid >= cut_off_time` and apply it to **all** signals' RMSE.
▶ ADD `_preprocess_force(...)` (R2_02 §3); use the returned `real_tau` (masked) in the force
loop instead of `r.tau_link`.
▶ ADD `per_signal_phys` (worst-axis raw RMSE, m / m/s / N) and `per_signal_phys_axis` to the
result dict (R2_03 §1). Keep `metric`, `per_signal`, `per_axis` unchanged.
**Acceptance:** existing `metric` for `force_opts=None, cut_off_time=0` matches pre-change
(within 1e-9); new keys present. Force-bias and sign tests in `R2_05` pass.

---

## TASK 3 — `src/elastic_sim/calibration.py`: thread force_opts, track best phys, requirement_met

▶ ADD `force_opts: dict | None = None` to `SimCalibrationProblem.__init__`; store it; pass it
(and each real rollout's `metadata`) into every `compare(...)` call in `loss()` and anywhere
else compare is used.
▶ In `loss()`, aggregate `per_signal_phys` across rollouts (mean) and maintain `self._best_loss`
+ `self._best_phys` (R2_03 §4). Initialise `self._best_loss = inf`, `self._best_phys = None`
in `__init__`.
▶ ADD methods `requirement_met(thresholds)` and `best_phys()` (R2_03 §4).
**Acceptance:** `problem.best_phys()` returns a dict after ≥1 eval; `requirement_met` returns
True only when all gated signals ≤ thresholds.

---

## TASK 4 — `src/elastic_sim/optimizers/base.py`: add `should_stop` to the interface

▶ ADD `should_stop: Callable[[], bool] | None = None` to the abstract `minimize` signature and
document it (stop early when it returns True; checked after each eval).
**Acceptance:** abstract signature updated; subclasses override consistently (Tasks 5–7).

---

## TASK 5 — `cma_backend.py`, `bo_backend.py`, `skrl_backend.py`: honor `should_stop`

▶ Each `minimize` accepts `should_stop=None` and, after each objective evaluation (post best
update), calls it; if True, break and return `(best_theta, history)`.
- CMA: inside the ask/tell loop, after evaluating each candidate (or after each generation),
  check and break.
- BO: after each evaluated point, including the initial design.
- skrl: between iterations at minimum.
▶ Preserve current behavior when `should_stop is None`.
**Acceptance:** a stub objective + `should_stop` that returns True after k evals stops at
~k evals for each backend (CMA/BO at least; skrl best-effort between iterations).

---

## TASK 6 — `src/elastic_sim/recordings.py` (NEW): shared loaders

▶ CREATE the module with `Recording`, `iter_flat_recordings`, `iter_structured_recordings`
(R2_01 §1).
**Acceptance:** importing and iterating over a sample flat dir yields `Recording`s with
`traj_config` set and `real` populated when the parquet exists.

---

## TASK 7 — `scripts/run_calibration.py`: use shared loaders + wire stopping/force

▶ REPLACE `_load_rollouts_flat` / `_load_rollouts` bodies with calls to
`elastic_sim.recordings` (keep returning `(real, config)` pairs + ids/timestamps so the rest
of the file is unchanged).
▶ READ `force` block → `force_opts`; pass into `SimCalibrationProblem(...)`.
▶ BUILD `thresholds` from `performance_requirements`; build `should_stop`; pass to
`optimizer.minimize(...)` (R2_03 §5).
▶ Pass `cut_off_time=0` into the sim runs and let `compare` apply the cut-off (so force
baselining has the transient) — i.e. construct `SimCalibrationProblem(..., cut_off_time=0)`
**but** pass the configured `cut_off_time` into `compare` via the problem. (Add a separate
`compare_cut_off_time` field to the problem, or keep `cut_off_time` as the compare value and
pass `cut_off_time=0` only to `run_rollout`. Pick one; document it. Recommended: problem
stores `compare_cut_off` and always runs sim with `cut_off_time=0`.)
▶ ADD the per-signal PASS/FAIL report and non-zero exit on unmet requirement (R2_03 §5–§7).
▶ EXTEND `_validate` to aggregate `per_signal_phys` and print PASS/FAIL if
`evaluate_on in ('validation','both')`.
**Acceptance:** end-to-end calibration prints the requirement report; achievable thresholds
stop early; impossible thresholds exit non-zero after saving best params.

---

## TASK 8 — `scripts/simulate_recordings.py` (NEW): batch simulate

▶ CREATE the script per `R2_01 §2` (CLI, build-model-once-per-backend, save `<backend>_sim_<id>.parquet`,
optional `--compare` + `--summary-csv`). Use `elastic_sim.recordings` loaders and
`compare()` (with `force_opts` read from `calibration.yaml` if `--compare`).
▶ Keep full sim record (`cut_off_time=0`); pass `--cut-off-time` only to `compare`.
**Acceptance:** see `R2_01 §3` and `R2_05`.

---

## TASK 9 (OPTIONAL — record-side bias) — `real_robot/record_real_rollout.py`

▶ Before `send_trajectory`, optionally spin for `bias_window_s[1]` seconds collecting F/T at
rest; store the mean as `metadata["ft_bias_xyz"]` in the saved `RolloutResult`. Guarded by a
CLI flag / config (`force.record_bias: true`). `compare()` already prefers a recorded bias.
**Acceptance:** when enabled, `real_*.parquet` metadata carries `ft_bias_xyz`; compare uses it.

---

## TASK 10 (OPTIONAL — needs user CONFIRM, Finding F2) — position observability

▶ DO NOT implement without confirmation. If approved: in `compare()` compute the position and
velocity terms (both metric and phys) on `q_motor`/`dq_motor` (observable on both sides)
instead of `q_link`/`dq_link`, leaving the force term to carry elasticity. Add a config flag
`compare.position_signal: motor|link` (default `motor` if adopted). Document the rationale
inline.
**Acceptance:** flag switches the compared signal; default preserves current behavior until
the user opts in.

---

## TASK 11 — Final sweep

▶ Confirm no new magic numbers: all thresholds/options come from `calibration.yaml`
(`grep -rn` for hard-coded mm/N targets returns none in `.py`).
▶ Confirm `compare()`'s new keys don't break `run_calibration._validate` or any other caller
(`grep -rn "compare(" --include=*.py src scripts`).
▶ Confirm raw `real_*.parquet` is never written by compare/calibration/simulate paths.
**Acceptance:** R2-INV-1…4 hold; proceed to `R2_05`.

---

## Signature-change summary

| Function | Change |
|----------|--------|
| `compare(sim, real, weights=None, *, cut_off_time=0.0)` | + `force_opts=None`, `metadata=None`; returns extra `per_signal_phys`, `per_signal_phys_axis` |
| `SimCalibrationProblem.__init__` | + `force_opts=None`; tracks `_best_phys`; new `requirement_met`, `best_phys` |
| `Optimizer.minimize(...)` (all backends) | + `should_stop=None` |
| `run_calibration.py` loaders | now delegate to `elastic_sim.recordings` |

New files: `src/elastic_sim/recordings.py`, `scripts/simulate_recordings.py`.
