# Round 2 — Batch simulation + calibration-stack review

> **Audience:** the Claude Code agent in VSCode. Implement mechanically.
> Read 00 → 05. Round 1 (`REFACTOR_SPECS/*.md`) is already implemented and verified;
> this round builds on it. Concrete edits live in `R2_04_IMPLEMENTATION_TASKS.md`.

## 1. The sim2real loop this round serves

The project goal, once real ground-truth recordings exist (Round 1 guarantees a baked
`trajectory_<ts>.json` + a fully-populated `real_<ts>.parquet`):

1. **Simulate the exact same trajectories** with a model loaded with default params.
2. **Read the error** between real recorded position / velocity / end-effector force and
   the simulated signals.
3. **Adjust model parameters** to minimize that error.
4. **Iterate** until a **performance requirement** is reached.

## 2. Review verdict (what already works vs what is missing)

| Step | Status | Where |
|------|--------|-------|
| 1 — simulate same trajectories | **OK** | `run_calibration.py` loads `(real, traj_config)` pairs and runs `_trajectory_from_config` on the baked config; sim duration uses `effective_sim_time`. Identical trajectory to the real run. |
| 2 — read error | **OK but improvable** | `compare()` resamples both onto a common grid and computes weighted normalised RMSE on `q_link`, `dq_link`, `tau_link`. See Findings F1–F3. |
| 3 — adjust params | **OK** | `SimCalibrationProblem.loss(theta)` + CMA / BO / skrl optimizers minimise the metric over stiffness (log-space), damping ratio, payload. |
| 4 — iterate to a requirement | **MISSING** | Optimizers stop at a fixed `max_evals`. There is no target / threshold / convergence stop and no pass/fail report. This round adds it (`R2_03`). |

**There is no convenience to "simulate every JSON in a folder" without calibrating.**
This round adds it (`R2_01`) — it is steps 1–2 in batch form and feeds calibration.

## 3. Findings (calibration stack review)

Severity: **[H]** correctness-affecting, **[M]** accuracy/robustness, **[L]** cleanup.

- **F1 [H] — Force physical consistency (real EE F/T vs sim elastic reaction).**
  `compare()` compares the real `tau_link` (raw `/ft_sensor/wrench` forces `fx,fy,fz`,
  see `record_real_rollout.py` ~203–213) directly against the sim `tau_link`
  (`qfrc_passive`, the elastic spring reaction). Two physical mismatches corrupt the loss:
  (a) **bias/offset** — a real F/T sensor has a non-zero resting reading (sensor bias +
  tool/gravity preload); the sim reaction is zero at rest. (b) **sign convention** — the
  sensor may report the force the environment exerts on the tool, opposite to the spring
  reaction. **Decision (user):** remove bias (baseline over a static pre-motion window) and
  verify/align sign; make it configurable. Spec in `R2_02`. This directly answers the
  project question *"do I have to remove bias? is the EE sensor force ok?"*.

- **F2 [M] — Position term observability.** On the real robot `q_link == q_motor`
  (the recorder cannot observe elastic deflection, `record_real_rollout.py` ~201–205),
  while the sim `q_link` includes deflection. So the position term compares
  *sim-elastic position* against *real-motor position*. This biases the optimizer toward
  **stiffening** the model (minimise apparent deflection) regardless of the true stiffness;
  the stiffness is really only constrained by the force term. **Recommended (CONFIRM with
  user before implementing):** compute the position/velocity error on `q_motor`/`dq_motor`
  (observable on both sides) and let the **force** term carry the elastic information.
  Tracked as an optional task in `R2_04` — do **not** implement silently.

- **F3 [L] — `cut_off_time` applied twice.** The sim `run_rollout` trims the first
  `cut_off_time` seconds *and* `compare()` sets `t_start = max(..., cut_off_time)`. Result
  is correct (real starts at 0, overlap window is `[cut_off, t_end]`), but the double
  handling is confusing and discards the transient that bias-baselining (F1) needs.
  **Recommended:** keep the full sim record (`cut_off_time=0` into `run_rollout`) and trim
  only in `compare()`. Low priority; bundle with the F1 work since baselining wants the
  pre-cut window.

- **F4 [M] — No per-signal physical reporting.** `compare()` returns normalised RMSE and a
  scalar metric only. Step 4's per-signal thresholds (m, m/s, N) need raw-unit errors.
  `R2_03` adds `per_signal_phys` to the `compare()` result.

- **F5 [L] — Duplicate recordings loaders.** `_load_rollouts_flat` /`_load_rollouts` live
  in `run_calibration.py`. The batch script (`R2_01`) needs the same logic. Extract into
  one shared helper (`src/elastic_sim/recordings.py`) and have both import it.

## 4. Locked decisions (do not re-litigate)

1. **Stopping criterion = physical per-signal thresholds** (position RMSE, velocity RMSE,
   force RMSE). Requirement met ⇔ every enabled signal is at/under its threshold (on the
   train set; also reported on validation). Optional plateau is *not* required this round.
2. **Force = bias-removed + sign-aligned**, configurable in `calibration.yaml`. Raw parquet
   is never mutated — preprocessing happens in the comparison path.
3. **Deliverable = MD specs only** (this set). No direct code edits to the repo this round.
4. **No new magic numbers** — all new thresholds/options live in `config/calibration.yaml`
   (carry over Round 1 INV-2 discipline).

## 5. Files in scope this round

| File | Change |
|------|--------|
| `scripts/simulate_recordings.py` | **NEW** — batch "simulate every JSON in a folder" (`R2_01`) |
| `src/elastic_sim/recordings.py` | **NEW** — shared recordings loader (`R2_01`, F5) |
| `src/elastic_sim/compare.py` | add `per_signal_phys`; add force bias/sign options (`R2_02`, `R2_03`) |
| `real_robot/record_real_rollout.py` | optional: record a static baseline window for bias (`R2_02`) |
| `src/elastic_sim/optimizers/base.py`, `cma_backend.py`, `bo_backend.py`, `skrl_backend.py` | add `should_stop` callback (`R2_03`) |
| `src/elastic_sim/calibration.py` | track best per-signal physical errors; `requirement_met()` (`R2_03`) |
| `scripts/run_calibration.py` | wire thresholds, early stop, pass/fail report (`R2_03`) |
| `config/calibration.yaml` | new `performance_requirements:` and `force:` blocks |
| `tests/` | new tests (`R2_05`) |

## 6. Invariants for this round

- **R2-INV-1:** `simulate_recordings.py` produces, for every `trajectory_*.json` with a
  matching `real_*.parquet`, one `<backend>_sim_<ts>.parquet` with the full Round-1 column
  schema, and (with `--compare`) a per-signal error summary.
- **R2-INV-2:** the simulated trajectory equals the real one (same baked config, same
  `effective_sim_time`) — carried over from Round 1.
- **R2-INV-3:** calibration stops as soon as the per-signal requirement is met and reports
  pass/fail per signal with physical-unit values; if never met within `max_evals`, it
  reports the best achieved and FAILS explicitly.
- **R2-INV-4:** force comparison uses bias-removed, sign-aligned signals when enabled; the
  on-disk parquet is unchanged.
