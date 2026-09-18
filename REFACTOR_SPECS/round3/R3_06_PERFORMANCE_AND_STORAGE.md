# R3_06 — Generation cost and dataset storage

Addresses finding **F10**. This file is optional for correctness and mandatory for
scale: every change in `R3_01`-`R3_03` pushes toward more robots, and the current
cost model makes that expensive for no good reason.

---

## 1. Where the time goes

Measured structure of `run_mujoco_elastic_torque` (`torque_runners.py` ~764-796):

```python
for index, sample_time in enumerate(grid):          # grid step = 6.2e-5 s
    motor_q  = np.asarray([data.qpos[...] for n in names])     # 4 list comprehensions
    motor_dq = np.asarray([data.qvel[...] for n in names])     # over 7 names each,
    elastic_q  = ...                                           # building 7-vectors
    elastic_dq = ...
    ...
    command = controller(sample_time, motor_q, motor_dq, tau_spring)   # Pinocchio RNEA
    mujoco.mj_forward(model, data)                                     # extra forward
    rows[...].append(...)            # 9 Python list appends per step
    mujoco.mj_step(model, data)
```

For the reference command:

| | |
|---|---|
| Steps per rollout | `10 s / 6.2e-5 s` ≈ **161 000** |
| Rollouts | 60 |
| Total steps | **~9.7e6** |
| Python-side work per step | 4 list comprehensions x 7, one RNEA, one extra `mj_forward`, 9 list appends |

MuJoCo's own `mj_step` on a 14-DoF contact-free model is a few microseconds. Nothing
else here is. The rollout is **Python-bound and RNEA-bound, not physics-bound**.

## 2. Four cost reductions, in order of payoff

### 2.1 Decouple the control rate from the integration rate — biggest win, needs care

The integration step is small because it must resolve a **900 Hz transmission mode**.
The controller's closed-loop bandwidth is `control_frequency: 25 rad/s ≈ 4 Hz`.
Running a 4 Hz controller at 16 kHz is 4000x oversampling.

Add `simulation.control_decimation: N` — evaluate the controller every `N` physics
steps and hold the torque between updates (zero-order hold, as a real drive does).

**What this changes and what to check.** A ZOH torque is not the same signal as a
continuously updated one, and the dataset's headline property is that the recorded
label is *exactly* consistent with the dynamics. That property survives — `ft` is
the spring torque, measured from the state, not computed from the command — but two
things must be re-verified:

- the feedback/feedforward ratio stays under 5 % (the existing test);
- the rigid tier's residual against Pinocchio's inverse dynamics stays at ~1e-8 Nm
  (the existing test).

Start at `control_decimation: 8` (2 kHz control, still 500x the bandwidth) and raise
it until either test degrades. Expect a 3-6x rollout speedup. **Record the value in
the manifest** — it is a property of the generated data, not just of the run.

Note the interaction with `R3_02`: if the modal probe is fitted, the *reference*
contains content up to ~160 Hz, so the control rate must stay well above that.
`control_decimation: 8` gives 2 kHz, which is 12x the probe top — fine. Assert
`1 / (time_step * control_decimation) > 10 * probe_top_hz`.

### 2.2 Hoist the address lookups out of the loop

`addresses[name]["motor_qpos"]` is resolved 7 x 4 times per step through two dict
lookups. Build four **integer index arrays** once, before the loop, and use
fancy indexing:

```python
motor_qpos_idx   = np.array([addresses[n]["motor_qpos"] for n in names])
motor_dof_idx    = np.array([addresses[n]["motor_dof"] for n in names])
elastic_qpos_idx = np.array([addresses[n]["elastic_qpos"] for n in names])
elastic_dof_idx  = np.array([addresses[n]["elastic_dof"] for n in names])
...
motor_q  = data.qpos[motor_qpos_idx]
motor_dq = data.qvel[motor_dof_idx]
```

Pure refactor, no behaviour change, and it removes most of the per-step Python.
Apply the same in `run_mujoco_torque` and both Newton runners. Expect 1.5-2x.

### 2.3 Preallocate the output arrays

`rows` is nine Python lists growing to 161 000 entries of 7-vectors each. Preallocate
`np.empty((len(grid), n_dof))` per signal and write by index. Removes ~1.5e6 list
appends per rollout and the final `np.asarray` copies.

Better still: **only store what is written.** The output grid is 2 ms and the physics
grid is 62 us — 32x oversampled. Record every `k`-th step where
`k = round(sample_time_step / time_step)`, plus the endpoints, and drop the
resampling interpolation in `rollout_frame` entirely (or keep it as a fallback when
`k` is not an integer). This cuts the recorded arrays by ~30x **and** removes an
interpolation step that currently low-pass-filters the signal in a way nobody has
characterized. Given `R3_02` puts real content at 40-160 Hz, that interpolation
stops being harmless — this change matters for correctness there, not just speed.

### 2.4 Remove the duplicate extra `mj_forward`

The loop calls `mj_forward` after setting `qfrc_applied`, then `mj_step`.
`mj_step` begins with its own forward pass. Check whether the explicit `mj_forward`
is needed for `data.qacc` to be current at record time — if `ddq` is recorded
**before** the step, it is; if it can be recorded after, the call is redundant.
Verify against the existing MuJoCo/Pinocchio agreement test before removing it; do
not remove it on inspection alone.

### 2.5 Parallelize across bags

Bags are fully independent. `generate`'s loop is the natural place for a process
pool: each worker builds its own MuJoCo model and returns a DataFrame. Gate it on
`--jobs N` (default 1) so the visualization path and the determinism story are
untouched. On an 8-core machine this alone turns 40 minutes into ~6.

Caveats to spell out in the docstring: `--visualize` must force `jobs=1`; the RNG
streams are already keyed on `(seed, index)` so worker order does not affect
results; MuJoCo models are not picklable, so send the **config**, not the model.

---

## 3. Storage

### 3.1 What the 103 columns are

| Group | Count | Note |
|---|---|---|
| `t`, `bag` | 2 | |
| `q0..q6`, `dq0..dq6` | 14 | link-side state |
| `tau0..tau6` | 7 | motor torque, model input |
| `ft0..ft6` | 7 | target |
| `q_motor*`, `dq_motor*` | 14 | genuinely extra information |
| `q_link*`, `dq_link*` | 14 | **exact duplicates of `q*` / `dq*`** |
| `experiment` | 1 | **exact duplicate of `bag`** |
| `tier`, `backend` | 2 | |
| `viscous__*`, `coulomb__*` | 14 | per-row constants |
| `stiffness__*`, `damping__*`, `damping_ratio__*`, `rotor_inertia__*` | 28 | per-row constants |

At 300 060 rows: ~31 M fields, **~0.5-0.7 GB of CSV text**, of which
**15 columns are duplicates** and **42 are constants repeated 300 000 times**
(~55 % of the file is redundant).

### 3.2 Recommendations

1. **Keep the duplicates.** `q_link*` / `dq_link*` / `experiment` exist because the
   consumer's column matchers look for them; removing them is a cross-repo change
   for a modest saving. Document them as intentional aliases instead. *(If the
   consumer is being touched anyway for `R3_05`, reconsider — but do not couple the
   two changes.)*
2. **Move the per-row constants to a sidecar.** The 42 constant metadata columns are
   already in `<dataset>.manifest.json`, keyed by bag. Add
   `dataset.metadata_columns: inline | sidecar` (default `inline` for
   compatibility). With `sidecar`, the CSV keeps `bag`, `tier`, `backend`, `split`
   and the signals; everything else is looked up by `bag`. Saves ~40 % of the file.
3. **Offer Parquet.** `write_dataset` dispatches on the output suffix: `.csv` as
   today, `.parquet` via `frame.to_parquet`. Typed columns and compression take the
   same data to roughly 30-60 MB — a 10x reduction — and the consumer already
   depends on `pandas`. Add a `--format` flag and note in the docs that the consumer
   needs a Parquet branch in `CustomDataset` (one `elif` on the suffix) before it
   can be the default.
4. **Downcast to float32 on write.** The consumer casts to `float32` on load anyway
   (`to_numpy(dtype=np.float32)`). Writing float32 halves Parquet size and, for CSV,
   writing with `float_format="%.7g"` cuts the text size by ~40 % with no loss the
   consumer would see.

### 3.3 Scale check

After `R3_03`'s recommended `robots: 20`, `trajectories: 3` and `R3_01`'s payload:
still 60 bags, so nothing changes. But the natural next step is `robots: 50,
trajectories: 5` = 250 bags = 1.25 M rows ≈ **2.5 GB of CSV**. That is the point at
which Parquet plus the sidecar stops being optional. Implement §3.2 items 2-4
before raising the robot count past ~25.

---

## 4. Acceptance

- `control_decimation: 8` leaves the rigid-tier Pinocchio residual below 1e-7 Nm and
  the feedback ratio below 5 %.
- Index-array hoisting changes no recorded value by more than 0 (bit-identical).
- Direct per-`k`-step recording reproduces the interpolated output to within 1e-9
  rad on a trajectory with no probe, and **differs measurably** on a trajectory with
  the `R3_02` probe (which is the point).
- `--jobs 4` produces a dataset bit-identical to `--jobs 1`.
- `.parquet` output round-trips through `pandas.read_parquet` to the same frame.

Tests in `R3_08 §E`.
