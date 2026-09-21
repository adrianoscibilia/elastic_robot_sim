# R3_10 — Review of implementation 1 (`a4b34bd`)

> **Audience:** the implementing agent and the project owner. This reviews
> `R3_09_IMPLEMENTATION_REPORT.md` against the **code at `a4b34bd`**, not
> against the report. Every claim below was checked by reading the code and,
> where no simulator is needed, by running it (numpy/scipy only; Pinocchio
> and MuJoCo were not installable in the review environment either, so
> nothing physics-dependent was executed by the reviewer).
>
> Status legend: **CLOSE** — done and verified · **CLOSE*** — done, with a
> small follow-up listed · **PENDING** — code done, closes once the named
> test has actually run · **REOPEN** — defect found, fix required · **OPEN**
> — not done.

## 0. Verdict

The report is accurate in almost everything it says, and it is candid about
what did not run. The sampling, split, payload, storage and multiprocessing
work is solid, and two of the agent's deviations from the spec are
**corrections of real flaws in the spec** (§4). The two bugs it caught in the
probe (shape mismatch, rest condition after two-block scaling) were real;
the rest condition now holds to ~1e-15 (re-verified).

But **the round's highest-severity finding, F3 (excitation never reaches the
transmission), is not fixed**, for two independent reasons found in review:

1. **The probe is scaled to zero on the joints it exists for** (§2.1). At the
   shipped defaults, A1, A2 and A4 get exactly zero probe amplitude in 100 %
   of candidates; at high regime accelerations every joint does.
2. **The reference CLI command drops the probe entirely** (§2.2):
   `generate_identification_dataset.py` rebuilds `FourierExcitationConfig`
   from scratch and never passes `probe_harmonics` — nor `centre_jitter`,
   which means the `d34a452` "enhanced variability" change has *never*
   reached a dataset built from the command line.

And the consumer-side work (Stage A1/A2) is **not in any repository this
review can see** (§2.3).

None of the physics-dependent tests have ever executed, anywhere. That is
still the gating item for the round.

## 1. Task status

| Task | Status | Note |
|---|---|---|
| **A1** consumer `ft0..ft{dof-1}` branch | **OPEN (not delivered)** | Not on `origin/master` (`45c5078`) nor in the owner's working copy. See §2.3 |
| **A2** `declared_split_loaders` | **OPEN (not delivered)** | Same |
| **A3** `SplitPolicy`, `split` column, manifest block | **CLOSE*** | §3.1: held-out robots are extreme only in *mean* stiffness; `holdout_trajectories` accepted but silently a no-op; zero-train config accepted |
| **A4** contract sidecar + uniform-step assertion | **CLOSE** | Assertion lives in `generate()` rather than `write_dataset()` — equivalent, fine |
| **A5** docs | **REOPEN** | `docs/IDENTIFICATION_DATASET.md` again claims a consumer fix that is not in the consumer repo. Same error as the one R3_05 was written to correct |
| **B1** stratified / Sobol | **CLOSE** | Re-measured with shipped YAML: stiffness coverage 0.930–0.995, rotor 0.935–0.992 |
| **B2** `nominal × factor`, legacy mapping | **CLOSE** | `common_fraction: 0.0` deviation accepted — the spec was wrong, §4.1 |
| **B3** rotor-inertia sampling | **CLOSE*** | Code correct; the "step does not fall >10 %" test was not written |
| **B4** coverage report + warning | **CLOSE*** | Code present; no test asserts the manifest block |
| **B5** `--manifest` guard | **CLOSE** | |
| **C1** `payload.py` | **CLOSE*** | §3.2: temp URDFs not git-ignored; offset is in `iiwa_link_7` frame, not the flange frame; payload has no collision geometry |
| **C2** `PayloadSampling`, stream `(seed, 2)` | **CLOSE** | |
| **C3** payload through `run_condition` | **CLOSE*** | Done in `generate()`; **not** in `run_identification_simulation.py` (acknowledged) |
| **C4** payload columns, `defl*` | **CLOSE*** | Manifest `robots[]` damping/step are payload-free and disagree with per-bag records when a payload is fitted |
| C-acceptance: peak `|tau|` < 80 % effort | **OPEN** | Not implemented anywhere, no test. §3.3 — matters at 6 kg × 8 rad/s² |
| **D1** explicit harmonic indices | **CLOSE** | |
| **D2** modal probe | **REOPEN** | §2.1 and §2.2 — the probe is inert in practice |
| **D3** regime randomization | **CLOSE*** | Works (realized peak-acc CV 0.26–0.36 vs 0.13–0.19 off). Test checks sampled config values, not realized trajectories. Digest-reproduction test missing and the CLI path currently *does* disagree (§2.2) |
| **D4** YAML probe on | **CLOSE** | Ineffective until D2 is fixed |
| **E1** `PARAMETER_PROVENANCE.md` | **CLOSE*** | §3.4: two rows over-claim class `P` |
| **E2** provenance letters | **CLOSE** | |
| **E3** `report_parameter_bounds.py` | **PENDING** | Never run end-to-end (needs Pinocchio) |
| **E4** `run_range_sensitivity.py` | **CLOSE** | Printing a Python snippet instead of fake CLI flags is the right call |
| **E5** docs | **CLOSE** | Except the consumer paragraph (A5) |
| **F1** index hoisting | **PENDING** | Reads correct; bit-identical test not run |
| **F2** preallocate / direct-step recording | **OPEN (deferred)** | Agree with deferring; §4.2 downgrades its urgency |
| **F3** `control_decimation` | **PENDING** | Design right (label never held); default 1 is correct until measured |
| **F4** `--jobs N` | **PENDING** | Pickling verified; bit-identical test not run |
| **F5** Parquet / sidecar / `%.7g` | **CLOSE*** | Keep `t` float64 in Parquet (§3.5) |

**Closeable now: 19 (of which 9 with a small follow-up). Pending a real test run: 4. Reopened: 2. Open: 4.**

---

## 2. Blocking defects

### 2.1 The probe is zeroed on every joint whose main trajectory is position- or velocity-limited

`_fit_to_limits` scales the main harmonics until the **tightest** of
position / velocity / `(1 − f)·max_acceleration` binds, then gives the probe
whatever is *left* of each budget:

```python
remaining_half_span = np.maximum(half_span - 0.5 * (q_main.max(0) - q_main.min(0)), 0.0)
remaining_velocity  = np.maximum(velocity_limit - np.abs(dq_main).max(0), 0.0)
a_probe, b_probe = _scale_block(..., remaining_half_span, remaining_velocity, f * max_acceleration)
```

On any joint where position or velocity is the binding limit, that remaining
budget is **exactly zero**, `_ratio(0, peak) = 0`, and the probe's scale is
zero. At 0.1 Hz base frequency with 5 harmonics, the heavy proximal joints are
almost always position- or velocity-limited — which are precisely the joints
whose 50–170 Hz modes the probe was designed to excite.

Measured over 200 candidates per setting (`sample_candidate`, shipped
YAML values):

| `max_acceleration`, `velocity_fraction` | Fraction of candidates with a non-zero probe, A1..A7 |
|---|---|
| 4.0, 0.75 (**shipped base**) | `0.00 0.00 0.06 0.00 0.66 0.71 0.66` |
| 2.0, 0.5 | `0.26 0.30 0.67 0.06 0.96 0.96 0.96` |
| 1.0, 0.3 | `0.70 0.68 0.90 0.35 0.99 1.00 0.99` |
| 8.0, 0.9 (top of regime range) | `0.00 0.00 0.00 0.00 0.00 0.00 0.00` |

So with regime randomization on, the probe is present on the proximal joints
only in low-acceleration trajectories, and absent on all joints in
high-acceleration ones.

**Fix.** Size the probe first, from its acceleration budget alone, then scale
the main harmonics against *what the probe leaves*, not the other way round.
The probe's position and velocity footprints are tiny (at 40 Hz and
0.8 rad/s²: ~1.3e-5 rad and ~3e-3 rad/s), so reserving them costs the main
trajectory nothing measurable:

```python
# 1. probe against its own acceleration budget only (position/velocity unconstrained)
a_p, b_p = _scale_block(a[:, n_main:], b[:, n_main:], ..., half_span=np.full(n, np.inf),
                        velocity_limit=np.full(n, np.inf),
                        acceleration_budget=np.full(n, f * max_acceleration))
q_p, dq_p, ddq_p = evaluate_series(a_p, b_p, 0, omega, time, indices=probe_indices)
# 2. main against the remainder of all three budgets
a_m, b_m = _scale_block(a[:, :n_main], b[:, :n_main], ...,
                        half_span=half_span - 0.5 * np.ptp(q_p, axis=0),
                        velocity_limit=velocity_limit - np.abs(dq_p).max(0),
                        acceleration_budget=max_acceleration - np.abs(ddq_p).max(0))
```

Add an assertion (and a test) that the probe's peak acceleration is
`≥ 0.9 · f · max_acceleration` on **every** joint for every candidate, at the
shipped settings and at both ends of the regime range.

### 2.2 The dataset CLI silently drops `probe_harmonics` and `centre_jitter`

`scripts/generate_identification_dataset.py`, lines 102–110:

```python
excitation = FourierExcitationConfig(
    n_harmonics=..., base_frequency=..., n_periods=..., time_step=sample_step,
    limit_margin=excitation.limit_margin,
    max_acceleration=args.max_acceleration or excitation.max_acceleration,
    velocity_fraction=excitation.velocity_fraction,
)
```

It reconstructs the config field by field and omits `centre_jitter`,
`probe_harmonics` and `probe_acceleration_fraction`, so they fall back to the
dataclass defaults (`0.0`, `()`, `0.2`). Consequences:

- The command the owner actually runs produces **no probe** and **centre
  jitter 0**. The `d34a452` jitter change (coverage 0.6 → 0.8, "where most of
  the information in the dataset comes from") has never reached a CLI-built
  dataset; this predates round 3 (the same omission is in `d34a452`).
- `run_identification_simulation.py` uses `config.excitation` directly and
  *does* get the probe and jitter, so the debug script and the dataset script
  now generate **different trajectories for the same `--tier/--trajectory`**
  — exactly the regression `R3_02 §3.2` called the most likely one.
- The test suite never sees this because every test calls `generate()` on a
  `load_config()` result, bypassing the script.

**Fix.** Replace the reconstruction with
`dataclasses.replace(config.excitation, **overrides)` where `overrides` holds
only the CLI flags that were actually given, plus `time_step=sample_step`.
Same pattern in both scripts. Add a test that parses the CLI for the default
config (e.g. factor the arg-to-config step into a function) and asserts
`config.excitation == load_config(DEFAULT_CONFIG).excitation` when no
excitation flags are passed.

Also: when `excitation.regime.enabled`, `regime_excitation` overwrites
`max_acceleration` and `velocity_fraction`, so `--max-acceleration` is
silently ignored. Either error on the combination or apply the CLI value as
a regime-range override; do not ignore it.

### 2.3 The consumer changes are not in any reachable repository

The report places the consumer work in `C:\Users\WKS\Projects\dynamic_model_nn`.
In the owner's copy (`C:\Users\adria\Projects\dynamic_model_nn`) and on
`origin/master` (both at `45c5078`, 2026-09-08), `dataset.py` still has only
the fixed `range(6)` branch; `git diff --ignore-cr-at-eol` is empty, so the
many `M` entries in `git status` are line endings only. There is no
`declared_split_loaders`, no `n_target_channels`, no `model_workflow.py`
guard.

**Action.** Commit and push the consumer work from the WKS machine, then
re-check. Until then **A1, A2 and A5 stay open** and datasets built from this
branch will still lose `ft6` in training.

---

## 3. Smaller defects and follow-ups

### 3.1 Held-out robots are not extrapolation (A3)

`assign_splits` ranks robots by **mean** log stiffness across the seven
joints. With `stiffness_common_fraction: 0.0` the joints are independent, so
the mean concentrates near nominal (its spread shrinks by ≈√7), and the
"extreme" robots are extreme only on average. With the shipped config
(20 robots, seed 20260917), expressed as log₂ of the factor (range ±1):

- test robots' mean factors: −0.49, −0.48, +0.41, +0.48;
- the per-joint softest/stiffest robots: A1 both in **train**, A4 both in
  **train**, A2 stiffest in **train**, A3 stiffest in **train**, …

So the test set measures generalization to unseen *combinations* (still
valuable, and a real improvement over the contiguous split), but not the
"measured extrapolation margin" the docstring and `R3_05 §2.2` promise.

**Fix, either:**
- (preferred) generate test robots from a **separate outer shell** — train on
  `nominal × [1/1.6, 1.6]`, test on `nominal × ([1/2, 1/1.6] ∪ [1.6, 2])` per
  joint — which is what `run_range_sensitivity.py` S2/S3 already do and which
  makes "extrapolation" true by construction; or
- keep ranking but rename the policy/docstring to "held-out robots
  (combination generalization)" so the claim matches the data.

Also: `holdout_trajectories` validates but labels everything `train` —
raise `NotImplementedError` instead. And `--robots 6` with the shipped
`test_robots: 4, val_robots: 2` yields **zero training robots** without
complaint — require at least one train robot.

### 3.2 Payload details (C1)

- `payload_asset` writes `identification_payload_*.urdf` next to the real
  URDF inside `assets/`. It is unlinked in `finally`, but a killed process
  (or a crashed pool worker) leaves it in the repo tree. Add
  `identification_payload_*.urdf` to `.gitignore`.
- The offset is expressed in the frame of `iiwa_link_7` (the child of A7),
  not the flange frame the YAML comment implies; the flange sits further out
  along z. Either inject on the fixed end-effector link when one exists, or
  correct the comment and the `offset_z` range.
- The payload has no `<collision>`, and trajectories are validated
  payload-free with contacts disabled, so a 25 cm box can pass through the
  table or the arm. Add a box collision element to the injected link and
  validate trajectories against the payload-fitted asset (only the geometry
  check needs it; conditioning can stay payload-free as `R3_01 §2.6` says).

### 3.3 Effort limit not checked (C acceptance)

`R3_01 §2.1` and §4 require peak `|tau| < 0.8 · effort` on every bag. Nothing
checks it. With a 6 kg payload at up to 0.2 m, and regime accelerations up
to 8 rad/s² (twice the value the 0–6 kg range was sized against), this is
the most likely way a bag becomes physically meaningless. Record
`peak_torque_ratio` per bag in the manifest and warn (or drop and re-draw)
above 0.8.

### 3.4 Provenance over-claims (E1)

- A2 stiffness and A2 rotor inertia are class **P** citing the Disney
  measurement, which is for the **base joint only**. Label them **C** (same
  robot, different joint) — the committee will check this.
- The Disney `Jc = 1.03 kg m²` is a *controlled* motor inertia, i.e. the
  apparent inertia under that paper's torque controller, which may include
  inertia shaping. Before using it as the physical reflected rotor inertia,
  confirm from the paper whether it is the physical value; if not, the
  anchor class for rotor inertia at A1 is at best **C**.

### 3.5 Parquet (F5)

The float32 downcast applies to every float64 column, including `t`. At
10 s the float32 spacing is ~1e-6 s — fine today, but `t` is what the
uniform-step checks and the Savitzky-Golay filter depend on, and longer
trajectories degrade it. Exclude `t` from the downcast.

### 3.6 Missing tests the report names or the spec required

Not present in `tests/` although the report's §10 refers to one of them by
name:

- `test_damping_ratio_becomes_observable_with_the_probe` — **the** test that
  closes F3;
- `test_single_rollout_script_reproduces_the_dataset_trajectory` — would
  have caught §2.2;
- `test_payload_gives_the_distal_channels_signal` (+ effort check);
- `test_payload_relaxes_rather_than_tightens_the_integration_step`;
- `test_probe_does_not_spoil_the_regressor_condition`;
- rotor-inertia step test (B3), manifest coverage test (B4).

### 3.7 Environment

`pin==3.4.0` on Windows versus `pin==4.1.0` elsewhere means the two
machines would run **different Pinocchio major versions**; any tolerance-
level regression (the 1e-8 Nm rigid residual, the 1e-4 base-parameter
recovery, the 43/57 base-parameter counts) could differ between them.
Recommendation: run the suite under **WSL2 with `pin==4.1.0`** and treat
that as the reference; keep the Windows marker only as a convenience. The
`cmeel-octomap` patch living in a uv cache is correctly flagged as
non-portable.

---

## 4. Corrections to the spec itself

Two of the agent's findings are flaws in `R3_03` / `R3_08`, and a third
comes from this review. The spec should be read with these corrections.

### 4.1 `stiffness_common_fraction > 0` cannot reach full coverage (R3_03 §2.2)

The spec's correlated sampler adds a shared and a per-joint uniform in log
space. Their sum is triangular-shaped: reaching a joint's marginal extreme
requires *both* draws at the same extreme for the same robot. Reproduced:
at `common_fraction 0.5`, minimum coverage is 0.68 / 0.76 / 0.78 at
20 / 40 / 80 robots. The agent's choice of `0.0` is correct for the
coverage requirement. If correlated arms are wanted later, keep stratified
marginals and **induce** rank correlation (Iman–Conover reordering), which
preserves the marginals exactly.

### 4.2 F2's correctness argument was overstated (R3_06 §2.3)

`rollout_frame` interpolates from the ~16 kHz physics grid onto the 2 ms
grid. Linear interpolation between samples 62 µs apart is, for content up
to 160 Hz, indistinguishable from point sampling (error ∝ (ωΔt)²/8 ≈ 1e-4
relative). It does **not** low-pass the probe. F2 is therefore a
performance item. The genuine signal-integrity risk is the opposite one:
point sampling onto 2 ms with **no anti-alias filter**, so any ringing of
A7's 640–900 Hz mode folds into the 0–250 Hz band. Check the A7 deflection
spectrum once rollouts run; if it rings, decimate with an anti-alias filter
rather than `np.interp`.

### 4.3 A 5-line probe comb cannot satisfy its own acceptance test (R3_02, R3_08 §B2)

The probe is five discrete lines, 30 Hz apart (40…160 Hz). A linear system
responds only at the excitation frequencies, so the deflection spectrum
peaks at the nearest **probe line**, not at the mode. `R3_08 §B2`'s
"peak within 10 % of the mode" assertion is therefore unreachable for most
modes (the implemented test's uniform-k robot has a 78 Hz mode vs the 70 Hz
line: 10.2 %). Worse for F3: `zeta` only changes the response at lines
within about the half-power bandwidth `2ζf` of the mode (10 Hz at
ζ = 0.05, f = 100 Hz), so with lines 30 Hz apart most robots' `zeta` stays
nearly unobservable even once §2.1 is fixed.

**Revised probe spec:** 25–40 lines, log-spaced over 30–200 Hz (still
multiples of `base_frequency`, still below the 0.4/Δt alias guard), with
random phases so the peak acceleration stays within budget. **Revised
acceptance:** estimate the deflection-to-torque FRF at the probe lines and
assert its magnitude peak lies within one line spacing of the predicted
mode; and the `zeta` contrast test (§3.6) — two robots differing only in
`zeta` differ by > 1 % RMS in `ft` with the probe on, and < 0.1 % with it
off.

---

## 5. Next steps, in order

1. **Push the consumer work** from the WKS machine (§2.3).
2. **Fix §2.2** (CLI drops probe and jitter) — one function, plus the
   CLI-equals-config test. Smallest change with the largest effect: it also
   restores `centre_jitter` that CLI datasets have never had.
3. **Fix §2.1** (probe budget order) and **§4.3** (denser comb), then write
   the `zeta`-contrast test.
4. **Run the whole suite on Linux/WSL with `pin==4.1.0`.** Expect
   `test_probe_excites_the_transmission_mode` to fail as written until
   steps 3 and the revised acceptance are in; the base-parameter recovery
   test is the invariant that must pass untouched.
5. Effort check (§3.3), split semantics (§3.1), provenance labels (§3.4).
6. Then the Definition-of-Done build from `R3_07`.
