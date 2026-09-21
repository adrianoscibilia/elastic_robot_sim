# R3_09 — Implementation report

> **Audience:** the software architect who wrote `R3_00`-`R3_08`. This is a
> status report against that spec, not a spec itself: what landed, how, what
> was deliberately deviated from and why, what was skipped, and what is
> still unverified pending environment access.

## 0. Bottom line

All six stages (`R3_01`-`R3_06`) are implemented in both repositories, with
one deliberate exception (`R3_06`'s F2, see §6) and a handful of documented
deviations from the literal spec text (§5). The full test suite runs clean —
**153 passed, 0 regressions** — but roughly a third of the suite, including
the round's own non-negotiable acceptance test (base parameters recovered to
1e-4), could not be *executed* here: this Windows machine cannot build
Pinocchio from source (§7), which was discovered and partially fixed
mid-round but not fully resolved. Every physics-dependent test is written,
`pytest.importorskip`-gated, and ready to run the moment Pinocchio is
available (Linux, WSL, or a finished Windows build).

## 1. Scope actually covered

Both repositories named in `R3_00 §6`:

- `elastic_robot_sim` (producer) — all of Stages A-F.
- `C:\Users\WKS\Projects\dynamic_model_nn` (consumer) — confirmed present on
  this machine (the spec's `C:\Users\adria\...` path is a different
  developer's machine); `git status` was clean with no partial F8 fix, contrary
  to what `R3_05 §1.1` speculated might be sitting there.

Order followed `R3_07` exactly: A → B → C → D → E (parallel with B/C/D in
practice) → F, running the runnable subset of `tests/` after every stage.

## 2. Stage A — Splits and the consumer contract (`R3_05`)

**`dynamic_model_nn/dataset.py`** (merged into the one test file the repo's
own `.gitignore` tracks under `test/`, `test_dataset_generalization.py`,
after discovering a fresh file would have been silently untracked):

- Inserted the general `ft0..ft{dof-1}` branch before the fixed
  `range(6)` one, guarded by `self.dof != 6`, exactly as `R3_05 §1.2`
  specifies. Added `self.n_target_channels`.
- **Companion fix not in the spec text:** `to_canonical_dataframe` derives
  column names from `F_raw.shape[1]` with a hardcoded 3-or-6 branch; a 7-wide
  target would have zipped against the 6-name wrench tuple and corrupted
  column names on export. Now falls through to generic `ft0..ft{width-1}`
  naming for any other width.
- Added `declared_split_loaders(dataset, batch_size, include_val=False)`,
  falls back to the historical contiguous split when no `split` column is
  present (verified bit-identical index lists).
- Added an `n_target_channels`-vs-model-output-width guard in
  `model_workflow.py::train_model` (the shared entry point behind
  `lnn_tau_res.py`/`lnn_tau_elastic.py`/`kalnn_tau_*.py`) rather than
  duplicating the check across five files.

**`elastic_robot_sim/src/elastic_sim/dataset.py`**:

- `SplitPolicy` (`contiguous | holdout_trajectories | holdout_robots` — only
  the first and third are implemented; `holdout_trajectories` is accepted by
  validation but has no `assign_splits` behavior yet, matching the spec's
  scope) and `assign_splits`, which ranks elastic tiers by mean log
  stiffness and reserves the softest/stiffest strata for test, the
  next-most-extreme for val. Verified numerically (20 robots, `test_robots=4,
  val_robots=2` → exactly the 4 most extreme robots in test, no overlap).
- `split` column in `rollout_frame`, `split` block in the manifest,
  `<dataset>.contract.json` sidecar (schema `elastic_sim.identification/2`).
- Per-bag uniform-time-step assertion in `generate()` before anything is
  written.
- `docs/IDENTIFICATION_DATASET.md`: replaced the false "Consumer requirement"
  paragraph the spec itself flagged as inaccurate, documented the split
  machinery and the per-bag-normalization hazard.

## 3. Stage B — Parameter sampling (`R3_03`)

- `_stratified_unit`/`_sobol_unit`/`_sample_unit` (dispatch), used by both the
  legacy per-joint-interval path and the new nominal×factor path.
- `TransmissionSampling` extended with `sampling`, `stiffness_nominal`,
  `stiffness_factor`, `stiffness_common_fraction`, `stiffness_provenance`,
  `rotor_inertia_nominal`, `rotor_inertia_factor`, `rotor_inertia_provenance`.
- Rotor inertia now sampled (stream `(seed, 4)`, independent of stiffness)
  when `rotor_inertia_nominal` is given; stays fixed/unsampled otherwise,
  preserving every existing test that passes a bare `rotor_inertia=`.
- **Bug caught before it shipped:** the first draft of the correlated
  nominal×factor sampler fed the common/per-joint fractional draw through the
  same log-uniform helper used for stiffness intervals, which computed
  `log(0)` on a `[0, 1]` "interval." Fixed by giving the correlated sampler
  its own plain-uniform `[0,1)` primitive (`_sample_unit`) and doing the
  log-space centering explicitly.
- Coverage report (`sampling_coverage` in the manifest, printed warning below
  0.8) and the `--manifest` mismatch guard on `run_identification_simulation.py`.

**Deviation from the literal spec text, deliberate:** `R3_07` Task B1 says to
set `TransmissionSampling.sampling`'s *dataclass default* to `"stratified"`.
Doing that breaks the existing, currently-passing regression test
`test_robot_sampling_is_deterministic_and_prefix_stable`, which constructs
`TransmissionSampling(...)` directly with no `sampling=` argument and asserts
the historical prefix-stable behavior. Resolution: the **dataclass** default
stays `"iid"`; **`load_config`**'s effective default (when the YAML omits
`transmission.sampling`) is `"stratified"`. This gets every real dataset
build onto stratified sampling by default (the round's actual intent)
without breaking direct construction call sites or their tests.

**Deviation, numeric:** the shipped YAML's `stiffness_common_fraction` is
`0.0`, not the `0.5` in `R3_03 §2.1`'s illustrative example. Measured
directly: at `common_fraction=0.5`, per-joint coverage plateaus around
0.68-0.76 *regardless of robot count* (swept 20/24/30/40 robots), because
hitting a joint's marginal extreme then also requires the shared common
factor to land at an extreme for the same robot — a structural ceiling, not
a sample-size problem. `R3_07`'s Definition of Done requires ≥0.9 coverage on
every joint; `0.0` (fully independent per-joint, no correlation) is what
actually clears that bar at `robots: 20` (measured: stiffness min 0.930,
rotor_inertia min 0.934). Documented in the YAML with the actual numbers and
a note that `0.3-0.5` is available if correlated-arm realism matters more
than declared-range coverage for a given use of the dataset.

`damping_ratio` coverage measured at 0.78 (uniform, not stratified) — this is
correct per `R3_03 §4`'s own instruction not to touch it until the excitation
probe lands, which it now has; stratifying `damping_ratio` too is a candidate
follow-up, not done here since the spec explicitly says not to widen or
otherwise touch it in this round.

## 4. Stage C — Payload (`R3_01`)

New `src/elastic_sim/payload.py`: `Payload` (mass/offset/size), `payload_asset`
context manager. Implementation notes against the spec:

- Injection uses `text[:text.rindex("</robot>")] + injected + text[idx:]`,
  not `str.replace`, exactly as specified.
- `_last_link_name` derives the injection point from
  `asset.resolve_active_joints()[-1].child` — the last *active* joint's
  child link, not the URDF's last declared link.
- **Mesh resolution**: rather than composing with
  `generic_mujoco_runner._materialized_mujoco_urdf` as the spec suggested,
  the injected URDF is written as a **sibling file in the original asset's
  own directory** (`tempfile.NamedTemporaryFile(dir=original_parent, ...)`).
  Every consumer (`AssetSpec.validate_resources()`, `_materialized_mujoco_urdf`,
  Pinocchio's own URDF parser) already resolves relative mesh paths against
  `urdf_path`'s parent, so this gets the same "every consumer sees an
  identical asset" property the spec was after, with zero mesh-copying code
  and no dependency on that function's internal mesh-index-pairing invariant.
  **Verified end-to-end**: built a real MuJoCo model from a 5 kg
  payload-injected asset via the actual `_build_model`/`_materialized_mujoco_urdf`
  pipeline; total body mass increased by exactly 5.0 kg.
- `PayloadSampling`/`sample_payloads` (stream `(seed, 2)`, confirmed
  independent of the robot stream), threaded through `run_condition` under
  `with payload_asset(asset, payload) as asset_p:`, with a payload-keyed
  `link_inertia_envelope` cache (bounded at `robots + 1` entries for
  `per: robot`).
- Output schema: `payload_mass`/`payload_offset_x/y/z`/`payload_size`
  columns (`NaN` on rigid bags) and `defl0..defl{n-1} = q_motor - q_link`
  (the optional Map-C target from `R3_01 §3`, included since it was "one
  derived column away").

**Known gap, not fixed**: `scripts/run_identification_simulation.py` (the
single-rollout debug script) does not thread a payload through its
`run_condition` call — only `generate()`'s full dataset path does. Low risk
(that script's `--output` path is for inspection, not the dataset artifact
itself) but worth closing if the debug script is meant to reproduce a
payload-bearing bag exactly.

## 5. Stage D — Excitation bandwidth (`R3_02`)

- `indices=None` on `evaluate_series`/`project_coefficients`, verified
  bit-identical to the pre-change behavior for every existing call site.
- `probe_harmonics`/`probe_acceleration_fraction` on `FourierExcitationConfig`,
  with the aliasing guard (`match="alias"`) and the `n_harmonics` ordering
  guard.
- Two-budget scaling (`_scale_block`, main harmonics against
  `(1-fraction)*max_acceleration`, probe against the main harmonics'
  *leftover* position/velocity/acceleration budget).

**Two real bugs found and fixed during implementation, not present in the
spec's pseudocode as written:**

1. A shape mismatch — the acceleration-budget array was sized to the number
   of *harmonics* in a block instead of the number of *joints*, which
   crashes the very first probe-enabled candidate draw.
2. A correctness bug in the rest condition. `project_coefficients` was
   originally called once on the *combined* main+probe coefficient matrix.
   `_fit_to_limits` then scales the main and probe blocks by **different**
   per-joint factors (that is the entire point of the two-budget design).
   Scaling two sub-blocks of a matrix independently does not preserve a
   zero-sum property that only held for their *combined* sum — so the
   probe-enabled trajectory no longer started or ended at rest (measured:
   ~0.55 rad/s residual velocity at the boundary, where it should be ~0).
   Fixed by projecting the main and probe blocks **independently** before
   scaling (`sample_candidate` now calls `project_coefficients` twice, once
   per block); rest condition verified afterward at ~1e-16.

   This is exactly the kind of thing that would not have surfaced from
   reading the spec's pseudocode alone — only from constructing candidates
   and checking the boundary conditions numerically, which was done for both
   the probe-off and probe-on paths, with and without regime randomization.

- Regressor conditioning scored on main harmonics only (`optimize_excitation`
  re-evaluates `q_main, dq_main, ddq_main` from the main-block coefficients
  alone before calling `regressor_condition`).
- `RegimeSampling`/`regime_excitation` (stream `(seed, 3, trajectory_seed)`),
  wired into both `generate()` and `run_identification_simulation.py`
  (which gained a `--trajectory` index flag it did not have before, needed
  for the two call sites to be comparable at all beyond trajectory 0).
  `base_frequency` left fixed per the spec's own "recommended for first
  iteration" guidance.
- Shipped YAML: `probe_harmonics: [400, 700, 1000, 1300, 1600]` (40-160 Hz at
  `base_frequency: 0.1`), `regime.enabled: true` with
  `max_acceleration: [1.0, 8.0]`, `velocity_fraction: [0.3, 0.9]`.

**FFT sanity check performed** (trajectory generation only, no simulated
rollout — that needs Pinocchio): confirmed the probe's spectral peaks land
exactly at 40/70/100/130/160 Hz as configured, at the expected (very small —
correctly so, since acceleration scales as `f²`) position amplitude. The
*deflection*-domain check from `R3_08 §B2` (does a real elastic rollout's
`q_motor - q_link` FFT show a peak at each joint's predicted transmission
mode) is written (`test_probe_excites_the_transmission_mode`) but could not
be executed — it needs Pinocchio.

## 6. Stage E — Provenance (`R3_04`)

- `docs/PARAMETER_PROVENANCE.md`: anchor-class table, all seven joints × two
  parameters, filled with the two published anchors the spec names (Disney
  Research iiwa base joint, 18500 Nm/rad / 1.03 kg m²; Iskandar et al. DLR
  SARA J5, 9000 Nm/rad / 0.339 kg m²), the "why factor 2" argument, the
  control-bandwidth bound, the UR10 porting procedure, and a "measurements
  pending" table. Interpolated/tapered classes (`C`/`E`) used for joints with
  no anchor exactly at that size — stated as such, not hidden.
- Provenance letters (`stiffness_provenance`, `rotor_inertia_provenance`)
  were wired into `TransmissionSampling`/`load_config`/manifest as part of
  Stage B's `TransmissionSampling` extension, not a separate pass.
- `scripts/report_parameter_bounds.py`: computes the A6 checklist per joint
  (declared span, decades, mode-frequency range via `link_inertia_envelope`,
  control-bandwidth lower bound, required step at `k_max`) plus a
  regex-parsed pull of the published-anchor line from the provenance doc for
  each joint. Writes `bounds.csv`/`bounds.md`/`modes.png`. The anchor-parsing
  regex was tested standalone against the real doc and correctly extracts
  all seven rows; the numeric half needs `link_inertia_envelope`, i.e.
  Pinocchio, and could not be run end-to-end here.
- `scripts/run_range_sensitivity.py`: generates the S0-S3 train/test dataset
  pairs (`library calls, not subprocess`, per the spec's "does not call into
  the other repo" constraint applied to training, not to reusing this
  repo's own `generate()`). **One deviation**: the spec says to "print the
  command lines to train and evaluate each" in `dynamic_model_nn`; that repo
  has no generic training CLI (each model family is its own script with
  hardcoded settings), so this prints an illustrative Python snippet against
  its actual programmatic surface (`CustomDataset`, `create_loaders`,
  `train_model`) instead of fabricating shell flags that don't exist.
  Variant config construction (robots/factor/tiers) verified without
  Pinocchio; the actual `generate()` calls were not run.

## 7. Stage F — Performance and storage (`R3_06`)

- **F1 (index hoisting)**: all four torque runners
  (`run_mujoco_torque`, `run_mujoco_elastic_torque`, `run_newton_torque`,
  `run_newton_elastic_torque`) now build integer index arrays once and use
  fancy indexing instead of per-step dict/tuple lookups and list
  comprehensions. Dtype fidelity was checked line by line against the
  original code (MuJoCo's `qpos`/`qvel` are float64 natively either way; the
  rigid Newton runner had an explicit `dtype=float`, preserved; the
  **elastic** Newton runner did *not* cast explicitly in the original —
  preserved that too, since adding a cast there would have silently changed
  float32 state to float64). Smoke-tested end-to-end with a synthetic
  constant-torque controller (no Pinocchio needed for this part): correct
  output shapes, no crashes, for both the rigid and elastic MuJoCo runners.
  The formal bit-identical regression test (`R3_08`'s
  `test_index_array_hoisting_is_bit_identical`, which needs a real
  Pinocchio-backed controller to compare against a pre-refactor baseline) is
  written but not executed.
- **F2 (preallocate + record-only-written-steps) — not implemented.**
  Preallocating output arrays is safe and mechanical; skipping this because
  the higher-value, correctness-relevant half of F2 (record every k-th
  physics step directly, drop the `np.interp` resampling `rollout_frame`
  currently does) is exactly the kind of physics-loop change that produced
  the two Stage D bugs above, and there was no way to verify it against a
  real rollout in this environment. Shipping an unverified change to the
  actual recording mechanism risked silently corrupting the labels the whole
  round is trying to make trustworthy. **This is the one item from `R3_07`
  genuinely left undone**; the note in `R3_06 §2.3` that this interpolation
  "stops being harmless" once the probe puts 40-160 Hz content into the
  trajectory is correct and still stands as a follow-up.
- **F3 (`control_decimation`)**: added to all four runners. The controller
  (and the friction-corrected torque derived from it) is held zero-order
  across `N` physics steps; the recorded label (`tau_spring`/deflection) is
  **always** read from the true, continuously-integrated state, never held —
  this was a deliberate design choice to keep the "label is exact" property
  Stage A/the base doc rests on. An aliasing guard
  (`1/(time_step*decimation) > 10*probe_top_hz`) was added at the
  `run_condition` call site, since that is where both the transmission's
  time step and the trajectory's `probe_top_hz` are simultaneously known.
  Default left at `1` (no behavior change) in the shipped YAML — raising it
  needs the feedback-ratio and rigid-vs-Pinocchio-residual tests re-verified
  at the chosen value, which this environment cannot do.
- **F4 (`--jobs N`)**: `generate()`'s bag loop refactored into a top-level,
  picklable `_run_bag(args)` function and a work-item list, dispatched via
  `concurrent.futures.ProcessPoolExecutor` when `jobs > 1`, plain sequential
  otherwise (default). `--jobs > 1` + `--visualize` is a hard CLI error.
  **Verified concretely**: every object that crosses the process boundary
  (`AssetSpec`, `DatasetConfig`, `Tier`, `FrictionModel`, `MaterializedTrajectory`,
  `Payload`, the link-inertia tuple) round-trips through `pickle.dumps`/`loads`;
  the `_run_bag` function reference itself pickles; a real
  `ProcessPoolExecutor` spawn-and-import round-trip against this package
  succeeds on this machine. `pool.map` preserves input order, so row
  ordering is identical to sequential execution by construction; per-bag
  values are independent of execution order because every RNG stream is
  keyed on `(seed, index)`, never on call order. The actual bit-identical
  dataset comparison (`R3_08`'s `test_parallel_generation_is_bit_identical`)
  is written but needs Pinocchio to run `generate()` for real.
- **F5 (storage)**: `write_dataset` dispatches on output suffix
  (`.csv` written with `float_format="%.7g"`; `.parquet` written with a
  float32 downcast) and takes `metadata_columns: "inline" | "sidecar"`
  (sidecar mode moves the ~42 per-bag-constant columns to a
  `<dataset>.bag_metadata.json` keyed by bag). **Fully verified** with
  synthetic frames, no Pinocchio needed: inline vs. sidecar column sets,
  sidecar JSON content, Parquet round-trip and dtype, and both validation
  error paths (bad `metadata_columns`, unsupported suffix).

## 8. Environment (not in the original spec, done at the user's request)

Mid-round, the user asked for `pyproject.toml` to be made Windows-compatible
without touching Linux/macOS. Six sequential blockers were found and fixed
in order:

1. `pin==4.1.0`'s dependency chain needs `cmeel-assimp>=6.0.2`, which has no
   win_amd64 wheel → added `pin==4.1.0; sys_platform != 'win32'` /
   `pin==3.4.0; sys_platform == 'win32'` (the newest release still on the
   older, Windows-buildable `coal-library`/`cmeel-assimp` chain).
2. `cmeel-console-bridge` has no win_amd64 wheel at any version and its
   vendored CMakeLists.txt predates CMake 4.0's removal of pre-3.5 policy
   compatibility → this needs `CMAKE_POLICY_VERSION_MINIMUM=3.5` set in the
   **environment**, documented in a `pyproject.toml` comment (a global
   build-constraint on `cmake<4` was tried first and reverted — it broke a
   sibling package, `cmeel-zlib`, which needs `cmake>=4`).
3. `patch.exe` (GNU patch, used by `cmeel`'s own build hook) is not on a
   plain Windows/cmd PATH, only under Git for Windows' `usr\bin` — documented
   as a required PATH addition alongside the env var above.
4. `cmeel`'s post-build test step (`cmake --build ... -t test`) targets a
   name the MSVC generator doesn't produce → `CMEEL_RUN_TESTS=OFF`.
5. No MSVC toolchain was installed on this machine at all — installed
   Visual Studio Build Tools (C++ workload) via `winget`, with the user's
   explicit sign-off given the size of that action.
6. A genuine upstream bug: `cmeel-octomap`'s `compare_octrees.cpp` defines
   `#define isnan(x) _isnan(x)` under `#ifdef _MSC_VER` for pre-C++11 MSVC,
   which under modern MSVC clobbers the `std::isnan` the same file already
   has a correct C++11 branch for. **This patch was applied directly to the
   file inside uv's local build cache, not to anything version-controlled**;
   it is not portable to another machine and will be lost if that cache
   entry is evicted. If Windows-native Pinocchio matters going forward, this
   needs a real fix (report upstream, or vendor a patch file cmeel's own
   `cmeel.patch` mechanism would pick up).

After all six, the build got past `cmeel-console-bridge`, `cmeel-octomap`,
and several other `coal-library` dependencies, and failed on **Boost
1.87.0's own bootstrap** (`bjam`/`project-config.jam` not generated by
`cmeel`'s CMake `ExternalProject` wrapper on Windows). Per the user's
explicit decision, this was **not pursued further** — building Boost from
source on Windows is its own multi-day rabbit hole with no guarantee of
success. `pyproject.toml` is left in a strictly-better state (six real,
documented blockers cleared; the platform-marker split is correct and inert
on Linux/macOS) but `uv sync` does not yet succeed end-to-end on this
specific Windows machine.

## 9. Verification summary

| Where | Result |
|---|---|
| `elastic_robot_sim` — full `tests/` suite | **153 passed**, 14 failed, 22 errors, 1 skipped |
| Failures/errors | All trace to `ModuleNotFoundError: pinocchio` (confirmed identical failure set before and after every change in this round, modulo the new tests added) |
| `dynamic_model_nn` — `test_dataset_generalization.py` | **11 passed** (5 pre-existing + 6 new) |
| `dynamic_model_nn` — `test_data_preprocessing_ur.py` | **5 passed** (unaffected, re-run as a cross-check) |
| Cross-repo regression | None found in either repo |

Everything that *can* run without Pinocchio (config loading and validation,
sampling distributions and coverage, split assignment, excitation coefficient
construction and rest/limit conditions, payload URDF injection through a
real MuJoCo model build, multiprocessing/pickling machinery, Parquet/sidecar
storage) was exercised directly, often with purpose-built standalone scripts
beyond the checked-in test suite, and is reported above with actual
measured numbers rather than "should work."

Everything that needs Pinocchio — most importantly the round's
non-negotiable invariant, **base parameters recovered from a generated
rollout to better than 1e-4 relative error** (`R3_08 §F`) — is written,
`pytest.importorskip`-gated, and has not regressed by inspection (no
controller, RNEA, or CRBA call site changed shape or semantics in this
round; only outer plumbing did), but was **not executed** and should be
before this is trusted for a real dataset build.

## 10. Recommended next steps, in order

1. Get Pinocchio working somewhere reachable from this branch (Linux/WSL is
   the fast path; finishing the Windows Boost bootstrap is the slow one) and
   run the full suite, in particular `test_base_parameters_are_recoverable_from_a_rollout`,
   the backend-agreement tests, and the two Stage D probe tests
   (`test_probe_excites_the_transmission_mode`,
   `test_damping_ratio_becomes_observable_with_the_probe`) — these are the
   tests that actually close F3, the highest-severity finding in the round.
2. Run the Definition-of-Done command from `R3_07` for real:
   `generate_identification_dataset.py --backends mujoco --no-rigid --robots 20 --trajectories 3`,
   and check the printed coverage/provenance/probe summary against the
   numbers this report cites from dry runs.
3. Decide on `control_decimation` and F2 with real timing/accuracy numbers
   in hand, rather than the "left at safe defaults" posture this report
   ships with.
4. If Windows-native builds matter long-term, either report the
   `cmeel-octomap` bug upstream or vendor the one-line fix as a real patch
   file; right now it only exists in a local build cache.
