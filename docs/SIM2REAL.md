# Sim-to-real calibration (`elastic_sim.sim2real`)

The sim2real loop replays recorded real-robot runs in simulation, compares
simulation and robot on the recorded torque channel, and calibrates the
plant parameters with an optimizer, judged on physical metrics. It reuses the
simulation core (`identification`, `plant_extras`, the MuJoCo builders); it
has no plant of its own.

Three replay engines: **R0** (inverse dynamics on the recorded state; the
calibration loop runs on it), **R1** (motor-forced elastic replay) and
**R2** (closed-loop replay of the robot's command through a KUKA
position-loop emulation). R1 and R2 run on the production MuJoCo elastic
runner through two opt-in hooks (`initial_state`, `model_setup`); with
neither set, the runner is bit-identical to before.

## Concepts

| term | meaning |
|---|---|
| real run | an `erd.lab/1` run folder (`data/real_robot/<robot>/<date>/<run>/`): dataset, frozen plan, raw stream, identification recording |
| bag | one recorded segment on its 1 kHz FRI cycle grid: `excitation_k` (from the dataset), `identify_excitation_0` and `identify_hold_k` (from `identification/raw.parquet`) |
| sidecar | what only `raw.parquet` has, matched row by row to each bag: the FRI command echo (`cmd`, the tracking reference), `ipo` (FRI's `ipoJointPosition`, the overlay's held base pose: kept, never a reference), external torque, `fri_cycle`. Cached in `data/sim2real/cache/<robot>/<run>/` |
| theta | the plant parameters by name, `<group>.<subject>.<field>[@<run>]`, e.g. `inertial.iiwa_link_2.mass`, `sensor.ft.iiwa_A4.offset@B` |
| registry | every parameter's value, prior (mean, sigma), bounds, scale and provenance; the free ones live in the optimizer's box `z in [-1, 1]` |
| overlay | a YAML of the stage's parameters in physical units: `working.yaml` (rewritten while a stage runs) and `best.yaml` (the result), each entry `calibrated: true \| false` with its data share `d_post` (and the diagonal `d`, a diagnostic) |
| session zero | the test run's sensor offsets, fitted on its holds after the stage fit with everything else frozen (A-13): the only test-run data training ever sees |

## R0 replay

Per kept row:

```
q_link = q - (ft - o) / (g K)                      measured deflection
tau_s  = RNEA_theta(q_link, dq, ddq) + f_link(dq)   rigid chain + link friction (eps = 1e-3)
ft_sim = g * tau_s + o[run]                         joint torque sensor model
```

* `dq`, `ddq`: Savitzky-Golay of the measured `q` (`replay.r0.sg_window`,
  `sg_poly`; default 31, 3); rows decimated (`decimate`, default 5); holds
  at rest, decimated by `hold_decimate` (default 50).
* The model carries theta's inertials plus the elastic build's 1 g
  intermediate bodies, so R0 and the elastic engines calibrate one plant.
* The Jacobian `d ft_sim / d theta` is analytic (joint-torque regressor and
  the chain rule of the mass / COM / inertia-scale parameterisation; RNEA
  derivatives for the deflection terms).
* Gate G-1: with the contract's recipe (dataset `dq`, SG(5, 3) of it, no
  friction, rigid, nominal), R0 reproduces each run contract's
  `baseline_rms_rigid` exactly.

## R1 replay (motor-forced elastic)

The measured motor trajectory is imposed on the elastic chain and the link
side integrated (`replay_r1.py`); `ft_sim = g * tau_s + o[run]`.

* **Reference:** a smoothing cubic spline of the measured `q`, smoothing set
  per joint so the fit residual equals the run's hold sigma(q) (the
  discrepancy principle; GCV would smooth away the real 10-50 Hz motor
  motion). `replay.r1.reference_filter: savgol` pre-filters `q` with R0's SG
  first (gate G-3).
* **Controller:** the exact motor-side law
  `tau_m = J_r a + tau_s + f_m(dq)`, `a = ddq_ref + kd e' + kp e`
  (`MotorForcing`; `track_bandwidth`, default 300 rad/s). It forces the
  measured motor motion; it is not KUKA's controller.
* **Plant:** theta's inertials (in memory), K and zeta from theta, rotor
  inertia as armature, motor and link friction (velocity-implicit, as in
  production). `replay.r1.extras: plant` adds the round-6 nonlinear spring and
  transmission error.
* **Start:** static equilibrium of the deflection (`equilibrium.py`, Newton,
  residual <= 1e-9 N m); with `windows`, each later window starts
  quasi-statically from the measured ft and its `burn_in_s` is not scored.
* **Rows:** R0's grid, so R0 and R1 results compare row for row. One process
  per bag (window) with `jobs`.

## R2 replay (closed loop)

The FRI command echo (1 kHz, a fresh command per cycle) drives the KUKA
position-loop emulation (`controllers.py`: `SeaMotorController` on the plant
theta's model x `controller.model_scale`, per-joint `controller.<joint>.
bandwidth` / `damping_ratio`, at `controller.drive_rate`, the command
`controller.delay_cycles` late; or `pd_gravity`) on the elastic plant,
starting at the measured pose in static equilibrium (`replay_r2.py`).
Observed: `ft` and the tracking error `q - cmd`, real vs simulated.

## Configuration

Stage configs extend the shared base:

```yaml
schema: elastic_sim.sim2real/1
extends: kuka_lbr_iiwa_14_r820_base.yaml      # runs, split, priors
prior_from: data/sim2real/kuka_lbr_iiwa_14_r820/iiwa_s0_static/best.yaml   # previous stage's result as prior
replay:
  mode: R0
  r0: {rows: all}                              # S0: quasi_static (|dq| < 0.05, |ddq| < 0.2)
parameters:
  free: ["inertial.*.mass", "inertial.*.com_*", "inertial.*.inertia_scale",
         "friction.link.*.*", "sensor.ft.*.offset@*", "sensor.ft.*.gain"]
loss:
  prior_weight_factor: 0.01                    # x the auto weight 1/N
optimizer: {method: least_squares, budget: 50}
```

In `parameters.priors` the last matching pattern wins, and a pattern an
extending file re-declares moves to the end.

The base config also sets the split and the identifiability procedure:

```yaml
split:                                         # A-13: per-session zero
  train: ["B:excitation_[0-6]", "B:identify_hold_*"]
  val: ["B:excitation_[7-9]", "B:identify_excitation_0"]
  zero: ["A:identify_hold_*"]                  # fits only sensor.ft.*.offset@A, after the stage
  test: ["A:excitation_*", "A:identify_excitation_0"]
identifiability: {mode: label, d_post_min: 0.9, partner_min: 0.5}   # A-12' (freeze_refit: A-12)
selection: val                                 # or end_point (S0: a warm start, F-6)
```

Offsets of a run no `train` pattern names are excluded from the free set
automatically.

## One stage, step by step

1. **Fit** (LS by default) within the budget; the result is selected on
   `selection`: `val` (lowest val data loss) or `end_point` (the optimizer's
   own end point).
2. **Calibrated label** (A-12′, `mode: label`): d_post once, on the full
   free set, at the selected point. Each free parameter is
   `calibrated: true` iff d_post >= 0.9, else `calibrated: false` ("not
   separated by these data") with its **partners** (free parameters with
   |posterior correlation| >= 0.5). No value changes: a not-calibrated value
   is nominal plus the minimum-norm share of a data-determined combination,
   to be changed only together with its partners.
   `mode: freeze_refit` is S2R_06's A-12 instead: the lowest-d_post
   parameter is frozen at its prior mean, d_post recomputed, until every free
   one has d_post >= 0.9, then a refit (`history_refit<k>.jsonl`).
3. **Session zero** (A-13) on the `zero` bags; theta0 gets the same zero.
4. **Scores**: the A-2 suite per split, the **strict** test column (the test
   run's offsets = the training run's) and, for row-filtered stages, every row.

| config | stage |
|---|---|
| `config/sim2real/kuka_lbr_iiwa_14_r820_base.yaml` | runs A/B, split (B train, A test), priors |
| `config/sim2real/iiwa_s0_static.yaml` | S0: masses, COM, offsets on holds + quasi-static rows |
| `config/sim2real/iiwa_s1_rigid.yaml` | S1: + inertia scales, link friction, sensor gain, every row |
| `config/sim2real/iiwa_s0_static_mirror.yaml`, `iiwa_s1_rigid_mirror.yaml` | the mirror split: A trains, B's holds zero B, B tests |
| `config/sim2real/iiwa_twin_r0.yaml` | gate G-2a: the R0 twin experiment on B's states |

## Loss and metrics

The loss is `J = sum_c w_c mean_bags mean_joints RMS(sim - real) / s_cj + lambda * prior_penalty`,
with `s_cj` the training RMS of the channel, floored at 10x the hold noise and
5 % of the largest joint. Least squares gets the same normalisation as a
stacked residual plus the prior rows.

`metrics(results, split)` is the one definition every report uses:

| | metric |
|---|---|
| (a) | torque RMSE, N m: per bag over joints and rows, median over bags (the contract baselines' statistic); per joint RMSE and NRMSE |
| (b) | hold gravity slopes per run, `ft = a (g tau_s) + b`, on joints whose gravity spans >= 2 N m |
| (c) | friction signature: residual mean in velocity bins, and its odd (Coulomb-like) part |
| (d) | residual spectrum: mean-square power <= 0.5, 0.5-2, > 2 Hz |
| (f) | plausibility: calibrated vs not calibrated, prior distance in sigma, `d_post` (and `d`), physical consistency, mass/COM moves |

Identifiability (`identifiability.json/.md`) reports per parameter the
sensitivity, the data share `d_post` (posterior, with correlations;
data-determined from 0.9, F-2) and the diagonal `d` (a diagnostic), the
collinearity index, the most collinear pairs, a recommended frozen set
(collinearity rule, reported) and A-12's frozen list with the d_post that
froze each.

## Scripts

| script | does |
|---|---|
| `scripts/sim2real/build_replay_cache.py A B` | builds the sidecar caches (no-op when unchanged); `--no-save` checks only |
| `scripts/sim2real/replay.py --run B --bag excitation_5 [--mode R1\|R2] [--theta working.yaml] --show` | watch one bag: metrics vs the contract baselines, ft real vs sim plots (R2: and q - cmd); `--show` saves nothing; `--visualize` (R1/R2) the MuJoCo viewer, nothing saved |
| `scripts/sim2real/identifiability.py --config ...` | the identifiability report at the config's start theta; `--no-save` prints |
| `scripts/sim2real/calibrate.py --config ...` | one stage, start to finish; `--method/--seed/--budget` for comparisons, `--free-from RUN_DIR` (comparators on that run's calibrated set), `--mode label\|freeze_refit`, `--resume`, `--dry-run`, `--report-only RUN_DIR` |
| `scripts/sim2real/sensitivity.py --config ...` | the stage at lambda x0.1 and x0.001, flagged differences into the stage's report; `--no-save` prints |
| `scripts/sim2real/candidate.py RUN_DIR` | the result as `candidate_calibrated.yaml` in the run folder (published only after review); `--show` prints the calibrated / not table |
| `scripts/sim2real/compare_runs.py RUN_DIR ...` | rank runs on the test metrics (zeroed and strict), calibrated count, slopes, efficiency |
| `scripts/sim2real/twin.py [--seeds N ...]` | gate G-2a': twins of run B at known theta* (default seeds 20261019, 7, 11, 3, 5), calibrated and scored; verdict over seeds; `--mode`, `--output` |
| `scripts/sim2real/gate_g3.py` | gate G-3: R1 with K x 100 vs R0 on three bags of B (full band, same kinematics, below 10 Hz); `--no-save` prints |
| `scripts/sim2real/f9_diagnostic.py` | the F-9 diagnostic: R0 residual per band against the twin floor, its coherence with q'' and cmd - q, the ripple order, R1 at the same theta; `--no-save` prints |
| `scripts/sim2real/r2_nominal.py` | R2 at nominal theta on every bag: divergences, q - cmd real vs sim per joint; `--no-save` prints |
| `scripts/sim2real/benchmark.py` | T-3.6: wall time per bag of R0/R1/R2, one worker and nproc - 1; flags an evaluation over 10 min |

## Run folder

`data/sim2real/<robot>/<run_id>/`: `config.yaml` (resolved), `inputs.json`
(dataset sha256, plan digests, `erd-src-digest/2`), `registry.yaml`,
`identifiability{,_theta0}.{json,md}`, `history.jsonl` (one line per
evaluation; resume replays it) and `history_refit<k>.jsonl` (A-12 refits),
`working.yaml`, `best.yaml`, `metrics.json` (theta0, the result, the other
selection rule's point, the strict test, the session zero, the
not-calibrated list), `replay_<split>.parquet` (`zero` included),
`figures/`, `report.md`, optionally `sensitivity.{md,json}` and
`candidate_calibrated.yaml`. Writes inside the repository go only under
`data/sim2real/` (and, for the final publish, `config/sim2real/calibrated/`).

### `replay_<split>.parquet` columns

| column | meaning |
|---|---|
| `run`, `bag`, `kind`, `split`, `mode` | identity of the row's bag (`kind`: excitation, identify_excitation, hold) |
| `row` | the row's index in the bag's 1 kHz grid |
| `t` | s from the bag's first cycle |
| `q{j}` | measured motor position, rad |
| `dq{j}`, `ddq{j}` | the replay's derivatives (SG of `q`; 0 at holds) |
| `q_link{j}` | link position the model was evaluated at |
| `ft{j}` | measured joint torque, N m, model sign |
| `tau_s{j}` | simulated spring torque |
| `ft_sim{j}` | `gain * tau_s + offset` |
| `res{j}` | `ft_sim - ft` |
| `gain{j}`, `offset{j}` | the observation model of the bag (constant per bag) |
