# Simulation-only identification datasets

This is a second, independent path through the repository. The sim-to-real
workflow (`run_experiment.py` → `run_calibration.py`) executes trajectories on
a real robot and calibrates the simulator against the recording. The path
described here never touches hardware: it builds a scene, designs trajectories
that excite the robot's dynamics, runs them as torque-driven rollouts in
MuJoCo and Newton, and emits a dataset for machine-learning dynamic model
identification.

## Quick start

Both scripts read their defaults from
`config/identification/kuka_lbr_iiwa_14_r820_table.yaml`, and every setting in
it can be overridden on the command line.

```bash
# 1. Compose the scene: KUKA LBR iiwa 14 on a 0.75 m table.
uv run python scripts/compose_scene_urdf.py \
    --asset kuka_lbr_iiwa_14_r820 --table-height 0.75 --write-asset-yaml

# 2. Generate the dataset, entirely from the config file.
uv run python scripts/generate_identification_dataset.py

# ...or override whatever you need.
uv run python scripts/generate_identification_dataset.py \
    --backends mujoco --robots 12 \
    --trajectories 10 --no-rigid \
    --output data/identification/iiwa14_table.csv

# 3. Re-check MuJoCo/Newton agreement of a written dataset, no simulation.
uv run python scripts/compare_identification_backends.py \
    data/identification/kuka_lbr_iiwa_14_r820_table.csv
```

### Other assets: UR10

`config/identification/ur10_table.yaml` runs the same stack on a UR10 (CB3)
on the same table, selected with `--config`:

```bash
uv run python scripts/compose_scene_urdf.py \
    --asset ur10 --table-height 0.75 --write-asset-yaml

uv run python scripts/generate_identification_dataset.py \
    --config config/identification/ur10_table.yaml
```

What differs from the iiwa, and why, is in
`docs/PARAMETER_PROVENANCE.md#ur10`; in short:

- **Excitation window.** The URDF joint limits are +-2 pi on five of the
  UR10's six joints, wide enough that an unconstrained excitation spreads
  trajectory centres over dynamically-duplicate (wrap-around) and
  table-colliding configurations. `excitation.position_window` (a per-joint
  `[lo, hi]` override, intersected with the URDF limit) narrows this to a
  sensible band; it is a generic feature, not UR10-specific, and a no-op
  (the iiwa keeps its URDF limits verbatim) when unset.
- **Control gains.** `simulation.control_gains.natural_frequency: [4.0, 7.5]`
  rad/s, well below the iiwa's `[15, 40]`: the UR10's heavier link inertia
  and reflected rotor inertia bring the control/transmission separation
  bound `k >= (5 omega)^2 J_eff` down with it. `simulation.control_separation`
  checks this bound per bag (manifest-only fields, every config including
  the iiwa's gets the check with the iiwa's own numbers).
- **Probe band.** `excitation.probe_harmonics` covers 3-150 Hz (iiwa:
  8-190 Hz) -- the UR10's proximal open-loop modes sit at 7.5-67 Hz, lower
  than the iiwa's 50-170 Hz, and the closed-loop observable resonance is
  lower still.
- **Friction is a no-op.** The UR10 URDF declares `damping="0"
  friction="0"` on every joint, so `friction_samples`/`friction_scale` do
  nothing on this asset and the rigid tier has `tau == ft` exactly, as on
  the iiwa.
- **6-DoF contract.** At `n_dof == 6`, `ft0..ft5` collides in name with a
  legacy end-effector wrench some consumers infer from column count; the
  written `.contract.json` sidecar's `target_kind: "per_joint_torque"` is
  load-bearing for this asset in a way it never was for the 7-DoF iiwa.
- **Cost.** The bare-flange wrist-3 mode (400-1200 Hz, the same situation as
  the iiwa's A7) sets the integration step, so a UR10 rollout is not cheaper
  than an iiwa one despite the arm looking "softer" overall. Measured
  (7-robot smoke builds): with the shipped `payload.mass: [0, 5]` fitted,
  the median elastic step is `~4.4e-4 s` -- itself already *coarser* than
  the iiwa's own `6-8e-5 s`, i.e. cheaper per step, not more expensive; the
  "not cheaper" correction bites specifically when the flange is bare
  (`payload.enabled: false` or every draw near 0), which drops the step to
  `~7.5e-5 s` and the wall time per bag to ~3.7-3.8x the payload-fitted
  case.
- **Payload.** `payload.enabled: false` gives every bag a bare flange (no
  payload drawn at all); with `enabled: true`, `payload.mass`/`offset_*`/
  `size` are YAML-settable intervals sampled the same way as every other
  robot parameter -- there is no separate "minimum tool mass" knob, just the
  interval's own floor.
- **Backends.** `--backends mujoco` only is recommended for a full UR10
  build. Newton's *elastic* path runs `SolverMuJoCo` (the MuJoCo-Warp GPU
  port, float32) rather than an independent solver, and was found to
  occasionally diverge on the softer/lower-damping end of the sampled
  stiffness prior -- deterministically given a fixed `--jobs` count, but on
  a *different* bag depending on the process/worker layout. That pattern
  (same divergence, different bag, when only the process/worker count
  changes) points at **state shared across bags inside one process**
  (a reused model/solver cache) rather than at float32 precision itself,
  which would be expected to hit the same bag regardless of process layout;
  not established either way yet. Newton (`SolverFeatherstone`) remains a
  genuine independent cross-check on the **rigid** tier only
  (`--robots 0 --backends mujoco newton`).

## Looking before you save

`run_identification_simulation.py` runs a *single* rollout and writes nothing
unless you pass `--output`. Use it to see whether a trajectory and the model's
numbers make sense before committing to a full dataset build.

```bash
# Watch the arm move, save nothing.
uv run python scripts/run_identification_simulation.py --visualize

# Fast-forward 4x, and look at sampled elastic robot e03 in Newton.
uv run python scripts/run_identification_simulation.py \
    --visualize --realtime-scale 4 --tier e03 --backend newton

# Check a trajectory without starting a simulator at all.
uv run python scripts/run_identification_simulation.py --trajectory-only

# Keep this one rollout after all.
uv run python scripts/run_identification_simulation.py --output /tmp/one.csv
```

It reports the regressor condition number, peak velocity and acceleration
against their limits, whether the path starts and ends at rest and stays inside
the joint-limit band and collision margin, then the tracking error, the
feedback/feedforward ratio, torque against the effort limit, for an elastic
robot its stiffness, damping, mode frequencies and integration step, and — for
the rigid tier — the residual against Pinocchio's inverse dynamics, which should be
about 1e-8 Nm.

`generate_identification_dataset.py` takes `--visualize` and `--no-save` too,
so a whole dataset build can be watched or rehearsed without writing anything.

Lower `--base-frequency` and more `--candidates` give a better-conditioned
trajectory. The config defaults to the good end (0.1 Hz over 10 s, 48
candidates, condition number ≈110); `--base-frequency 0.5 --candidates 8` runs
in a fraction of the time but conditions around 38000, which amplifies torque
error into parameter error by the same factor. Use the fast values for smoke
runs, not for a dataset you intend to train on.

> The MuJoCo passive viewer segfaults on teardown in some environments, after
> the run has completed and all results are printed. This reproduces with a
> six-line stock MuJoCo script and is not specific to this repository.

The result is one flat CSV plus a `.manifest.json` recording every bag's tier,
per-joint transmission parameters, friction sample, backend, solver, trajectory
digest and regressor condition number, and — when more than one backend ran —
a `.backend_comparison.csv` (see [Backend comparison](#backend-comparison)).

## Why torque-driven rollouts

The position-tracking runners record `tau = Kp(q_ref − q) + Kd(dq_ref − dq)`,
which is a *controller command*, not a joint torque consistent with the
dynamics — and MuJoCo and Newton compute it differently. It is not a usable
identification label.

These rollouts invert the arrangement. A torque is computed from the model and
injected as a pure generalized force; the simulator only answers what motion
that torque produces. The label is then exact by construction and identical
across backends, so backend disagreement appears purely as state divergence.

The control law is **computed torque**:

```
tau = rnea(q, dq, ddq_ref + kd*e_dot + kp*e) + friction(dq)
```

Substituted into `M qdd + C qd + g = tau − friction`, the model cancels and
leaves `e_ddot + kd e_dot + kp e = 0`. Two properties follow, and both matter:

- The commanded torque still satisfies `tau = rnea(q, dq, qdd) + friction(dq)`
  on the *achieved* state, so the recorded label is exactly consistent with the
  dynamics.
- A single scalar gain pair fixes every joint's closed-loop bandwidth. This is
  not a convenience: the iiwa's joint-space inertia matrix has a condition
  number around 1.5e4 and its last link's inertia is 3e-4 kg m², so independent
  per-joint PD gains either drift or go unstable.

Pure feedforward with no feedback drifts about 3.5 rad over 2 s, and that drift
is identical across every MuJoCo integrator and timestep — it is the physics of
an unstable plant, not an integration artifact.

## Backend parity

The same URDF is read differently by the two importers:

| URDF `<dynamics damping="10.0" friction="0.1"/>` | MuJoCo | Newton |
|---|---|---|
| `damping` | `dof_damping` (passive viscous force) | `joint_target_kd` (a **PD controller gain**) |
| `friction` | `dof_frictionloss` | `joint_friction` |

Left alone, the two backends would simulate different machines. The torque
runners therefore neutralize all passive effects in both — damping, friction
loss, armature and Newton's built-in PD — and re-inject friction explicitly
through the applied torque. Both then integrate
`M(q) qdd + C(q,qd) qd + g(q) = tau_solver` and nothing else.

On the **rigid** path Newton uses `SolverFeatherstone` deliberately:
`SolverMuJoCo` would run MuJoCo's own engine under Newton's API and could not
serve as a cross-check. On an identical applied torque the two agree to about
2e-6 rad, which is the error bar that justifies running both.

On the **elastic** path this is not available. Newton's `SolverFeatherstone`
and `SolverSemiImplicit` both diverge within a few milliseconds on the
doubled-degree-of-freedom elastic chain — at every transmission-body size and
time step tried — so that path falls back to `SolverMuJoCo` and is **not** an
independent check on the MuJoCo rollout. Every rollout reports this as
`independent_of_mujoco`, and it is also several times slower than the MuJoCo
elastic runner. It still runs a different implementation (MuJoCo Warp, single
precision, on the GPU), so agreement is a useful consistency check, but not a
second physics model.

Contacts are disabled during a rollout. Excitation trajectories are validated
collision-free against the real geometry beforehand, so contacts can only add
forces that are not part of the model being identified. This is also a
correctness requirement for the elastic chain: inserting a transmission body
between two links stops MuJoCo treating them as a parent/child pair, so it
stops excluding their contact automatically, and neighbouring links are then
found deeply interpenetrated at their shared joint — forces that silently lock
the arm.

## Excitation trajectories

`src/elastic_sim/excitation.py` implements the standard finite Fourier series,
one per joint:

```
q_i(t) = q_i0 + sum_k [ a_ik/(k w) sin(k w t) − b_ik/(k w) cos(k w t) ]
```

Constraining `sum_k a_ik = 0` and `sum_k k b_ik = 0` makes velocity and
acceleration vanish at `t = 0`; periodicity makes them vanish at the end of
every period too, so the robot starts and stops at rest and trajectories can be
repeated or averaged over periods.

Candidates are scaled to the largest amplitude that respects joint position
limits, URDF velocity limits and an acceleration cap, then scored by the
condition number of the base-parameter regressor and checked for collision.
On this asset the optimized result conditions the regressor about 1.9x better
than the existing random point-to-point generator (≈145 against ≈277).

### The modal probe

The main harmonics excite the rigid-body dynamics but leave essentially no
energy above ~0.5 Hz, so a sampled robot's transmission mode (tens to
hundreds of Hz) never gets excited and its damping ratio has no signature in
the data at all — only the static deflection `tau / k` is visible. A small
extra comb of high-frequency harmonics, `excitation.probe_harmonics`, fixes
this: 40 log-spaced lines from 8-190 Hz by default
(`log_spaced_probe_harmonics(8.0, 190.0, 40, 0.1)`), given
`excitation.probe_acceleration_fraction` (default 0.2) of the acceleration
budget, scaled *before* the main harmonics so it is never crowded out by
whichever limit binds the main trajectory (position or velocity, on most
proximal joints). Conditioning is still scored on the main harmonics only —
the probe would alias the regressor's stride-10 sampling and is not part of
what the rigid-body regressor needs to be well conditioned.

The probe band starts at 8 Hz, not the structural mode's own open-loop
frequency, because of the closed-loop finding below: on a heavy joint the
*observable* resonance sits well below the open-loop
`sqrt(k / J_eff)` prediction, and a probe that only reached down to 30 Hz (an
earlier version of this feature) missed it.

A trajectory's samples are evaluated **analytically** from its Fourier
coefficients (`MaterializedTrajectory.analytic`), not interpolated from the
recorded grid: linearly interpolating a signal that has real content above
~100 Hz on a 2 ms grid attenuates it (`sinc²`, ~40% down at 190 Hz) and
folds an image back in near `500 - f` Hz, which for this asset's fastest
mode (A7, ~600-900 Hz open-loop) landed on top of it. `optimize_excitation`
and `trajectory_from_metadata` both populate the analytic evaluator; only a
trajectory loaded from a plain saved JSON (no coefficients) falls back to
interpolation.

### The closed-loop finding: what a rollout actually shows is not the open-loop mode

`ComputedTorqueController`/`SeaMotorController` compute `tau = rnea(q, dq,
ddq_ref + kd·e_dot + kp·e) + friction(dq)`. Because `rnea` is affine in its
acceleration argument, this is exactly `tau = h(q, dq) + M(q)(ddq_ref + kd·e_dot
+ kp·e) + friction(dq)`: the feedback term contributes an *extra* motor-side
stiffness/damping of `(M_ii + J_rotor) kp` / `(M_ii + J_rotor) kd`, on top of
the transmission's own `(k, d)`. On a heavy joint (A1-A2 on this arm) that
motor-side damper is roughly ten times the transmission's own at the shipped
gains, and it dominates what the deflection channel actually shows: the
*observable* resonance is a closed-loop quantity, shifted well below the
open-loop `sqrt(k / J_eff)` `TransmissionSpec.natural_frequency()` predicts,
and — because `M_ii` varies with the arm's configuration along the
trajectory — is not even a single fixed frequency.
`TransmissionSpec.closed_loop_mode_frequency` models this (the deflection's
frequency-response peak under a linearized single-joint model, evaluated at
a given link inertia); it is close enough to a direct simulation to use as a
diagnostic, but it ignores cross-joint coupling, gravity and the true `M(q)`
trajectory, so nothing in this pipeline asserts against it directly —
`tests/test_identification_dataset.py`'s mode/ζ tests use it only to choose
*where to look*, and assert a contrast (probe on vs. off), not a match to
its point prediction.

One practical consequence: **ζ (transmission damping ratio) is only weakly
observable on heavy joints, by construction, no matter what the probe does**
— the motor loop's own damping swamps it. `damping_ratio` sampling is left
un-stratified for exactly this reason (see below). This is demonstrated, not
just asserted: with the shipped probe, ζ is clearly observable on at least
one mid-weight joint (A3 on this arm, >8x RMS contrast on/off) and provably
unobservable without the probe (<1e-3 relative difference); it stays weak on
the two heaviest joints (A1/A2) as this finding predicts, and came out
genuinely ambiguous on one further joint (A4) under two different
measurement methodologies — not confidently either way, and left as such
rather than tuned toward a conclusion.

`simulation.control_gains` (disabled by default at the dataclass level,
enabled in the shipped YAML) randomizes `natural_frequency`/`damping_ratio`
**per trajectory** instead of using one fixed gain for the whole dataset,
recorded per bag as `control_natural_frequency`/`control_damping_ratio`.
Without it, every bag shares the exact same closed-loop behaviour above, so
a model trained on the dataset implicitly bakes in one specific control
loop; varying it is what makes that dependency visible in the data instead
of a hidden confound. `run_identification_simulation.py` derives the same
per-trajectory draw as `generate()`, so `--tier eNN --trajectory k`
reproduces it, unless `--control-frequency` explicitly overrides it.

## Tiers and sampled robots

A tier is a model-fidelity level. The **rigid** tier uses the URDF's own
inertials plus its declared `damping = 10.0 Nm s/rad` and `friction = 0.1 Nm`.
The other tiers are **sampled elastic robots** `e00, e01, …`: the same arm with
a series-elastic transmission at every joint, payload 0.

Elastic robots exist because a rigid dataset cannot identify elastic
parameters: with `q_motor ≡ q_link` the elastic potential has no observable
effect, and the link-side target would equal the motor-side input, degenerating
the learning task.

### Sampling

The stiffness and rotor-inertia intervals are not estimates of the true
values. Each is the support of a prior over a parameter the nominal model
cannot observe, and the dataset is a domain-randomization ensemble drawn
from that prior. An interval is therefore judged by two criteria only: does
it contain the true value with high confidence, and is it no wider than the
physically admissible range? Neither criterion requires knowing the true
value, which is what makes the framing defensible without a measurement in
hand. See `docs/PARAMETER_PROVENANCE.md` for the full argument, the anchors
that ground each nominal, and why the multiplicative factor is `2` (not a
guess -- it is the documented reproducibility of a harmonic-drive stiffness
measurement).

Configured in the `transmission` block of the YAML:

- **Stiffness** — `stiffness_nominal` (one value per joint, with provenance
  recorded in `docs/PARAMETER_PROVENANCE.md`) times a log-uniform
  `stiffness_factor`, `k_j = nominal_j * exp(U(-ln r, +ln r))`. A
  `stiffness_common_fraction` splits that log-uncertainty between one
  arm-wide scale (correlated joints, as a real gearbox family would produce)
  and independent per-joint scatter; `0` keeps every joint independent and
  maximizes marginal coverage, `1` gives a one-dimensional family. The legacy
  `stiffness: [[min, max], ...]` interval form is still accepted (with a
  deprecation warning) and maps directly to a per-joint log-uniform draw.
- **Sampling method** (`transmission.sampling`) — `iid` (independent per
  robot; a robot depends only on the seed and its index, so raising `robots`
  leaves existing robots unchanged, but six-ish i.i.d. draws visit only
  15–46 % of a wide declared interval), `stratified` (one draw per
  equal-width stratum in log space, permuted independently per joint — full
  marginal coverage at the same robot count, but redefines the whole set
  when `robots` changes) or `sobol` (a scrambled Sobol sequence — full
  coverage *and* prefix-stable, needs `scipy.stats.qmc`). The shipped config
  uses `stratified` with `robots: 20`, giving ≥ 0.9 per-joint coverage versus
  ~0.4–0.7 at the historical 6 i.i.d. robots.
- **Damping ratio** — one interval, sampled uniformly per joint (default
  0.05–0.2, the lightly damped range of a geared joint). Left un-stratified
  deliberately: it is unobservable in the data at all without the excitation
  modal probe (see "The modal probe" below), and even with the probe on it
  is only clearly observable on some joints — the motor control loop's own
  damping dominates the transmission's on the heaviest ones, a closed-loop
  effect explained under "The closed-loop finding" below — so widening its
  coverage would add label noise on the joints where it cannot be identified
  from this data regardless.
- **Damping coefficient** — never configured. It is derived as
  `d = 2 ζ sqrt(k J_eff)`.
- **Rotor inertia** — `rotor_inertia_nominal x rotor_inertia_factor`, sampled
  exactly like stiffness but from its own stream, **independent** of the
  stiffness draw: `k` and `J_rotor` come from different physical components
  and correlating them would hide the `sqrt(k / J_eff)` mode-frequency
  degeneracy rather than cover it. Give only `rotor_inertia` (a fixed value
  or one per joint) to keep the historical unsampled behaviour.

Every dataset build prints a per-joint, per-parameter coverage report
(declared vs. realized log span) and warns below 80 % coverage; it is also
written into the manifest as `sampling_coverage`.

Robots are drawn from their own random stream in index order when
`sampling: iid`, so a robot depends only on the dataset seed and its index:
raising `robots` adds robots without changing existing ones, and
`run_identification_simulation.py --tier e03` reproduces the dataset's `e03`.
`stratified` sampling does not have this property — it is defined by the
whole batch, so changing `robots` redefines every robot; pass
`--manifest <dataset>.manifest.json` alongside `--tier` to fail loudly rather
than silently reproducing the wrong robot when the two disagree. Rigid and
elastic bags interleave in the file (see below).

### Why the link inertia matters

A transmission mode is a rotor oscillating against a link through the spring,
at `sqrt(k / J_eff)` with `J_eff = J_rotor J_link / (J_rotor + J_link)`. The
link-side inertia `M_ii` is sampled over the joint range with Pinocchio: on
this arm it spans about 2–3 kg m² at A1–A2 down to **3e-4 kg m² at A7**, where
the link is far lighter than the rotor and sets the mode alone. The median
`M_ii` sizes the damping; the minimum bounds the highest mode, and so the
integration step.

The earlier rotor-only estimate put every joint's mode at `sqrt(k / 0.1)`: at
`k = 1e6` it predicted 503 Hz where the wrist mode is 9.1 kHz, so the step it
chose could not resolve the wrist.

With the default intervals the modes sit at about 50–170 Hz for A1–A6 and
600–900 Hz for A7, giving a step of about 6–8e-5 s. The dataset is sampled at
2 ms (Nyquist 250 Hz): proximal modes are visible in the data, the wrist mode
only through its quasi-static deflection `tau / k`.

Two details are easy to get wrong and are handled explicitly:

- **Rotor inertia is applied as joint armature**, not as body inertia. A geared
  rotor adds inertia about its own joint axis only; writing it into the
  fictional transmission body's inertia tensor loads every downstream joint
  with it and changes the arm being identified.
- **The motor joint of a serial chain carries all downstream inertia**, not just
  its rotor. Sizing the motor-side controller by the rotor inertia alone
  under-estimates the true inertia by roughly fifty times on this arm and drives
  the motor into a large-amplitude oscillation against its torque limit.

`TransmissionSpec.require_stable_step` refuses a time step that cannot resolve
the transmission mode and reports the step actually needed, rather than
producing noise that looks like physics.

Measured near-rigid convergence (MuJoCo, 2 s excitation, uniform stiffness):

| stiffness [Nm/rad] | max deflection [rad] | RMS vs rigid [rad] |
|---|---|---|
| 1e6 | 5.5e-5 | 2.3e-5 |
| 2e5 | 2.7e-4 | 7.1e-5 |
| 4e4 | 1.4e-3 | 3.3e-4 |
| 1e4 | 6.2e-3 | 1.3e-3 |

### Friction is not a useful axis of variation

`friction_samples` defaults to 1, and raising it does not add information.
The controller compensates friction exactly and the runner subtracts the same
friction from the plant, so it cancels: two friction samples of one condition
produce bit-identical motion and a bit-identical link-side target, differing
only in the recorded motor torque. For a model learning `ft` from
`(q, dq, tau)` that is a contradiction, not extra data. Vary robots and
trajectories instead.

A friction sample would only become a real condition if the controller were
given a *different* friction model from the plant, which is a deliberate
model-mismatch experiment rather than a dataset axis.

This cancellation is exact regardless of `control_decimation`: holding the
whole solver torque (command and the friction subtraction together) constant
across a decimation window is what keeps the cancellation exact and the
rollout numerically stable (see `torque_runners.run_mujoco_torque`'s comment
at the decimation gate) — the plant never physically experiences friction
either way, decimated or not.

### What the payload-free wrist can teach, and why the dataset now fits one

The learning target is the link-side torque. With no tool fitted, the link
past the A7 transmission is the bare flange (`M_77 = 3e-4 kg m^2`), so its
link-side torque is essentially zero: measured RMS per joint is about
`[1.3, 31, 1.7, 9.7, 0.2, 0.2, 0.0] Nm`, while the *motor* torque at those
joints is 4-7 Nm of friction and rotor inertia. The distal channels therefore
carry almost no signal, and per-channel normalization will amplify their
noise. Only a payload or tool changes this; larger accelerations do not.

There is a second, deeper reason to fit one. The link-side map `(q, dq, ddq)
-> ft` is *literally the URDF's own* `rnea`, identical for every sampled
robot regardless of stiffness -- only `stiffness`, `damping` and
`rotor_inertia` differ, and none of them appear on the right-hand side of
that equation. Randomizing the transmission therefore changes *which states
get visited* (a covariate shift) but not the function relating them to `ft`.
A payload changes `M(q)`, `C(q, dq)` and `g(q)` themselves, so the link-side
dynamics genuinely differ robot to robot -- which stiffness alone does not
give a link-side-only model.

`payload.enabled: true` in the YAML fits a randomized uniform-box tool at the
flange (mass, offset and size ranges configurable), injected directly into
the URDF (see `src/elastic_sim/payload.py`) so every consumer of the asset --
Pinocchio, both simulators, the collision checker -- sees the same robot by
construction; MuJoCo-only `body_overrides` would de-tune the controller
instead. Which target a given model family actually needs depends on what it
consumes:

| Map | Inputs | Target | Depends on stiffness? |
|---|---|---|---|
| A | `q, dq, ddq` (link) | `ft` | No -- see above |
| B | `q, dq, ddq, tau` | `ft` | Yes, via `tau - ft` |
| C | `q_motor, dq_motor, tau` | `defl = q_motor - q_link` | Yes, directly |

Map C (the `defl0..defl{n-1}` columns) is the cleanest target for
identifying the transmission parameters themselves: its magnitude is
`tau / k`, a direct readout of the parameter being randomized.

A payload adds real geometry, so a trajectory validated collision-free
against the bare asset is not necessarily collision-free once one is
fitted. With `dataset.trajectories_per_robot: true` (the shipped default),
each robot's payload is known before its trajectory is searched, so
candidates are scored against *payload-fitted* collision geometry directly
— a collision just rejects a candidate, the same as any other infeasible
one. `resolve_bag` (`src/elastic_sim/dataset.py`) is the single place this
derivation happens: `generate()` and `run_identification_simulation.py`
both call it, so `--tier eNN --trajectory k` reproduces the exact same
payload-fitted trajectory *and* runs its rollout with that payload, not the
bare asset. With `trajectories_per_robot: false` (one trajectory shared
across every tier), no single payload applies during the search, so that
path stays payload-free and a post-hoc check re-draws an alternative
payload — rather than aborting the whole build — if the shared trajectory
and a bag's drawn payload turn out to collide.

## Backend comparison

When a dataset runs more than one backend, every pair of bags that differs only
by backend (same trajectory, tier and friction sample) is compared on the common
output grid. Per joint:

| metric | meaning |
|---|---|
| `q_link_rms`, `q_link_max` | link position difference [rad] |
| `dq_link_rms`, `q_motor_rms` | link velocity and motor position difference |
| `ft_relative_rms`, `tau_relative_rms` | link and motor torque difference over that torque's RMS |
| `deflection_relative_rms` | difference of `q_motor − q_link` over its RMS (elastic only) |

A pair passes when its worst joint is within the `comparison` limits in the
YAML (defaults `q_link_rms 1e-4 rad`, `ft_relative_rms 1 %`,
`deflection_relative_rms 5 %`). Failing pairs are **flagged, never dropped**:
the table is printed at the end of generation, written to
`<dataset>.backend_comparison.csv` and summarized in the manifest.

The torque is closed-loop, so a state difference feeds back into the recorded
label; that is why torques are compared as well as states. The `independent`
column is `True` only where Newton ran its own solver (rigid, Featherstone).
On elastic bags it ran `SolverMuJoCo`, and the table marks these pairs
`(same engine)`.

`scripts/compare_identification_backends.py` runs the same comparison on a
written dataset, with limits overridable on the command line.

Measured on a 2 s smoke run: rigid pair `8.8e-8 rad`; elastic pairs
`1.8e-6 rad`, link torque `0.29 %`.

## Output contract

One row per sample, consumed directly by `dynamic_model_nn`'s `CustomDataset`.

| Column | Meaning |
|---|---|
| `t` | strictly increasing within a bag, uniform step |
| `bag` | one trajectory under one condition |
| `q0..q{n-1}` | joint position, on the side `dataset.signals.position_side` names (default `link`) [rad] |
| `dq0..dq{n-1}` | joint velocity, same side [rad/s] |
| `tau0..tau{n-1}` | commanded **motor-side** torque [Nm] — model input |
| `ft0..ft{n-1}` | training target, on the side `dataset.signals.target` names (default `link_torque`) [Nm] |
| `q_motor*`, `dq_motor*`, `q_link*`, `dq_link*` | ingested, available to elastic models |
| `defl0..defl{n-1}` | `q_motor - q_link` [rad] — Map C target, direct readout of `tau / k` |
| `split` | `"train"`, `"val"` or `"test"`, one label per bag (see below) |
| `payload_mass`, `payload_offset_x/y/z`, `payload_size` | the fitted tool, constant within a bag, `NaN` on the rigid tier |
| `exc_base_frequency`, `exc_max_acceleration`, `exc_velocity_fraction`, `exc_probe_top_hz` | this bag's realized excitation regime (see "Per-trajectory regime randomization" below); constant within a bag |
| `control_natural_frequency`, `control_damping_ratio` | this bag's realized closed-loop gain (see "The closed-loop finding" above); constant within a bag, equal to `simulation.control_frequency`/`control_damping_ratio` unless `simulation.control_gains.enabled` |
| `control_position_gain`, `control_velocity_bandwidth`, `control_integral_time` | this bag's realized velocity-loop gains, when `simulation.controller.mode: velocity_pi`; held at the spec's own value under every other mode so a cross-mode comparison reads one schema |
| `*_clean{i}` (`q_link_clean*`, `tau_link_clean*`, `q_ref_clean*`, …) | the exact simulator values on the same grid, written only when `dataset.signals.clean_columns` is set; debug/diagnostic columns, never model inputs |
| `gain_motor__<joint>`, `gain_link__<joint>` | this bag's torque calibration errors, written only when a sensor model is configured |
| `link_viscous__<joint>`, `link_coulomb__<joint>`, `ripple_amplitude`, `ripple_order` | this bag's link-side plant extras, written only when configured |
| `tier`, `backend`, `experiment`, `viscous__<joint>`, `coulomb__<joint>`, `stiffness__<joint>`, `damping__<joint>`, `damping_ratio__<joint>`, `rotor_inertia__<joint>` | metadata, ignored by the loader; transmission columns are empty on rigid bags |

Per bag, the manifest's `records` entries additionally carry
`trajectory_digest` and `trajectory_signal_digest` (a hash of the
trajectory's time/position/velocity/acceleration samples only, excluding
diagnostic fields like the regressor condition number that agree only to
basis precision even for an identical trajectory — use this one for
reproducibility checks, not `trajectory_digest`), `condition_number` and
`peak_torque_ratio` (warned above 0.8 of the joint's effort limit).

`ddq` is deliberately **not** emitted: the consumer always recomputes it with a
Savitzky-Golay filter and ignores the column. That filter requires a uniform
time step *within* a bag (not across bags — see the excitation regime
randomization below), so every bag is resampled onto a common grid regardless
of the step its tier needed, and `generate()` asserts every bag's `t` column
is uniform to `1e-12` before writing anything.

In the rigid tier `tau` and `ft` are identical, because a rigid chain has no
transmission compliance. That tier is meant for validating the pipeline against
the analytic model, not for training.

Bags are written with trajectory varying slowest and backend fastest, so
consecutive bags differ by condition rather than by regime; that ordering is
independent of `split`, which is assigned explicitly per bag (see below) and
does not depend on file position.

### Round 5: which signals, which controller, which instrument

Four config blocks decide what a dataset *means*, independently of the robot.
Every one of them defaults to round 4's behaviour, so an existing config keeps
producing the dataset it produced before.

| Block | What it chooses | Default |
|---|---|---|
| `dataset.signals` | which side of the spring `q0..`/`ft0..` are measured on | `link` / `link_torque` (collocated) |
| `simulation.controller` | `exact_ct`, `nominal_ct`, `pd_gravity`, `pd` or `velocity_pi` | `exact_ct` |
| `simulation.measurement` | encoder quantization, noise, torque gain error, delay | perfect instruments |
| `simulation.plant_extras` | link-side friction, a nonlinear spring, torque ripple | none (a purely Lagrangian link side) |

**The signal pair is the load-bearing one.** A dataset that pairs *link*
position with *link* torque carries no elastic signature at all, however soft
the robot is: the link equation gives `tau_s = M(q) qdd + c + g` exactly, with
the spring nowhere in it. Elasticity only appears when position and torque are
read on opposite sides of the transmission — motor position with link torque,
which is what the round-5 configs (`fmrr_tecnobody.yaml`,
`ur10_table_round5.yaml`) set and what the three real platforms actually
expose. Every dataset records the choice in its `.contract.json`, with a
`collocated` flag, so a consumer comparing an elastic model class against a
residual one can tell whether the comparison is meaningful before training
anything.

**The controller decides what `tau_cmd` is worth.** Exact computed torque
knows the plant, including its payload, so the commanded torque, the state and
the target are all near-exact functions of the same reference: many different
decompositions fit equally well and which one training lands on is arbitrary.
The other four modes break that on purpose; `velocity_pi` — a model-free outer
position loop over a PI velocity loop whose gains are drawn per bag — is the
one that has a counterpart on all three platforms.

`scripts/diagnose_controller_modes.py` measures the difference without
training anything: the collinearity of `tau_cmd` with the state regressor, the
conditioning of the achieved motion, the decomposition of what the nominal
rigid model leaves unexplained, probe-band content, and the deflection's
signal-to-noise. Run it before generating a production dataset with a new
controller or a new probe band.

```bash
uv run python scripts/diagnose_controller_modes.py --generate     --config config/identification/fmrr_tecnobody.yaml     --modes exact_ct velocity_pi --trajectories 2 --robots 3     --out reports/round5/qc_fmrr
```

Plant extras are rejected on the **rigid reference tier**: that tier's torque
has to keep satisfying `rnea(achieved state)` for the Pinocchio cross-check and
the MuJoCo/Newton-Featherstone comparison to mean anything.

### Other assets: FMRR (3-axis Cartesian gantry)

`config/identification/fmrr_tecnobody.yaml` runs the same stack on the
Tecnobody FMRR platform. Three things differ structurally:

- **Units are metres and newtons** — every axis is prismatic, so "stiffness"
  is N/m and "rotor inertia" a reflected mass in kg.
- **No table**: FMRR is its own gantry scene, so `--asset fmrr_tecnobody`
  needs no `compose_scene_urdf.py` step.
- **The mass matrix is constant and diagonal** (2.0, 1.2, 1.0 kg) and gravity
  loads the z axis alone, which makes it the family's analytic test case.

Two generic features were needed to bring it in, both no-ops elsewhere: the
arithmetic-only xacro subset is expanded before Pinocchio sees the URDF, and
every 1-DoF joint the asset does not declare active (FMRR's `joint_yaw`) is
locked to `fixed` in all three consumers — Pinocchio, MuJoCo and Newton — so
they simulate the same robot. Its `active_joints` are declared in the URDF's
kinematic-chain order `[joint_y, joint_x, joint_z]`, which Pinocchio builds
and cannot permute.

### Splits

`dataset.split` in the YAML (`SplitPolicy` in `src/elastic_sim/dataset.py`)
declares how tiers are assigned to `train` / `val` / `test`, written into the
`split` column and into the manifest's top-level `split` block:

- `contiguous` (default) assigns every bag `"train"` — the historical
  behaviour, where a consumer splitting a contiguous half of the file puts
  every robot in both halves and can only measure generalization to a new
  *trajectory*, not a new *transmission*.
- `holdout_robots` reserves whole robots for `val`/`test`, chosen as the
  softest and stiffest strata by mean log stiffness rather than the last *N*
  by index, so the held-out set is a measured extrapolation margin. This is
  the only split that can show whether a model generalizes to an unseen
  stiffness, which is the entire point of randomizing it.

`declared_split_loaders` in `dynamic_model_nn/dataset.py` honours the `split`
column when present and falls back to the historical contiguous 50/50 split
otherwise, so old CSVs load unchanged.

### Consumer requirement

`dynamic_model_nn`'s `dataset.py` accepts a general `ft0..ft{dof-1}` block in
`CustomDataset._load_dataframe`, checked before the fixed six-channel wrench
form so a 7-channel per-joint target is not silently truncated to six. At
`dof == 6` the column names `ft0..ft5` are ambiguous on their own — a
genuinely 6-DoF arm's per-joint torque and a legacy end-effector wrench use
identical names for different quantities — so `CustomDataset` reads the
`<dataset>.contract.json` sidecar (every generated dataset gets one next to
the CSV, stating `n_dof`, the input and target column ranges and the target
semantics) and trusts its `target_columns` over the `dof == 6`-means-wrench
guess when the sidecar is present; a CSV with no sidecar keeps that
historical guess. Read the sidecar before wiring up a new training script.

### Normalization erases magnitude — read this before training

`CustomDataset._normalize`'s optional per-bag mode (`stats_by_bag`) subtracts
each bag's own mean and divides by its own standard deviation. Applied to the
target, this removes each bag's *scale* — exactly the axis that differs
between a soft and a stiff sampled robot — so a model sees the same
normalized shape regardless of which robot generated it. It also amplifies
any channel whose true RMS is near zero (see "What the payload-free wrist can
teach" above) to unit-variance noise. **Use global normalization for the
target**; per-bag normalization for the *inputs* only, if at all.

## Generation cost and storage

The rollout is Python- and RNEA-bound, not physics-bound (`R3_06`), so three
knobs pay for themselves once the robot count rises past ~25:

- `simulation.control_decimation: N` evaluates the controller every `N`
  physics steps and holds the torque between updates (zero-order hold, as a
  real drive does). The physics step is set by the transmission mode
  (~640 Hz-16 kHz), not the controller's ~4 Hz closed-loop bandwidth, so this
  recovers most of that gap as speed with no change to what is recorded.
  Left at `1` by default; raising it needs the feedback-ratio and
  rigid-vs-Pinocchio-residual checks re-verified at the chosen value.
- `--jobs N` on `generate_identification_dataset.py` parallelizes the bag
  loop with a process pool (forced to 1 under `--visualize`). Bags are fully
  independent and their RNG streams are keyed on `(seed, index)`, not on
  execution order, so `--jobs 4` produces a bit-identical dataset to
  `--jobs 1`.
- `dataset.output` ending in `.parquet` instead of `.csv` writes typed,
  compressed, float32 columns (~10x smaller); `dataset.metadata_columns:
  sidecar` moves the ~42 per-bag-constant metadata columns to a
  `<dataset>.bag_metadata.json` keyed by bag instead of repeating them on
  every row (~40% smaller). Both default to the historical inline-CSV
  behaviour.

## Verification

```bash
uv run pytest tests/test_identification_dataset.py -q
```

The suite locks in: MuJoCo/Pinocchio agreement to 1e-6; the transmission mode
using the rotor/link reduced inertia and damping reproducing the sampled ratio;
robot sampling staying inside its intervals, log-uniform and seed-stable; the
backend comparison pairing bags and flagging a divergent backend; the regressor
reproducing inverse dynamics; 43 rigid and 57 friction-augmented base
parameters; excitation endpoint conditions and limit compliance; the optimized
trajectory beating the point-to-point baseline; the recorded torque being the
applied torque; feedback staying under 5 % of feedforward; Newton and MuJoCo
agreeing to 1e-4 rad; the near-rigid limit converging; and — the acceptance
test — **base parameters recovered from a generated rollout to better than 1e-4
relative error**.
