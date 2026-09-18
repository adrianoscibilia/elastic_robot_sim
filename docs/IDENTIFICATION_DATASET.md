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
    --trajectories 8 --friction-samples 3 \
    --output data/identification/iiwa14_table.csv

# 3. Re-check MuJoCo/Newton agreement of a written dataset, no simulation.
uv run python scripts/compare_identification_backends.py \
    data/identification/kuka_lbr_iiwa_14_r820_table.csv
```

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
candidates, condition number ≈150); `--base-frequency 0.5 --candidates 8` runs
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

Configured in the `transmission` block of the YAML:

- **Stiffness** — one `[min, max]` interval per joint, sampled log-uniformly
  (as the original FMRR generator did), so a wide interval is not dominated by
  its stiff end. The defaults are order-of-magnitude values for a harmonic
  drive in series with a joint torque sensor, stiffer at the base than at the
  wrist: A1–A2 1.5e4–3.5e4, A3–A4 1e4–2.5e4, A5 5e3–1.5e4, A6–A7 3e3–1e4
  Nm/rad. They are not datasheet values; refine them when measurements exist.
- **Damping ratio** — one interval, sampled uniformly per joint (default
  0.05–0.2, the lightly damped range of a geared joint).
- **Damping coefficient** — never configured. It is derived as
  `d = 2 ζ sqrt(k J_eff)`.
- **Rotor inertia** — reflected rotor inertia per joint, applied as armature;
  fixed, not sampled.

Robots are drawn from their own random stream in index order, so a robot
depends only on the dataset seed and its index: raising `robots` adds robots
without changing existing ones, and `run_identification_simulation.py --tier
e03` reproduces the dataset's `e03`. Rigid and elastic bags interleave in the
file (see below).

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
| `q0..q{n-1}` | link-side joint position [rad] |
| `dq0..dq{n-1}` | link-side joint velocity [rad/s] |
| `tau0..tau{n-1}` | applied **motor-side** torque [Nm] — model input |
| `ft0..ft{n-1}` | **link-side** torque [Nm] — training target |
| `q_motor*`, `dq_motor*`, `q_link*`, `dq_link*` | ingested, available to elastic models |
| `tier`, `backend`, `experiment`, `viscous__<joint>`, `coulomb__<joint>`, `stiffness__<joint>`, `damping__<joint>`, `damping_ratio__<joint>`, `rotor_inertia__<joint>` | metadata, ignored by the loader; transmission columns are empty on rigid bags |

`ddq` is deliberately **not** emitted: the consumer always recomputes it with a
Savitzky-Golay filter and ignores the column. That filter requires a uniform
time step, so every bag is resampled onto a common grid regardless of the step
its tier needed.

In the rigid tier `tau` and `ft` are identical, because a rigid chain has no
transmission compliance. That tier is meant for validating the pipeline against
the analytic model, not for training.

Bags are written with trajectory varying slowest and backend fastest.
`dynamic_model_nn` splits train/test as a *contiguous half* of the file, so a
dataset ordered by tier would put whole tiers on one side of that split.

### Consumer requirement

`dynamic_model_nn` originally accepted a target of only three or six channels
while its models emit one per joint, so a 7-DoF arm failed in the loss. Its
`dataset.py` now accepts a general `ft0..ft{dof-1}` block, checked before the
fixed six-channel form so a 7-channel target is not silently truncated.

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
