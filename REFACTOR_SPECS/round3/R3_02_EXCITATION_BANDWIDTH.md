# R3_02 — Excitation bandwidth and dynamic-regime variability

Addresses findings **F3** (the excitation never reaches the transmission modes, so
`zeta` is unobservable and `k` is only quasi-static) and **F6** (all 60 trajectories
sit at one point in excitation-hyperparameter space).

---

## 1. The bandwidth gap

### 1.1 Numbers

`FourierExcitationConfig` with the shipped defaults:

```
n_harmonics: 5, base_frequency: 0.1 Hz, n_periods: 1
```

The series is `q_i(t) = q_i0 + sum_{k=1..5} [ a_ik/(k w) sin(k w t) - b_ik/(k w) cos(k w t) ]`
with `w = 2 pi * 0.1`. Its spectral support is exactly the five lines
`{0.1, 0.2, 0.3, 0.4, 0.5} Hz`. There is no energy anywhere else — this is a
pure-tone comb, not a broadband signal.

The transmission modes, from `TransmissionSpec.natural_frequency()` with the
sampled robots and `J_eff = J_rotor J_link / (J_rotor + J_link)`:

| Joints | Mode frequency |
|---|---|
| A1-A6 | ~50-170 Hz |
| A7 (`M_77 = 3e-4 kg m²`) | ~640-900 Hz |

The ratio between the highest excitation line and the lowest transmission mode is
about **100**. Between 0.5 Hz and 50 Hz the dataset contains zero energy.

### 1.2 The two consequences

**(a) `zeta` is unobservable.** A second-order mode's damping ratio manifests in the
decay of its resonance. With no energy at or near the resonance, the transmission
never rings, so nothing in `q`, `dq`, `tau` or `ft` responds to `zeta`. The seven
uniformly-sampled `damping_ratio` values per robot are, from the model's point of
view, **label noise**: two robots differing only in `zeta` produce essentially
identical data with different metadata.

This is not speculative — it is the same argument `docs/IDENTIFICATION_DATASET.md`
already makes correctly for friction ("the controller compensates friction exactly
and the runner subtracts the same friction from the plant, so it cancels ... that is
a contradiction, not extra data"). The `zeta` case is the same failure mode by a
different route: not cancellation, but non-excitation.

**(b) `k` is visible only quasi-statically.** Below the mode, the transmission is a
pure spring: `q_motor - q_link = tau / k`. That *is* observable — at A2 with
`tau ~ 31 Nm` and `k ~ 2e4` the deflection is ~1.5e-3 rad, which is far above
numerical noise and is recorded in `q_motor*`/`q_link*`. So stiffness is
identifiable, but only as a **static compliance**, and every robot is
indistinguishable from a rigid robot plus a torque-proportional offset. Nothing in
the dataset distinguishes a series-elastic transmission from a static compliance
model.

### 1.3 Why this matters for the stated goal

The project's framing is "elastic transmission modelling for dataset generation"
with stiffness randomized because it is non-observable from the URDF. If the
generated data can only ever reveal the *static* part of that transmission, then:

- the dataset supports learning a compliance-augmented rigid model;
- it does **not** support learning or validating a flexible-joint dynamic model,
  which is the thing the `*_elastic` model family in `dynamic_model_nn` is for;
- and a reviewer who asks "what in your data would change if I halved every damping
  ratio?" has no answer.

Fixing the bandwidth is what turns the second and third bullets around.

---

## 2. Fix: a superimposed high-frequency probe

### 2.1 Design constraints

- The output grid is `sample_time_step 0.002 s` → **Nyquist 250 Hz**. Content above
  250 Hz cannot be represented in the written dataset, so aliasing must be avoided:
  the probe's top frequency must sit below ~200 Hz with margin.
- A1-A6 modes (50-170 Hz) are therefore **inside** the representable band and can be
  excited directly.
- A7's mode (640-900 Hz) is **not** representable at 2 ms. Two honest options:
  (i) leave A7 quasi-static and say so; (ii) fit a payload (`R3_01`), which raises
  `J_link` at the wrist and drops that mode into the representable band. Option (ii)
  is another reason to do `R3_01` first.
- The probe must not break the guarantees the excitation module currently provides:
  start and end at rest, stay inside the safe joint band, respect velocity and
  acceleration limits, stay collision-free.
- The probe must not wreck the regressor conditioning that the existing search
  optimizes for.

### 2.2 Shape

Add a second, small-amplitude, high-frequency component to the reference:

```
q_i(t) = q_i^fourier(t) + eps_i(t)
eps_i(t) = A_i * sum_{m} c_im / (2 pi f_m) * sin(2 pi f_m t + phi_im)
```

with the probe frequencies `f_m` drawn on a log grid inside
`[probe.f_min, probe.f_max]` and the amplitude `A_i` set so the probe's **position**
contribution is a small fraction of the main trajectory's, e.g. 1e-3 rad peak.

Key point: the probe's **acceleration** contribution is `A_i (2 pi f_m)`, which at
100 Hz is 628x the position amplitude per unit — so a 1e-3 rad probe costs
~0.6 rad/s² of acceleration budget at 100 Hz, and ~4 rad/s² at 700 Hz. Size the
probe from the *acceleration* budget, not the position budget, and take the
remaining acceleration headroom explicitly:

```python
probe_acceleration_budget = probe.acceleration_fraction * config.max_acceleration
```

with `probe.acceleration_fraction` defaulting to `0.2`, i.e. the main trajectory is
re-scaled to `0.8 * max_acceleration` and the probe gets the other 20 %.

### 2.3 Periodicity and rest conditions

Keep the existing guarantees by **quantizing the probe frequencies to integer
multiples of the base frequency**:

```python
f_m = round(f_m_target / base_frequency) * base_frequency
```

With `base_frequency 0.1 Hz`, that is a 0.1 Hz grid — fine enough that quantization
is irrelevant. The probe is then just extra harmonics of the same series, so
`project_coefficients` applies unchanged and the "starts and ends at rest,
repeatable, averagable over periods" properties are preserved exactly.

**This is the implementation that should be written**: do not add a separate
additive signal path. Instead, allow `FourierExcitationConfig.n_harmonics` to be a
*set* of harmonic indices rather than a count, and let the probe be high-index
harmonics with their own amplitude scaling.

```python
@dataclass(frozen=True)
class FourierExcitationConfig:
    ...
    # Harmonic indices carrying the main excitation (1..n_harmonics as today).
    n_harmonics: int = 5
    # Extra high-index harmonics used as a small-amplitude modal probe.  Indices
    # are multiples of base_frequency, so the rest/periodicity conditions and
    # project_coefficients apply to them unchanged.  Empty disables the probe.
    probe_harmonics: tuple[int, ...] = ()
    # Fraction of max_acceleration reserved for the probe.
    probe_acceleration_fraction: float = 0.2
```

For `base_frequency 0.1 Hz`, a probe covering 40-160 Hz is
`probe_harmonics = (400, 700, 1000, 1300, 1600)`.

`evaluate_series` already handles arbitrary harmonic counts; extend it to take an
explicit index vector instead of `arange(1, H+1)`:

```python
def evaluate_series(a, b, offset, omega, time, indices=None):
    harmonics = np.arange(1, a.shape[1] + 1, dtype=float) if indices is None \
                else np.asarray(indices, dtype=float)
    ...
```

This is a strictly backward-compatible change — every existing call passes
`indices=None`.

`project_coefficients` also needs the index vector (it projects `b` off the
harmonic-index direction):

```python
def project_coefficients(a, b, indices=None):
    harmonics = np.arange(1, a.shape[1] + 1, dtype=float) if indices is None \
                else np.asarray(indices, dtype=float)
```

### 2.4 Amplitude allocation in `_fit_to_limits`

`_fit_to_limits` currently computes **one** scale per joint from the tightest of
position / velocity / acceleration. With a probe, run it **twice**:

1. Scale the main harmonics against `(half_span, velocity_limit,
   (1 - probe_acceleration_fraction) * max_acceleration)`.
2. Scale the probe harmonics against the **remaining** budget: the leftover
   position span, the leftover velocity, and
   `probe_acceleration_fraction * max_acceleration`.

Both are still exact one-pass scalings because every quantity is linear in the
coefficients. Compute the offset from the **combined** signal, as today.

### 2.5 Integration step

The probe does not change `TransmissionSpec.required_time_step` — that is set by the
mode, not by the reference. But it **does** mean the trajectory itself now contains
content at up to `probe.f_max`, so the *controller's* discrete update must resolve
it. It already does: the controller runs at the physics step (6e-5 s = 16 kHz),
which is 100x the probe's top frequency. No change needed, but add an assertion in
`optimize_excitation` that `max(probe_harmonics) * base_frequency < 0.4 / time_step`
(the materialization step), so a misconfigured probe fails loudly instead of
aliasing into the reference.

### 2.6 Regressor conditioning

The probe adds broadband acceleration content, which generally **improves** the
inertial regressor's conditioning, not worsens it. But verify: the scoring in
`optimize_excitation` uses `condition_stride=10` on the 2 ms grid, i.e. it scores at
20 ms (50 Hz sampling) — which **aliases the probe**. Fix by scoring the condition
number on the main harmonics only, or by setting `condition_stride = 1` when a probe
is configured. The first is cheaper and is the right semantics: the probe is there to
excite the transmission, not to condition the rigid-body regressor.

```python
# Score conditioning on the main harmonics alone.  The probe exists to excite
# the transmission modes; including it here would both alias (condition_stride
# samples at 50 Hz) and conflate two separate design objectives.
q_main, dq_main, ddq_main = evaluate_series(a_main, b_main, offset, omega, time)
condition = idn.regressor_condition(pin, model, data,
                                    q_main[::stride], dq_main[::stride], ddq_main[::stride], basis, ...)
```

### 2.7 Config

```yaml
excitation:
  n_harmonics: 5
  base_frequency: 0.1
  n_periods: 1
  limit_margin: 0.12
  velocity_fraction: 0.75
  max_acceleration: 4.0
  centre_jitter: 1.0
  candidates: 48
  # Modal probe.  Small-amplitude high harmonics that put energy at the
  # transmission resonances (50-170 Hz on A1-A6), without which the sampled
  # damping ratios leave no signature in the data at all and the stiffness is
  # only visible as the static deflection tau/k.  Indices are multiples of
  # base_frequency; 400 -> 40 Hz at base_frequency 0.1.  Keep the top index
  # below 0.4 / sample_time_step (2000 here) to avoid aliasing the output grid.
  probe_harmonics: [400, 700, 1000, 1300, 1600]
  probe_acceleration_fraction: 0.2
```

---

## 3. Fix: per-trajectory regime randomization (F6)

### 3.1 The problem

`_fit_to_limits` scales **every** candidate up to the tightest limit:

> "Scaling *up* is allowed and wanted: a larger feasible amplitude excites the
> dynamics more."

That is right for a single trajectory and wrong for a dataset. It means all 60
trajectories sit at `velocity_fraction 0.75`, `max_acceleration 4.0` — the
acceleration histogram is nearly identical bag to bag, and the ratio of inertial to
gravitational torque is nearly constant. The only inter-trajectory variability that
survives is `centre_jitter`, i.e. the gravity load.

A model trained on this sees one dynamic regime. It has no basis to generalize to a
slow, gravity-dominated motion or to a fast, inertia-dominated one.

### 3.2 Fix

Randomize the excitation *hyperparameters* per trajectory, from their own stream:

```yaml
excitation:
  ...
  # Per-trajectory regime randomization.  Without it every trajectory is
  # amplitude-maximized against the same caps, so the acceleration and velocity
  # distributions are near-identical bag to bag and the dataset spans a single
  # dynamic regime.  Each range is sampled log-uniformly per trajectory.
  regime:
    enabled: true
    base_frequency: [0.05, 0.3]      # Hz -> periods of 3.3 s to 20 s
    max_acceleration: [1.0, 8.0]     # rad/s^2
    velocity_fraction: [0.3, 0.9]
```

Implementation: in `generate`, before each `optimize_excitation` call, derive a
per-trajectory `FourierExcitationConfig` via `dataclasses.replace`, using a
dedicated stream `np.random.default_rng((seed, 3, trajectory_seed_value))` so the
draw is reproducible from `(seed, tier, index)` exactly like the trajectory itself.
`run_identification_simulation.py` must derive the same values, or `--tier e03
--trajectory 2` stops reproducing the dataset's bag — this is the single most
likely regression in this change, so lock it with a test (`R3_08 §B`).

**Caution on `base_frequency`.** Changing it changes the **trajectory duration**
(`duration = n_periods / base_frequency`) and therefore the number of samples per
bag. Downstream that is fine — `rollout_frame` resamples onto a uniform grid per bag
and the consumer's Savitzky-Golay filter only requires uniformity *within* a bag —
but it does mean bags will have different lengths (167 to 1000 samples per second of
motion; 3.3 s to 20 s of motion). Three things to check:

- The consumer's contiguous 50/50 split is by **row**, not by bag, so variable bag
  lengths shift where the split falls. `R3_05` replaces that split anyway; do
  `R3_05` first.
- `sample_time_step` must stay fixed (0.002) across bags — only the duration varies.
- Total dataset size scales with the mean duration. `[0.05, 0.3]` log-uniform has a
  mean period of ~8.6 s, close to today's 10 s, so size is roughly unchanged.

If variable-length bags are unwelcome, hold `base_frequency` fixed and randomize
only `max_acceleration` and `velocity_fraction`. That captures most of the regime
variability at zero schema risk. **Recommended for the first iteration.**

### 3.3 What to record

Add to each manifest record and, as columns, to `rollout_frame`:

```
exc_base_frequency, exc_max_acceleration, exc_velocity_fraction, exc_probe_top_hz
```

These are the knobs a reviewer will ask about and the conditioning variables a
regime-aware model would use.

---

## 4. Acceptance

- Probe on: the PSD of `q_motor - q_link` for a sampled robot shows a peak within
  10 % of `TransmissionSpec.natural_frequency()` for at least joints A1-A6.
- Probe on, two robots identical except for `zeta`: their `ft` traces differ by
  more than 1 % RMS. (With the probe **off**, this test should fail — write it both
  ways and assert the contrast; that is the direct evidence that the probe fixed
  F3.)
- Probe on: the regressor condition number of the main harmonics is within 10 % of
  its probe-off value.
- Probe on: peak `|ddq|` still respects `max_acceleration`; the trajectory still
  starts and ends at rest to 1e-9; still collision-free.
- Regime randomization on: across 20 trajectories, the standard deviation of the
  per-bag peak acceleration is at least 30 % of its mean (today it is near zero).
- `run_identification_simulation.py --tier eNN --trajectory k` reproduces the
  dataset bag's trajectory digest exactly.

Tests in `R3_08 §B`.
