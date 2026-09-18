# R3_04 — Defending the parameter ranges, and porting the procedure to a second robot

Addresses finding **F9**: the ranges are asserted, not derived, and there is no
recorded provenance or portable procedure.

This file has two halves. **Part A** is the argument — what to write in the paper
and say at a review. **Part B** is the implementation — the files, config keys and
scripts that make the argument reproducible from the repository.

---

# Part A — The argument

## A1. Frame the interval correctly

The single most important move is to stop defending the *numbers* and start
defending the *role of the interval*.

> The interval is not an estimate of the true stiffness. It is the support of a
> prior over a parameter that the nominal model cannot observe. The dataset is a
> domain-randomization ensemble drawn from that prior. The interval is therefore
> judged by two criteria only: **(i)** does it contain the true value with high
> confidence, and **(ii)** is it no wider than the physically admissible range?

Criterion (i) is defended by anchors (A2). Criterion (ii) is defended by an upper
bound on compliance and a lower bound from control stability (A3, A6). Neither
criterion requires knowing the true value, which is exactly why the framing works.

A reviewer's natural question — "why 15000 and not 12000?" — has no good answer.
The reframed question — "does your interval contain the published measured value
for this robot, and is it narrower than the range over which the linearized
stiffness physically varies?" — has a checkable answer, and the answer is yes.

## A2. Four independent anchors

Use all four. Each one alone is attackable; together they bracket the interval from
different directions.

### Anchor 1 — Component level (upper bound and nominal)

Joint compliance is a series combination of the elements between motor and link:

```
1 / k_joint  =  1 / k_harmonic_drive  +  1 / k_torque_sensor  +  1 / k_structure
```

Harmonic-drive manufacturers publish torsional stiffness per size and per unit
as a piecewise-linear curve (K1 / K2 / K3), in Nm/rad, at the gear output. This
gives:

- a **hard upper bound**: the joint cannot be stiffer than its softest element, so
  `k_joint < min(k_HD, k_sensor, k_structure)`;
- a **credible nominal**: `k_joint ~ k_HD` when the harmonic drive dominates, which
  it does on LWR-class joints with a strain-gauge torque sensor in series.

State the assumed gear size per joint. If it is unknown, say so and fall back to
Anchor 2 for the nominal — but keep the series formula in the paper, because it is
what makes the upper bound non-arbitrary.

### Anchor 2 — Published identifications of the same robot class (the interval must contain these)

Two independently measured, citable values for this robot class:

| Source | Robot / joint | Stiffness | Motor inertia | Link inertia |
|---|---|---|---|---|
| Disney Research, *Toward Torque Control of a KUKA LBR IIWA for pHRI* | LBR iiwa 14, **base joint** | **18500 Nm/rad** | 1.03 kg m² | 5.6 kg m² |
| Iskandar et al., IROS 2020, *Joint-Level Control of the DLR Lightweight Robot SARA* | SARA, **joint 5** | **9000 Nm/rad** | 0.339 kg m² | 0.56 kg m² |

Both fall inside the shipped intervals: A1 `[1.5e4, 3.5e4]` contains 18500; A5
`[5e3, 1.5e4]` contains 9000. **Say this explicitly.** The sentence to write is:

> The intervals are chosen to bracket independently published, experimentally
> identified values for this robot class, with approximately a factor-of-two margin
> on either side.

That sentence converts "we guessed" into "we cover the measurements", and it is
verifiable by the reviewer in two clicks.

### Anchor 3 — Why factor 2, and why log-uniform (the strongest argument)

This is the argument that closes the "but why *that* width" question, and it is not
hand-waving.

**Harmonic-drive torsional stiffness is genuinely nonlinear.** The manufacturer's
own curve gives three regimes K1 < K2 < K3, typically spanning a factor of roughly
1.5-2 from low torque to rated torque, plus hysteresis and a dead-band near zero
torque. Consequently:

> The linearized stiffness that any experiment recovers is only defined to within a
> factor of about two, depending on the operating torque at which it was measured.

So a log-uniform interval of `nominal x [1/2, 2]` is not an admission of ignorance.
It is the **known range over which the effective linear stiffness actually varies
during operation**, and it simultaneously covers the spread between independent
identifications of the same robot model (different units, preload, temperature,
lubricant state). Randomizing over it is the correct thing to do *even if you had a
perfect measurement*, because a single linear `k` is not a property the joint
possesses.

**Why log-uniform and not uniform.** Stiffness is a positive scale parameter. The
scale-invariant (Jeffreys) prior for such a parameter is uniform in `log k`. A
uniform prior on a decade-wide interval puts most of its mass in the stiff end and
would make "half to double" asymmetric. Log-uniform also makes the `nominal x
factor` reparametrization in `R3_03 §2` exactly symmetric, which is what lets a
single scalar `r` carry the whole uncertainty statement. The repo already samples
log-uniformly; the contribution here is to *say why* in the paper rather than
leaving it as an implementation detail.

### Anchor 4 — A falsifiable consequence, plus the experiment that closes it

The chosen `k` and `J_eff` predict a first joint resonance:

```
f = sqrt(k / J_eff) / (2 pi),      J_eff = J_rotor J_link / (J_rotor + J_link)
```

This is already computed by `TransmissionSpec.natural_frequency()`. Requiring the
predicted `f` to fall inside the measured band bounds `k` from **both** sides, and
the measurement is cheap and non-invasive:

- **Tap test.** Accelerometer on the link, impulse, FFT. Minutes per joint.
- **Stop-response FFT.** Command a hard stop and FFT the joint-torque-sensor signal
  or the motor current. No extra hardware on an iiwa.

And the direct measurement, which an iiwa makes uniquely easy because it has
**both** motor encoders and joint torque sensors:

- **Static deflection.** Hold a pose, apply a known payload, read `q_motor - q_link`
  and `tau`. Then `k = tau / delta`, to a few percent, per joint, in an afternoon.
  Repeat at several torque levels to trace the K1/K2/K3 nonlinearity directly — which
  then *measures* the factor `r` of Anchor 3 instead of assuming it.

**The framing that wins the review** is not "our numbers are right". It is:

> The interval is a stand-in for a measurement we have specified but not yet made.
> Here is the experiment, here is what it would return, and here is the sensitivity
> analysis showing how much our conclusions would move if it returned the extremes
> of our interval.

## A3. The internal consistency bound (a lower bound you can state as a constraint)

The motor-side computed-torque controller runs at
`simulation.control_frequency: 25.0 rad/s ~ 3.98 Hz`. Collocated motor-side control
of a flexible joint is stable, but the closed-loop bandwidth must stay well below
the transmission mode or the "the model cancels exactly" property degrades. Impose,
say, a factor of five:

```
k_min  >=  (2 pi * 5 * f_control)^2 * J_eff
```

For A2 with `J_link_median ~ 2.5`, `J_rotor 0.1` → `J_eff ~ 0.0962`, and
`f_control = 3.98 Hz`:

```
k_min >= (2 pi * 19.9)^2 * 0.0962  ~  1.5e3 Nm/rad
```

The shipped lower bound of `1.5e4` sits a full decade clear. **Show this**: it turns
the lower bound from an assertion into a satisfied constraint, and it is the kind of
plot (mode frequency versus control bandwidth, per joint, with the interval shaded)
that answers a whole class of questions in one figure.

## A4. The sensitivity analysis that removes the range as a threat to validity

Train on one interval, test on a shifted one. Concretely:

| Run | Train robots drawn from | Test robots drawn from |
|---|---|---|
| S0 (baseline) | `nominal x [1/2, 2]` | held-out draws from the same |
| S1 (narrow) | `nominal x [1/1.4, 1.4]` | `nominal x [1/2, 2]` |
| S2 (shifted) | `nominal x [1/2, 2]` | `nominal x [1/4, 1/2]` (softer than trained) |
| S3 (shifted) | `nominal x [1/2, 2]` | `nominal x [2, 4]` (stiffer than trained) |

Then report degradation. Two outcomes, both publishable:

- **Insensitive** → the range stops being a threat to validity; say so.
- **Sensitive** → that *is* the finding: quantify how far outside the training
  support a learned dynamic model degrades, which is a genuine contribution to the
  domain-randomization literature and far more interesting than the baseline result.

Either way you have pre-empted the question. `R3_05` provides the held-out-robot
machinery this needs.

## A5. Porting to a UR10 (or any second robot)

Do **not** justify a lower range by asserting "the UR10 is more compliant".
Run the same four anchors; only what the hardware changes, changes.

### What is different about a UR-class arm

| | LBR iiwa 14 | UR10 |
|---|---|---|
| Joint torque sensors | Yes, every joint | **No** |
| Direct `k = tau/delta` measurement | Yes, from internal signals | Not from internal signals |
| Link-side position | Observable | **Not observable** (motor encoder only) |
| Stiffness nonlinearity / hysteresis | Present, characterized in the literature | Present and generally **stronger** |
| First joint resonance | ~50-170 Hz (this repo's model) | Typically an order of magnitude lower |

### The procedure, adapted

**Anchor 1** is unchanged: series combination, gearbox datasheet for the assumed
size, no torque sensor in the chain (which removes one series compliance and, on its
own, would make the joint *stiffer*, so do not claim otherwise without evidence).

**Anchor 2** — cite UR-specific identifications rather than the iiwa ones. Starting
points in the literature: Madsen et al., *Comprehensive modeling and identification
of nonlinear joint dynamics for collaborative industrial robot manipulators*
(Control Engineering Practice, 2020); *Model-Based On-line Estimation of
Time-Varying Nonlinear Joint Stiffness on an e-Series Universal Robots Manipulator*
(IROS 2019); Testa et al., *Experimental identification of the joints stiffness of
the UR5 robot arm*; *Electro-mechanical modeling and identification of the UR5
e-series robot* (Robotica). Extract the per-joint numbers and enter them in
`docs/PARAMETER_PROVENANCE.md` with their citation — do not carry the iiwa nominal
across.

**Anchor 4 becomes the primary anchor, because Anchor 4's cheap version is
unavailable.** With no joint torque sensor and no link-side encoder, `k = tau/delta`
cannot be read from internal signals. Replace it with either:

- **The modal route (recommended).** Measure the first joint resonance `f` — step
  command, then FFT the motor current or motor velocity — and invert:

  ```
  k = (2 pi f)^2 * J_eff,     J_eff = J_r J_l / (J_r + J_l)
  ```

  with `J_l` from `link_inertia_envelope` on the UR10 URDF (the repo already
  computes this) and `J_r` from the motor datasheet reflected through the gear
  ratio squared. This needs no extra hardware and no disassembly.
- **The external-metrology route.** Laser tracker, stereo camera or a flange IMU
  measuring TCP deflection under a known payload across a set of poses; fit `k` per
  joint. More accurate, more setup.

The reasoning to write down is the *inversion*, not the assertion:

> We do not assume the UR10 is more compliant. We measure its first joint
> resonance, compute `J_eff` from the manufacturer's URDF inertials and the motor
> datasheet, and invert `k = (2 pi f)^2 J_eff`. The resulting nominal is
> substantially lower than the iiwa's because the measured resonance is
> substantially lower, not because of a prior belief about the robot.

The repo's own `config/assets/ur10_elastic_example.yaml` already carries
`stiffness: 1500.0` (800 at `wrist_3_joint`). Those are consistent with a
UR-class arm — but they are currently presented as example values. After running
Anchor 4, present them as **derived from a measured resonance**, and record the
measurement in `docs/PARAMETER_PROVENANCE.md`.

**Anchor 3 — widen the factor, and say why.** Use `r = 3` rather than `r = 2` for
the UR10, justified by two documented facts, both of which the reviewer can check:

1. the identification is **indirect** — with no torque sensor, `k` is estimated
   jointly with motor friction and the current-to-torque gain, so its confidence
   interval is wider than a directly measured one;
2. UR-class joints exhibit **stronger stiffness nonlinearity and hysteresis** than a
   torque-sensored LWR joint, so the "linearized `k` is only defined to a factor"
   argument bites harder.

This is the portable principle, and it is the sentence to put in the paper:

> **The interval is centred on a nominal derived from the specific hardware's
> gearbox and sensor stack, and its width equals the documented reproducibility of
> the identification method available for that hardware.**

One procedure. It yields a stiff, narrow range for the iiwa and a compliant, wider
one for the UR10, without either being chosen by hand.

**Re-derive the bounds and the cost.** For the UR10, re-run A3's control-bandwidth
constraint with the UR10's `J_eff`, and re-run `required_time_step`. With softer
transmissions and a heavier distal chain, the mode frequencies drop by roughly an
order of magnitude, so `elastic_time_step` will land near 1e-3 s rather than 6e-5 s
— **UR10 rollouts are roughly an order of magnitude cheaper per second of motion.**
Spend that budget on more robots (`R3_03 §1.3`), not on longer trajectories.

## A6. Bounds checklist to include in the paper

For each joint, tabulate and show that the interval sits inside:

| Bound | Source | Direction |
|---|---|---|
| `k < min(k_HD, k_sensor, k_structure)` | Anchor 1 | upper |
| `k` contains published identified values | Anchor 2 | containment |
| width = factor `r`, `r` = documented reproducibility | Anchor 3 | width |
| predicted `f` inside measured resonance band | Anchor 4 | both |
| `k >= (2 pi * 5 f_control)^2 J_eff` | A3 | lower |
| `required_time_step(k_max)` affordable | cost | upper |

That table **is** the justification section. It is six rows, every row is checkable,
and none of them requires knowing the true value.

---

# Part B — Implementation

## B1. New file: `docs/PARAMETER_PROVENANCE.md`

Create it. It is the artifact that makes Part A reproducible, and it is what a
reviewer or a future maintainer reads when they ask where a number came from.

Required structure — one table per robot asset, one row per parameter per joint:

```markdown
# Parameter provenance

Every non-observable parameter randomized by the identification dataset has a row
here. A parameter with no row must not appear in a shipped config.

## Anchor classes

| Class | Meaning | Confidence |
|---|---|---|
| `M` | Measured on our own hardware; procedure and date recorded below | highest |
| `P` | Published identification of this exact robot model | high |
| `C` | Published identification of a robot in the same class | medium |
| `D` | Component datasheet (gearbox, motor), reflected to the joint | medium |
| `E` | Engineering estimate from first principles; no measurement | lowest |

## kuka_lbr_iiwa_14_r820

| Joint | Parameter | Nominal | Factor | Class | Source | Notes |
|---|---|---|---|---|---|---|
| A1 | stiffness | 2.4e4 Nm/rad | 2.0 | P | Disney Research, iiwa base joint, 18500 Nm/rad | nominal set above the single published point to centre the interval over A1-A2 |
| A5 | stiffness | 9.0e3 Nm/rad | 2.0 | C | Iskandar et al. IROS 2020, DLR SARA J5, 9000 Nm/rad | SARA is one size class down; treated as a class anchor, not a model anchor |
| ... | | | | | | |
| A1 | rotor_inertia | 1.0 kg m² | 2.0 | P | Disney Research, iiwa base joint, Jc = 1.03 kg m² | |
| A5 | rotor_inertia | 0.25 kg m² | 2.0 | C | Iskandar et al., SARA J5, 0.339 kg m² | tapered toward the wrist |
| A1-A7 | damping_ratio | - | - | E | lightly damped geared joint, 0.05-0.2 | **unobservable until R3_02 lands** |
| flange | payload mass | - | - | E | 0-6 kg, inside the 14 kg rating with effort headroom verified | see R3_01 |

## Why factor 2

[the Anchor 3 argument, in full, with the harmonic-drive K1/K2/K3 citation]

## Measurements pending

| Parameter | Experiment | Effort | Owner | Status |
|---|---|---|---|---|
| iiwa k, all joints | static deflection, `k = tau/delta`, 5 torque levels | 1 day | | not started |
| iiwa first resonance | tap test / stop-response FFT | 0.5 day | | not started |
| UR10 first resonance | step + motor-current FFT | 0.5 day | | not started |
```

**Rule to enforce:** a nominal value that does not appear in this file must not
appear in a shipped config. Add the check to `R3_08 §D`.

## B2. Config: record provenance inline

Add an optional `provenance:` key per parameter block, carrying the anchor class
only (the detail lives in the doc):

```yaml
transmission:
  stiffness_nominal: [2.4e4, 2.4e4, 1.6e4, 1.6e4, 9.0e3, 5.5e3, 5.5e3]
  stiffness_provenance: [P, P, C, C, C, E, E]
  rotor_inertia_nominal: [1.0, 1.0, 0.5, 0.5, 0.25, 0.15, 0.15]
  rotor_inertia_provenance: [P, P, C, C, C, E, E]
```

`load_config` validates the letters against `{M, P, C, D, E}` and fails on anything
else. `generate` copies them into the manifest and prints a one-line summary:

```
provenance: stiffness [P P C C C E E]  rotor_inertia [P P C C C E E]
```

so that the weakest link in the parameterization is visible on every run.

## B3. New script: `scripts/report_parameter_bounds.py`

Produces the A6 table and the A3 figure directly from a config, so the paper's
justification section is generated, not typed.

```
usage: report_parameter_bounds.py [--config CONFIG] [--asset ASSET] [--output DIR]

For each joint, reports:
  - declared interval [k_min, k_max] and its width in decades
  - published anchors that fall inside it (read from docs/PARAMETER_PROVENANCE.md)
  - predicted mode frequency range sqrt(k/J_eff)/2pi at k_min and k_max,
    using link_inertia_envelope for J_link
  - the control-bandwidth lower bound (2 pi 5 f_control)^2 J_eff
  - required_time_step at k_max, and the implied cost per second of rollout
and writes:
  - bounds.csv        the A6 table
  - bounds.md         the same, as a markdown table ready to paste
  - modes.png         mode frequency vs joint, interval shaded, control
                      bandwidth and output-grid Nyquist as horizontal lines
```

This is ~150 lines and reuses `link_inertia_envelope`, `TransmissionSpec` and
`load_config` — no new physics.

## B4. New script: `scripts/run_range_sensitivity.py`

Drives the A4 study: generates the S0-S3 datasets by overriding
`stiffness_factor` and the seed, writes them to a directory with a small index
JSON, and prints the command lines to train and evaluate each in
`dynamic_model_nn`. Deliberately does **not** call into the other repo — it prints
the commands, so the coupling stays one-directional.

## B5. Docs to update

- `docs/IDENTIFICATION_DATASET.md` — replace the paragraph "They are not datasheet
  values; refine them when measurements exist" with a pointer to
  `docs/PARAMETER_PROVENANCE.md` and a one-paragraph summary of the A1 framing.
- `config/identification/kuka_lbr_iiwa_14_r820_table.yaml` — the long comment above
  `stiffness` becomes a two-line pointer; the detail belongs in the doc, not in the
  config, so there is exactly one place to keep current.
- Add `config/identification/ur10_table.yaml` once Anchor 4 has been run for the
  UR10, with its own provenance rows.

## B6. Sources referenced in Part A

- *Toward Torque Control of a KUKA LBR IIWA for Physical Human-Robot Interaction* —
  https://la.disneyresearch.com/wp-content/uploads/Toward-Torque-Control-of-a-KUKA-LBR-IIWA-for-Physical-Human-Robot-Interaction-Paper.pdf
- Iskandar et al., *Joint-Level Control of the DLR Lightweight Robot SARA*, IROS 2020 —
  https://elib.dlr.de/138637/1/Iskandar_IROS2020.pdf
- *An Experimental Study of Nonlinear Stiffness, Hysteresis, and Friction Effects in
  Robot Joints with Harmonic Drives and Torque Sensors* —
  https://www.researchgate.net/publication/220806951
- Madsen et al., *Comprehensive modeling and identification of nonlinear joint
  dynamics for collaborative industrial robot manipulators*, Control Engineering
  Practice 2020 — https://www.sciencedirect.com/science/article/abs/pii/S0967066120300988
- *Model-Based On-line Estimation of Time-Varying Nonlinear Joint Stiffness on an
  e-Series Universal Robots Manipulator*, IROS 2019 —
  https://ieeexplore.ieee.org/document/8793935/
- Testa et al., *Experimental identification of the joints stiffness of the UR5
  robot arm* — https://www.semanticscholar.org/paper/563b40a956156ad22d81577a2912adb6b59e616d

Verify each citation against the published version before it goes in the paper;
the values quoted in A2 were read from the linked PDFs and apply to the single
joint named, not to the whole arm.
