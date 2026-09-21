# Parameter provenance

Every non-observable parameter randomized by the identification dataset has a
row here. A parameter with no row must not appear in a shipped config
(enforced by `tests/test_identification_dataset.py::test_provenance_letters_are_validated`
and `test_every_shipped_nominal_has_a_provenance_row`).

See `REFACTOR_SPECS/round3/R3_04_RANGE_JUSTIFICATION_AND_PROVENANCE.md` for
the full argument this table supports.

## Anchor classes

| Class | Meaning | Confidence |
|---|---|---|
| `M` | Measured on our own hardware; procedure and date recorded below | highest |
| `P` | Published identification of this exact robot model | high |
| `C` | Published identification of a robot in the same class | medium |
| `D` | Component datasheet (gearbox, motor), reflected to the joint | medium |
| `E` | Engineering estimate from first principles; no measurement | lowest |

## The interval is not an estimate

The interval is not an estimate of the true stiffness. It is the support of
a prior over a parameter that the nominal model cannot observe. The dataset
is a domain-randomization ensemble drawn from that prior. The interval is
therefore judged by two criteria only: **(i)** does it contain the true value
with high confidence, and **(ii)** is it no wider than the physically
admissible range? Criterion (i) is defended by Anchor 2 below; criterion
(ii) by Anchors 1 and 3. Neither requires knowing the true value.

## kuka_lbr_iiwa_14_r820

| Joint | Parameter | Nominal | Factor | Class | Source | Notes |
|---|---|---|---|---|---|---|
| A1 | stiffness | 2.4e4 Nm/rad | 2.0 | P | Disney Research, *Toward Torque Control of a KUKA LBR IIWA for pHRI*, iiwa base joint, 18500 Nm/rad | nominal set above the single published point to centre the interval over A1-A2 |
| A2 | stiffness | 2.4e4 Nm/rad | 2.0 | C | Disney Research (as A1) -- same robot, adjacent joint, not a direct A2 measurement | downgraded from P: the anchor is A1's, not A2's (R3_10 Sec 3.4) |
| A3 | stiffness | 1.6e4 Nm/rad | 2.0 | C | Interpolated between the A1/A2 (P/C) and A5 (C) anchors for a mid-arm joint of this size class | no direct published point for A3 specifically |
| A4 | stiffness | 1.6e4 Nm/rad | 2.0 | C | As A3 | |
| A5 | stiffness | 9.0e3 Nm/rad | 2.0 | C | Iskandar et al., IROS 2020, *Joint-Level Control of the DLR Lightweight Robot SARA*, joint 5, 9000 Nm/rad | SARA is one size class down; treated as a class anchor, not a model anchor |
| A6 | stiffness | 5.5e3 Nm/rad | 2.0 | E | Tapered below A5 for a smaller wrist-class gearbox; no anchor at this size for either robot | |
| A7 | stiffness | 5.5e3 Nm/rad | 2.0 | E | As A6 | |
| A1 | rotor_inertia | 1.0 kg m^2 | 2.0 | C | Disney Research, iiwa base joint, controlled motor inertia Jc = 1.03 kg m^2 | downgraded from P: Jc is the *controlled* (closed-loop apparent) motor inertia under that paper's torque controller, which may include inertia shaping -- not confirmed here to equal the physical reflected rotor inertia this field means; upgrade to P only after checking the paper's definition of Jc (R3_10 Sec 3.4) |
| A2 | rotor_inertia | 1.0 kg m^2 | 2.0 | C | Disney Research (as A1) | same downgrade as A1, plus A2 is a different joint from the anchor |
| A3 | rotor_inertia | 0.5 kg m^2 | 2.0 | C | Interpolated between A1/A2 (C) and A5 (C) | |
| A4 | rotor_inertia | 0.5 kg m^2 | 2.0 | C | As A3 | |
| A5 | rotor_inertia | 0.25 kg m^2 | 2.0 | C | Iskandar et al., SARA J5, motor inertia 0.339 kg m^2 | tapered toward the wrist |
| A6 | rotor_inertia | 0.15 kg m^2 | 2.0 | E | Tapered below A5 | |
| A7 | rotor_inertia | 0.15 kg m^2 | 2.0 | E | As A6; conservative regardless, since `J_eff ~ J_link` at A7 (`M_77 = 3e-4 kg m^2 << J_rotor`) | |
| A1-A7 | damping_ratio | 0.05-0.2 (interval, not nominal x factor) | - | E | Lightly damped geared joint, engineering estimate | **unobservable until the excitation modal probe (R3_02) is enabled**; do not widen without a measurement, since a wider unobservable range only adds label noise |
| flange | payload mass | 0-6 kg (interval) | - | E | Inside the 14 kg iiwa rating with effort headroom verified at dataset-generation time (`R3_08 Sec C4`) | see `R3_01` |

## Why factor 2

Harmonic-drive torsional stiffness is genuinely nonlinear. The manufacturer's
own curve for a harmonic-drive gearbox gives three regimes, K1 < K2 < K3,
typically spanning a factor of roughly 1.5-2 from low torque to rated
torque, plus hysteresis and a dead-band near zero torque. Consequently, the
linearized stiffness that any experiment recovers is only defined to within
a factor of about two, depending on the operating torque at which it was
measured. A log-uniform interval of `nominal x [1/2, 2]` is therefore not an
admission of ignorance: it is the known range over which the effective
linear stiffness actually varies during operation, and it simultaneously
covers the spread between independent identifications of the same robot
model (different units, preload, temperature, lubricant state). Randomizing
over it is the correct thing to do even given a perfect single measurement,
because a single linear `k` is not a property the joint possesses.

**Why log-uniform and not uniform.** Stiffness is a positive scale
parameter. The scale-invariant (Jeffreys) prior for such a parameter is
uniform in `log k`. A uniform prior on a decade-wide interval puts most of
its mass at the stiff end, which would make "half to double" asymmetric.
Log-uniform also makes the `nominal x factor` reparametrization exactly
symmetric, so a single scalar factor carries the whole uncertainty
statement.

## The internal consistency bound

The motor-side computed-torque controller runs at
`simulation.control_frequency: 25.0 rad/s ~ 3.98 Hz`. Collocated motor-side
control of a flexible joint is stable, but the closed-loop bandwidth must
stay well below the transmission mode or the "the model cancels exactly"
property this dataset relies on degrades. Imposing a factor of five:

```
k_min >= (2 pi * 5 * f_control)^2 * J_eff
```

For A2, with `J_link` in the low single digits of kg m^2 and `J_rotor ~ 1.0`,
`J_eff` is well under 1, giving `k_min` on the order of 1e3-1e4 Nm/rad --
comfortably below every shipped lower bound (`nominal / factor`, i.e.
`1.6e4-2.4e4 / 2 = 8e3-1.2e4` for A1-A4, `4.5e3` for A5, `2.75e3` for A6-A7).
Re-derive this per joint with `scripts/report_parameter_bounds.py` rather
than trusting this paragraph, since `J_eff` is asset- and payload-dependent.

## Falsifiable consequence and the experiment that closes it

The chosen `k` and `J_eff` predict a first joint resonance
`f = sqrt(k / J_eff) / (2 pi)`, already computed by
`TransmissionSpec.natural_frequency()`. Requiring the predicted `f` to fall
inside a measured band bounds `k` from both sides, and the measurement is
cheap: a tap test (accelerometer on the link, impulse, FFT) or a
stop-response FFT (command a hard stop, FFT the joint-torque-sensor signal
or the motor current) both take minutes per joint on hardware with motor
encoders. The iiwa additionally has joint torque sensors, which make a
direct measurement possible: hold a pose, apply a known payload, read
`q_motor - q_link` and `tau`, then `k = tau / delta` to a few percent per
joint in an afternoon -- repeated at several torque levels, this measures
the factor `r` of "Why factor 2" directly instead of assuming it.

## Bounds checklist

| Bound | Source | Direction |
|---|---|---|
| `k < min(k_HD, k_sensor, k_structure)` | Series compliance of the harmonic drive, torque sensor and structure | upper |
| `k` contains published identified values | Disney Research (A1/A2), Iskandar et al. (A5) | containment |
| width = factor `r`, `r` = documented reproducibility | Harmonic-drive K1/K2/K3 nonlinearity | width |
| predicted `f` inside measured resonance band | Tap test / stop-response FFT (pending) | both |
| `k >= (2 pi * 5 f_control)^2 J_eff` | Control-bandwidth separation | lower |
| `required_time_step(k_max)` affordable | Cost (see `docs/IDENTIFICATION_DATASET.md`) | upper |

Run `scripts/report_parameter_bounds.py` to regenerate this table's numeric
columns (`bounds.csv`, `bounds.md`) and the accompanying `modes.png` figure
from the current config, rather than hand-editing numbers here.

## Porting to a UR10 (or any second robot)

Do not justify a lower range by asserting "the UR10 is more compliant." Run
the same anchors; only what the hardware changes, changes.

The UR10 has no joint torque sensors and no link-side encoder, so Anchor 2's
cheap direct measurement (`k = tau/delta` from internal signals) is
unavailable, and the falsifiable-consequence anchor becomes primary instead
of confirmatory: measure the first joint resonance (step command, then FFT
the motor current or motor velocity), compute `J_eff` from the manufacturer's
URDF inertials (`link_inertia_envelope` already does this for any asset) and
the motor datasheet reflected through the gear ratio squared, then invert
`k = (2 pi f)^2 J_eff`. Widen the factor to `r = 3` rather than `r = 2` for
two documented, checkable reasons: the identification is indirect (`k` is
estimated jointly with motor friction and the current-to-torque gain, so its
confidence interval is wider than a directly measured one), and UR-class
joints exhibit stronger stiffness nonlinearity and hysteresis than a
torque-sensored LWR joint. `config/identification/ur10_table.yaml` should
carry its own provenance rows here once that measurement is run -- do not
carry the iiwa nominal across.

## Measurements pending

| Parameter | Experiment | Effort | Owner | Status |
|---|---|---|---|---|
| iiwa k, all joints | Static deflection, `k = tau/delta`, 5 torque levels | 1 day | | not started |
| iiwa first resonance | Tap test / stop-response FFT | 0.5 day | | not started |
| UR10 first resonance | Step + motor-current FFT | 0.5 day | | not started |
| Disney Jc definition | Re-read the paper's controller section to confirm whether Jc = 1.03 kg m^2 is the physical reflected rotor inertia or a closed-loop apparent value including inertia shaping; determines whether A1/A2 rotor_inertia can go back to P | 0.5 hr | | not started |

## Sources

- *Toward Torque Control of a KUKA LBR IIWA for Physical Human-Robot Interaction* --
  https://la.disneyresearch.com/wp-content/uploads/Toward-Torque-Control-of-a-KUKA-LBR-IIWA-for-Physical-Human-Robot-Interaction-Paper.pdf
- Iskandar et al., *Joint-Level Control of the DLR Lightweight Robot SARA*, IROS 2020 --
  https://elib.dlr.de/138637/1/Iskandar_IROS2020.pdf
- *An Experimental Study of Nonlinear Stiffness, Hysteresis, and Friction
  Effects in Robot Joints with Harmonic Drives and Torque Sensors* --
  https://www.researchgate.net/publication/220806951
- Madsen et al., *Comprehensive modeling and identification of nonlinear
  joint dynamics for collaborative industrial robot manipulators*, Control
  Engineering Practice 2020 -- https://www.sciencedirect.com/science/article/abs/pii/S0967066120300988
- *Model-Based On-line Estimation of Time-Varying Nonlinear Joint Stiffness
  on an e-Series Universal Robots Manipulator*, IROS 2019 --
  https://ieeexplore.ieee.org/document/8793935/
- Testa et al., *Experimental identification of the joints stiffness of the
  UR5 robot arm* -- https://www.semanticscholar.org/paper/563b40a956156ad22d81577a2912adb6b59e616d

Verify each citation against the published version before it goes in a
paper; the values quoted above were read from the linked PDFs and apply to
the single joint named, not to the whole arm.
