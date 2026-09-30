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

## ur10

Joint labels are the URDF joint names, not `A1..A6`. The upper end of every stiffness interval is the
gearbox-only datasheet stiffness (`K2`) of the Harmonic Drive size inferred
from the joint's URDF effort limit (UR does not publish its supplier or
size); the centre is `K2 / r` with `r = 3` (wider than the iiwa's `r = 2`:
indirect identification, no torque sensor, stronger nonlinearity/hysteresis
-- see "Why factor 2" below, which applies with `r = 3` substituted here).

| Joint | Parameter | Nominal | Factor | Class | Source | Notes |
|---|---|---|---|---|---|---|
| shoulder_pan | stiffness | 2.6e4 Nm/rad | 3.0 | D | HD SHD/CSD catalogue, size 32, K2 = 7.8e4, ratio >= 100 [S-3, S-4] | upper end = gearbox-only K2; centre = K2/r; gearbox size inferred from the 330 Nm peak (UR size-4 joint, CSF size 32 = 333 Nm [S-5, S-6]), not published by UR |
| shoulder_lift | stiffness | 2.6e4 | 3.0 | D | as shoulder_pan | |
| elbow | stiffness | 1.23e4 | 3.0 | D | HD size 25, K2 = 3.7e4 (CSG-2A: 3.4e4 [S-4]) | size inferred from 150 Nm (UR size 3; CSF size 25 = 157 Nm [S-5, S-6]); HFUS-25 on the UR5e's size-3 joints [S-7] |
| wrist_1 | stiffness | 5.7e3 | 3.0 | D | HD size 17 (CSF/HFUS family), K2 = 1.1e4 [S-4, S-5, S-6, S-7] | **corrected in round 6** (`R6_00` Sec 4): UR size 2 = 54 Nm, which is the CSF/HFUS ratio-100 repeated-peak row of **size 17** [S-5, S-6]; the UR5e identification puts UR on that family, not CSD [S-7]. Round 4 inferred size 20 from the CSD table's 57 Nm. Size 17's K2 = 1.1e4 Nm/rad lies inside the unchanged interval 1.9e3-1.7e4, so the numbers stay |
| wrist_2 | stiffness | 5.7e3 | 3.0 | D | as wrist_1 (size 17) | |
| wrist_3 | stiffness | 5.7e3 | 3.0 | D | as wrist_1 (size 17) | |
| shoulder_pan | rotor_inertia | 4.0 kg m^2 | 2.0 | E | power-law extrapolation of Clochiatti et al. 2024 (UR5e) sizes 1 and 3 in rated torque | |
| shoulder_lift | rotor_inertia | 4.0 kg m^2 | 2.0 | E | as shoulder_pan | |
| elbow | rotor_inertia | 1.7 kg m^2 | 2.0 | C | Clochiatti et al. 2024, UR5e J1-J3 (size 3), J_m + J_red, N = 100 | same joint size, e-series vs CB3 |
| wrist_1 | rotor_inertia | 0.5 kg m^2 | 2.0 | E | power-law interpolation, as above | |
| wrist_2 | rotor_inertia | 0.5 kg m^2 | 2.0 | E | as wrist_1 | |
| wrist_3 | rotor_inertia | 0.5 kg m^2 | 2.0 | E | as wrist_1 | |
| shoulder_pan..wrist_3 | damping_ratio | 0.05-0.2 (interval, not nominal x factor) | - | E | as iiwa | same observability caveats; weaker still on pan/lift (heavier motor loop) |
| flange | payload mass | 0-5 kg (interval) | - | E | 50% of the 10 kg rating (iiwa: 6/14 = 43%); effort compliance asserted at generation time | bare flange allowed (drawn continuously from this interval, `payload.enabled: false` forces it); the floor and ceiling are configured per-asset in `config/identification/*.yaml` |

Containment cross-check: the iiwa A1 published value (18500 Nm/rad, Disney
Research) against its size-32-class gearbox (K2 = 7.8e4) is a ratio of 0.24,
inside `[1/9, 1]` -- i.e. the same `nominal = K2/r, k in [K2/r^2, K2]` rule
applied to the iiwa would have contained its only published measurement. The
UR10 has no torque sensor in series (one fewer series compliance than the
iiwa), so if anything its ratio should be higher, nearer the nominal, not
lower. The size inference from the URDF effort limit is the weakest link in
this chain and is the first thing "Measurements pending" below should close.

## fmrr_tecnobody

Joint labels are the URDF joint names, in the kinematic-chain order the asset
declares (`joint_y` carries `joint_x`, which carries `joint_z`). Every axis is
**prismatic**, so stiffness is N/m and "rotor inertia" is a reflected **mass**
in kg. Unlike both arms, the stiffness nominals here are not datasheet
readings: they are the values a sim2real calibration against the real platform
converged on (`config/assets/fmrr_tecnobody_elastic.yaml`, carried over from
the legacy Newton work), which is why the factor is `r = 2` rather than the
UR10's `r = 3`.

| Joint | Parameter | Nominal | Factor | Class | Source | Notes |
|---|---|---|---|---|---|---|
| joint_y | stiffness | 7162 N/m | 2.0 | C | sim2real calibration of the real FMRR platform (legacy `elastic_cart_robot_newton.py` line, `config/assets/fmrr_tecnobody_elastic.yaml`) | fitted on this exact platform, but through a controller and a target that `R5_01 Sec 3` shows were both wrong; treated as same-class rather than published-this-robot until a re-record confirms it |
| joint_x | stiffness | 5977 N/m | 2.0 | C | as joint_y | |
| joint_z | stiffness | 3861 N/m | 2.0 | C | as joint_y | softest axis; carries the vertical load |
| joint_y | rotor_inertia | 1.178 kg | 2.0 | E | `J / r^2` of a Delta ECMA-C10604 (400 W, `J = 0.277e-4 kg m^2`) through the **measured** transmission ratio `r = 4.850e-3 m/rad` | `R5_Q` Q-1 **closed** (`R5_08` Sec 5): `r` comes from the deployed EtherCAT configuration -- `tecnobody_workbench/config/delta400y.yaml` declares a `0x6064` position factor of 2.380734375e-8 m/count, which at the ASDA-A2's 1,280,000 counts/rev is 30.473 mm per motor revolution; the same file's `0x606c` velocity factor gives the same lead to 0.03 %. Only the rotor inertia is still class E (Delta's spec table is behind a robots.txt block), which is what the factor 2.0 now covers -- a medium-inertia ECMA variant is ~2.5x the low-inertia one. Replaces a uniform 12.0 kg that assumed a 10 mm lead |
| joint_x | rotor_inertia | 1.178 kg | 2.0 | E | as joint_y (`delta400x.yaml`, same factor) | |
| joint_z | rotor_inertia | 19.236 kg | 2.0 | E | `J / r^2` of a Delta ECMA-C10807 (750 W, `J = 1.13e-4 kg m^2`) through `r = 2.4237e-3 m/rad` | `delta750.yaml`'s position factor is 1.189742e-8 m/count, i.e. 15.229 mm per revolution -- exactly half the distal axes'. This axis' mode is therefore set by the *motor*, not the link (19.2 kg reflected against 8.9 kg moved). `F = tau_rated / r` = 986 N independently reproduces the ~1000 N patient-holding duty the owner described |
| joint_y..joint_z | damping_ratio | 0.4-1.2 (interval, not nominal x factor) | - | C | the legacy platform's calibrated ratios, 0.49-1.2 (`R5_00 Sec 3`) | an order of magnitude above a harmonic drive's, as a belt/ballscrew axis is |
| flange | payload mass | 0-2 kg (interval) | - | E | the handle beyond the F/T sensor plus the user's hand load; the real `fz` has a mean of 8.8 N, about 0.9 kg (`R5_01 Sec 5`, O-1) | on a Cartesian gantry the payload's *offset* changes no joint dynamics (a prismatic joint's gravity load is `m g . axis`, its inertia is `m`), so only the collision geometry sees it |
| joint_y..joint_z | link-side friction | 8/6/5 N per m/s viscous, 10/8/6 N Coulomb | 2.0 (`per_robot` tier only) | E | engineering estimate for a ballscrew and linear guide | `R5_Q` Q-3 **closed as a modelling decision** (owner, 2026-09-23): link friction is never a separately identified parameter, it is part of what the residual term absorbs, and it cannot be measured on this platform. With the CAD masses its share of the unexplained residual fell from 96 % to 30 % (`R5_05 Sec 6.3`), well clear of the 90 % gate above which a platform cannot resolve elasticity at all. Which tier a dataset uses -- `off`, `fixed` or `per_robot` -- is now declared (`R5_07` T-16) |

## Round 6 priors (`R6_00` Secs 5-6)

Source IDs refer to `REFACTOR_SPECS/SOURCES.md`.  Every row below is new in
round 6; the configs are `config/identification/*_round6.yaml`.

### Motor friction (Sec 5.4) -- all three platforms, class E

One rule for every joint, from the URDF's effort and velocity limits, replacing
whatever `<dynamics damping/friction>` the URDF declares (the iiwa asset's
blanket `b = 10 Nm s/rad`, `c = 0.1 Nm` was an asset artefact [I-7]):

`b_j = 0.05 * effort_j / velocity_j`, `c_j = 0.02 * effort_j`, `epsilon = 1e-2 rad/s (m/s)`,
drawn per robot with `dataset.friction_scale: [0.5, 2]` per joint and
coefficient; the rigid tier keeps the nominal.  It acts before the spring, so it
lands in `tau_cmd - tau_s`, not in the target.

| Platform | Joint(s) | effort, velocity (URDF) | b (nominal) | c (nominal) | Class | Source |
|---|---|---|---|---|---|---|
| iiwa | A1, A2 | 320 Nm, 1.484 rad/s | 10.79 Nm s/rad | 6.4 Nm | E | `R6_00` Sec 5.4 rule [I-7]; limits [S-13] (`R6_02` P2-4) |
| iiwa | A3 | 176, 1.745 | 5.04 | 3.52 | E | as above |
| iiwa | A4 | 176, 1.309 | 6.72 | 3.52 | E | as above |
| iiwa | A5 | 110, 2.269 | 2.42 | 2.2 | E | as above |
| iiwa | A6, A7 | 40, 2.356 | 0.85 | 0.8 | E | as above. Pass 1 inherited the URDF's uniform 200 Nm (4 Nm Coulomb on A6/A7); the asset now carries KUKA's per-axis maxima |
| UR10 | pan, lift | 330 Nm, 2.094 rad/s | 7.88 | 6.6 | E | as above |
| UR10 | elbow | 150, 3.142 | 2.39 | 3.0 | E | as above |
| UR10 | wrist 1-3 | 56, 3.142 | 0.89 | 1.12 | E | as above |
| FMRR | y, x | 261.9 N, 1.0 m/s | 13.1 N s/m | 5.24 N | E | as above; effort = rated torque / r [I-2, I-3] |
| FMRR | z | 986.1 N, 1.0 m/s | 49.3 | 19.7 | E | as above |

### Nonlinear spring (Sec 5.2)

Knees are given in torque and converted per robot with its sampled `k`
(`theta_1 = T_1 / (f_1 k)`, `theta_2 = theta_1 + (T_2 - T_1) / k`); `k` stays the
middle-region (`K2`) reference.

| Platform | Joint(s) | factors `[K1/K2, 1, K3/K2]` | knees `T1, T2` | Class | Source |
|---|---|---|---|---|---|
| UR10 | pan, lift (HD 32) | 0.69, 1, 1.44 | 29, 108 Nm | D | CSG-2A stiffness table [S-4]; CSD cross-check [S-3]; size inferred [S-5, S-6, S-7] |
| UR10 | elbow (HD 25) | 0.74, 1, 1.68 | 14, 48 Nm | D | as above |
| UR10 | wrist 1-3 (HD 17) | 0.74, 1, 1.45 | 3.9, 12 Nm | D | as above |
| iiwa | A1-A7 | 0.8, 1, 1.2 | 20 % / 75 % of each joint's effort limit: 64 / 240 Nm (A1, A2), 35.2 / 132 (A3, A4), 22 / 82.5 (A5), 8 / 30 (A6, A7) | E | the HD curve [S-4] diluted by the linear torque sensor in series [S-10]; catalog T1/T_rated ~ 0.21, T2/T_rated ~ 0.72-0.79 |
| FMRR | - | off | - | - | structural compliance (frame, belts, screws), no documented knee |

### Transmission error / ripple (Sec 5.3), phase drawn per bag

| Platform | Form | Order `n` | Amplitude | Class | Source |
|---|---|---|---|---|---|
| iiwa | kinematic error inside the spring, `e = A sin(n theta + phi)` | 320 rad^-1 (2 x ratio 160) | 2e-4 rad | P | measured on the base joint [S-1]; ratio 160 inferred from its period; applied to every joint |
| UR10 | kinematic error inside the spring | 200 rad^-1 (2 x ratio 100) | 2e-4 rad (~1 arcmin accuracy class) | E | ratio 100 [S-7]; nonlinear joint effects on UR arms [S-8] |
| FMRR | motor torque ripple | 206.19 rad/m (y, x), 412.58 rad/m (z) = 2 pi / lead | 1 % of the axis effort limit | E | screw leads 30.473 / 15.229 mm [I-2] |

### Measurement noise (Sec 6), modelling-error class

`sigma = p * RMS(x - mean(x))` per bag and joint (for the in-loop `dq`,
`x` is the bag's reference trajectory, the only signal known before it runs),
then gain `1 + U(-G, G)`, offset `U(-O, O) * effort`, quantization, delay.
Percentages are class E by construction (a modelling-error choice, not a claim
about the sensor); quantization is exact where the interface's is known.
**Positions are the exception** (`R6_02` Sec 3): a percentage of RMS is the
wrong convention for an encoder (0.01 % of RMS was ~800 iiwa counts, which the
in-loop derivative turned into rotor shaking), so `q` is quantization plus
`sigma_quanta = 1` count of white noise.

| Platform | Channel | p | G | O | quantization | delay | Class | Source |
|---|---|---|---|---|---|---|---|---|
| iiwa | q | - (sigma = 1 count) | - | - | 5.989e-8 rad | 1 | E / P (quantization) | [S-1]; `R6_02` Sec 3 |
| iiwa | dq | derived: the contract's SG derivative of the recorded q (FRI has no velocity) | - | - | - | - | - | `R6_00` Sec 1 |
| iiwa | ft | 1 % | 1 % | 0.2 % of effort | - | 1 | E | strain-gauge joint sensor |
| UR10 | q | - (sigma = 1 count) | - | - | 3.835e-6 rad | 1 | E | the drive's joint encoder: >= 14-bit motor encoder behind the 100:1 gear [S-7], 2 pi / (100 x 2^14) (`R6_04` A-4; was the round-5 2 pi / 2^17) |
| UR10 | dq | 0.5 % | - | - | - | 1 | E | RTDE drive estimate |
| UR10 | ft | 2 % | 3 % | 0.5 % | - | 1 | E | current-based proxy (post-training chain) |
| FMRR | q | - (sigma = 1 count) | - | - | 2.381e-8 m (y, x), 1.190e-8 m (z) | 1 | E / D (quantization) | `0x6064` factors [I-2]; `R6_02` Sec 3 |
| FMRR | dq | 0.5 % | - | - | 5.079e-5 m/s (y, x), 2.539e-5 m/s (z) | 1 | E / D | `0x606C`, 0.1 rpm units [I-2] |
| FMRR | ft | 2 % | 2 % | 0.5 % | 0.26 N (y, x), 0.99 N (z) | 1 | E / D | `0x6077` in 0.1 % of rated torque (1.27 / 2.39 Nm [I-3]) / r [I-2] |
| FMRR | w_* (cell) | 0.25 N floor | 3 % | - | - | 1 | E | round-5 Axia90 figures (`R5_Q` Q-10) |

### iiwa joint limits (`R6_02` P2-4)

| Joint | effort [Nm] | velocity [deg/s] | Class | Source |
|---|---|---|---|---|
| A1, A2 | 320 | 85 | P | KUKA per-axis maxima [S-13] (torques to verify in the KUKA PDF, which returned 403) |
| A3 | 176 | 100 | P | as above |
| A4 | 176 | 75 | P | as above |
| A5 | 110 | 130 | P | as above |
| A6, A7 | 40 | 135 | P | as above |

The URDF's velocity limits already matched [S-13]; its uniform 200 Nm effort
was replaced in both iiwa 14 assets (`assets/robots/kuka_lbr_iiwa_14_r820*/`).
The effort limits set the motor-friction rule, the PD saturation, the spring
knees and, through them, the Sec 3 step (`max_time_step: auto` gives 2.5e-4 s
elastic, 5e-4 s rigid).

### Probe and excitation (`R6_02` P2-1, `R6_04` D-1, A-7)

`probe_budget: additive`: the main trajectory keeps its full acceleration
budget and the probe adds `probe_acceleration_fraction` of it on top, raised
until peak `|tau|/effort` reaches 0.8 or the probe equals the main budget.

| Platform | comb | probe fraction (additive) | peak `|tau|/effort` over the sweep | Class | Source |
|---|---|---|---|---|---|
| iiwa `_drive` | 32 lines, 2-90 Hz | 1.0 (probe = main budget) | 0.52-0.57 | D | `link_modes` [S-14]; `reports/round6/fraction_*` |
| iiwa `_bus` | 32 lines, 2-90 Hz | 1.0 | 0.45-0.48 | D | as above |
| UR10 | 32 lines, 1.5-45 Hz | 1.0 | 0.86 at every fraction: the main trajectory alone, at its full budget, is over 0.8; the probe adds <= 0.01 | D | as above |
| FMRR | 29 lines, 0.6-7.1 Hz | 1.0 | 0.22-0.24 (at 0.5 m/s^2) | D | as above |

| Platform | `excitation.max_acceleration` | regime | Class | Source |
|---|---|---|---|---|
| FMRR | 2.0 m/s^2 (the cap, 4x the URDF's 0.5); peak `|tau|/effort` 0.40 there | [0.15, 2.0] | D | `R6_04` A-7; `reports/round6/acceleration_fmrr_tecnobody_round6.json` |

The comb is derived at load (`excitation.probe_design`), from
`min(lowest_hz, 0.5 f_link,low)` to `min(1.5 f_link,high, 0.09 x rate)`.

### Controller location (`R6_02` Sec 7 amended [I-8], `R6_04` A-1)

| Platform | location | loop rate | delay in loop | setpoint delay | gain sizing | Class | Source |
|---|---|---|---|---|---|---|---|
| UR10 | drive (CB3 has no joint-torque interface) | 2 kHz | 0 | 1 bus sample | `J_nom + m_nom` | E | [S-7]; owner answer [I-8] |
| FMRR | drive (CSP/CSV, ASDA) | 4 kHz | 0 | 1 bus sample | `J_nom + m_nom` (the moved masses) | E | owner answer [I-8] |
| iiwa `_drive` | drive | 4 kHz | 0 | 1 bus sample | `J_nom + m_nom` | E | interface undecided [I-8]: both variants |
| iiwa `_bus` | bus (FRI torque mode) | 1 kHz | 1 sample | - | `J_nom`, integral `T_i = 30 / omega` | E | `R6_00` Sec 2, `R6_04` A-5 |

`m_nom` is the nominal (payload-free) `M_jj` averaged along the reference, the
load-inertia ratio a commissioned drive knows.  The drive's velocity is its
encoder difference through a first-order low-pass at `drive_rate / 10`
(`R6_04` A-4, class E: a generic drive estimator, not the ASDA's or UR's
documented filter).

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
the same anchors; only what the hardware changes, changes. The UR10 port is
done: see the `## ur10` section above for its rows. "The UR10 is more
compliant" is true at the **arm** level (long, heavy links) but not at the
**joint** level (similar-size gearboxes to the iiwa, no lower nominal
stiffness) -- the softness is inertia, not a softer spring. Two corrections
to the original porting prediction: the proximal resonance drop from the
iiwa is real but modest (a factor of ~2.5-7, not "an order of magnitude"),
and it is driven by link/rotor inertia, not lower stiffness; and the bare
wrist-3 mode (400-1200 Hz, the same situation as the iiwa's A7) sets the
integration step, so a UR10 rollout is **not** cheaper than an iiwa one
despite the arm looking softer overall (`docs/IDENTIFICATION_DATASET.md`
"Other assets: UR10" has the measured wall-time numbers).

For a robot with no joint torque sensor and no link-side encoder (the UR10's
situation), Anchor 2's cheap direct measurement (`k = tau/delta` from
internal signals) is unavailable, and the falsifiable-consequence anchor
becomes primary instead of confirmatory: measure the first joint resonance
(step command, then FFT the motor current or motor velocity), compute
`J_eff` from the manufacturer's URDF inertials (`link_inertia_envelope`
already does this for any asset) and the motor datasheet reflected through
the gear ratio squared, then invert `k = (2 pi f)^2 J_eff`.

## Measurements pending

| Parameter | Experiment | Effort | Owner | Status |
|---|---|---|---|---|
| iiwa k, all joints | Static deflection, `k = tau/delta`, 5 torque levels | 1 day | | not started |
| iiwa first resonance | Tap test / stop-response FFT | 0.5 day | | not started |
| UR10 first resonance, per joint | Step + motor-current FFT (would move the `## ur10` stiffness rows from `D` to `M`, the only thing that should) | 0.5 day | | not started |
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
- Madsen et al., *Model-Based On-line Estimation of Time-Varying Nonlinear
  Joint Stiffness on an e-Series Universal Robots Manipulator*, ICRA 2019
  (not IROS) --
  https://ieeexplore.ieee.org/document/8793935/
- Testa et al., *Experimental identification of the joints stiffness of the
  UR5 robot arm* -- https://www.semanticscholar.org/paper/563b40a956156ad22d81577a2912adb6b59e616d

Verify each citation against the published version before it goes in a
paper; the values quoted above were read from the linked PDFs and apply to
the single joint named, not to the whole arm.
