# FMRR Tecnobody — CAD mass properties (source data for the URDF inertials)

Source: `urdf_summary_01.xlsx`, supplied by the platform's mechanical engineer
(2026-09-23), exported from the CAD assembly. Colours in the accompanying
render: **grey = fixed frame**, **red = x-axis subassembly**,
**blue = y-axis subassembly**, **green = z-axis subassembly**.

Conventions stated with the file:
- World frame: z up, origin on the floor, centred on the mounting plate of the
  rectangular column (the electrical-cabinet one).
- Every subassembly's tensor is expressed in axes parallel to the world frame.
- Spreadsheet units: g, mm, g·mm². Converted below (1 g·mm² = 1e-9 kg·m²).

## Converted to SI (kg, m, kg·m²), inertia **at the centre of mass**

| Subassembly | mass [kg] | CoM (x, y, z) [m] | ixx | iyy | izz | ixy | ixz | iyz |
|---|---|---|---|---|---|---|---|---|
| frame (fixed) | 291.400 | (1.1089, 1.0361, 1.7926) | 747.90 | 871.50 | 1048.00 | −101.00 | −59.01 | −67.09 |
| x axis | 53.2103 | (1.1412, 1.2353, 2.7045) | 62.960 | 3.2720 | 63.740 | −1.5400 | 0.3294 | 0.9949 |
| y axis | 7.9491 | (1.1741, 0.8304, 2.5788) | 0.2153 | 0.1946 | 0.0463 | 0.0001 | −0.0023 | 0.0031 |
| z axis | 8.4898 | (1.1365, 0.8258, 0.7825) | 1.2540 | 1.3170 | 0.0713 | −0.0001 | −0.0841 | −0.0000 |

The "at origin" block of the spreadsheet is the same tensors translated to the
world origin; the URDF needs the CoM form above plus the `<origin xyz>`, so
use it as a cross-check (parallel-axis) rather than as input.

## Axis mapping (owner, 2026-09-23 — `R5_Q` Q-9 answered)

The CAD's axis names are **not** this project's. In the owner's/URDF frame
(z up, the reference the whole stack uses):

| CAD colour | CAD name | **this project's joint** | mass [kg] |
|---|---|---|---|
| red | "x axis" | **`joint_y`** (the bridge, first in the chain) | 53.2103 |
| blue | "y axis" | **`joint_x`** (the carriage on the bridge) | 7.9491 |
| green | "z axis" | **`joint_z`** (the vertical column) | 8.4898 |

This confirms the URDF's kinematic chain `joint_y → joint_x → joint_z`
(round 5's D-1 reorder): the physical stacking is bridge → carriage → column,
and the heaviest assembly is on `joint_y`, which is the first joint.

## What this implies for the dynamics

The three coloured subassemblies are disjoint, so the **mass moved by each
joint** is the cumulative sum along the chain:

| joint | moves | mass [kg] |
|---|---|---|
| `joint_y` (bridge) | red + blue + green | **70.06** |
| `joint_x` (carriage) | blue + green | **16.85** |
| `joint_z` (column) | green | **8.90** |

The green column is taken as the engineer's **8.9 kg** (2026-09-23) rather
than the CAD's 8.4898 kg: his figure is the as-built axis, "F/T sensor,
carbon poles and the added ballast included". Of those 8.9 kg, **about 5 kg
is the handle below the sensor**, leaving about 3.9 kg of column structure
above it.

For three orthogonal prismatic joints the joint-space mass matrix is constant
and diagonal, `diag(69.65, 16.44, 8.49) kg`, with gravity acting only on the
vertical axis (**83.3 N** of static load). Off-diagonal inertia terms and the
CoM offsets do not enter the joint-space dynamics; they matter only for the
end-effector wrench and for visualisation.

**The current URDF is not this machine.** `fmrr_tecnobody.urdf` carries
placeholder inertials (`link1` 0.2 kg, `link1_beam` 0.6, `link2` 0.2,
`link3` 0.97) — 1-2 orders of magnitude light — and schematic geometry
(4 m beams, 2.5 m columns). Every FMRR number derived so far, including the
legacy dataset's "≈ 1 kg per axis", inherits that error.

## Consequences for the round-5 FMRR priors

With the legacy calibrated stiffnesses (5977 / 7162 / 3861 N/m) and the
assumed 12 kg reflected motor mass (`R5_Q` Q-1):

| joint | k [N/m] | moved mass [kg] | mode [Hz] nominal | soft corner | stiff corner |
|---|---|---|---|---|---|
| `joint_y` (bridge) | 7162 | 70.06 | **4.21** | 2.25 | 8.10 |
| `joint_x` (carriage) | 5977 | 16.85 | **4.65** | 2.77 | 8.27 |
| `joint_z` (column) | 3861 | 8.90 | **4.37** | 2.74 | 7.39 |

(Stiffnesses paired by joint name, as `config/assets/fmrr_tecnobody_elastic.yaml`
declares them; the legacy calibration used this project's axis names.)

versus the 7.0-17.4 Hz the round-5 FMRR config assumed from the placeholder
masses. Two direct consequences:

- the shipped probe comb (5-30 Hz) **misses the modes**; it needs to cover
  roughly 1.5-20 Hz;
- deflection is now large on the bridge: at 0.5 m/s² `joint_y` deflects
  **4.9 mm** (1 m/s² → 9.7 mm) against an encoder resolution of micrometres.
  The distal axes are smaller but still far above the noise (`joint_x`
  1.4 mm, `joint_z` 1.2 mm at 0.5 m/s²). Reaching the mode no longer needs an
  aggressive excitation, which largely dissolves `R5_Q` Q-2.

## Still unknown

- ~~The mass below the F/T sensor~~ — **answered (engineer, 2026-09-23):**
  the whole z axis is 8.9 kg including the F/T sensor, the carbon poles and
  the added ballast, and **the handle below the sensor is about 5 kg**. The
  end-effector-wrench target therefore carries 49.1 N of static weight plus
  the handle's inertial force; at the bridge's 4.2 Hz mode and its 4.9 mm
  deflection the elastic part of that force is about 17 N, so the
  transmission's signature reaches the target channel directly. The static
  offsets measured on the sensor remain unusable as a mass estimate (frame
  unrecorded, magnitude 251 N ≈ 25.6 kg, i.e. electrical offset), but nothing
  depends on them any more.
- Reflected motor mass per axis (`R5_Q` Q-1): still an estimate; the drive and
  screw/belt pitch would settle it, and it now matters more, since it is of the
  same order as the moved mass on the two distal axes.
