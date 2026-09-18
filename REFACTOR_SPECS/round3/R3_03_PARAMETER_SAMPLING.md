# R3_03 — Parameter sampling: coverage, reparametrization, rotor inertia

Addresses findings **F4** (six i.i.d. draws cover the intervals poorly), **F5**
(`rotor_inertia` is frozen and confounded with `k`), and prepares the ground for
**F9** (`R3_04`, justification).

---

## 1. Coverage: what six i.i.d. draws actually buy

`sample_robots` draws each joint's stiffness log-uniformly and independently.
Reproducing the shipped configuration (`seed: 20260917`, `robots: 6`) gives realized
per-joint spans:

| Joint | Declared interval | Realized span | Decades covered / declared |
|---|---|---|---|
| A1 | 15000-35000 | 19870-34599 | 0.24 / 0.37 |
| A2 | 15000-35000 | 19141-27251 | **0.15 / 0.37** |
| A3 | 10000-25000 | 10852-24969 | 0.36 / 0.40 |
| A4 | 10000-25000 | 10096-24811 | 0.39 / 0.40 |
| A5 | 5000-15000  | 5509-10875  | 0.30 / 0.48 |
| A6 | 3000-10000  | 3387-9801   | 0.46 / 0.52 |
| A7 | 3000-10000  | 4821-7743   | **0.21 / 0.52** |

A2 sees 40 % of its declared range and A7 sees 40 % of its. A model trained on this
has never seen a soft A2 or a stiff A7, and any claim of "trained across the
plausible stiffness range" is false for those joints.

This is not bad luck — it is what i.i.d. sampling does with n=6. The expected
coverage of a uniform interval by n i.i.d. draws is `(n-1)/(n+1)` = 71 % for n=6.

### 1.1 Fix: stratified sampling per joint

Replace the i.i.d. draw with a **stratified (Latin-hypercube) draw in log space**:
split each joint's `[log k_min, log k_max]` into `robots` equal strata, take one
point uniformly inside each stratum, then permute the strata **independently per
joint**.

```python
def _stratified_log_uniform(low: np.ndarray, high: np.ndarray, count: int,
                            rng: np.random.Generator) -> np.ndarray:
    """``count`` draws per joint, one per equal-width stratum in log space.

    Independent uniform draws leave large parts of a wide interval unvisited at
    the counts this dataset uses (six robots cover ~40 % of some joints' declared
    ranges).  Stratifying guarantees full marginal coverage at identical cost;
    permuting the strata independently per joint keeps the joints' values from
    being correlated by construction.

    Returns shape ``(count, n_joints)``.
    """
    n = len(low)
    edges = np.linspace(0.0, 1.0, count + 1)
    draws = np.empty((count, n))
    for joint in range(n):
        u = rng.uniform(edges[:-1], edges[1:])       # one per stratum
        rng.shuffle(u)                                # decorrelate joints
        draws[:, joint] = np.exp(np.log(low[joint]) + u * (np.log(high[joint]) - np.log(low[joint])))
    return draws
```

### 1.2 The reproducibility property this breaks, and how to keep it

The current code documents and the test suite locks in:

> "Robots are drawn in order from their own random stream, so a robot depends only
> on the dataset seed and its index: raising `robots` adds robots without changing
> the existing ones."

Stratification is **global** — it depends on `count` — so raising `robots` from 6 to
12 would renumber and move every robot. That is a real loss: it breaks
`run_identification_simulation.py --tier e03` against an older dataset and it breaks
incremental dataset growth.

Two ways to keep the property; pick one and document it:

**(a) Recommended — make stratification explicit and versioned.** Add
`transmission.sampling: iid | stratified` to the YAML, default `stratified` for new
configs, and record `sampling` plus `robots` in the manifest. Then state plainly:
*a stratified draw is defined by `(seed, robots)`, so changing `robots` defines a
new robot set.* Add a guard in `run_identification_simulation.py` that reads
`robots` and `sampling` from the dataset manifest when `--tier` is used against a
written dataset, and errors if the current config disagrees. This is honest,
cheap, and removes a class of silent mismatch that exists already for any config
edit.

**(b) Alternative — hierarchical stratification.** Use a scrambled Sobol or a
van der Corput / Halton sequence, which is *extensible*: the first `n` points of the
sequence are stratified for every `n`, so growing the robot count preserves the
existing robots. `scipy.stats.qmc.Sobol(d=n_dof, scramble=True, seed=seed)` gives
this directly, and `scipy` is already a dependency of the consumer repo — confirm it
is one here before relying on it.

Option (b) preserves the existing promise exactly and is only a few lines more. If
`scipy.stats.qmc` is available, **prefer (b)**; fall back to (a) otherwise.

### 1.3 The other reason to prefer more robots over more trajectories

At fixed bag count, `robots x trajectories` is a budget. Today it is `6 x 10`.
Parameter coverage is the scarce axis (`R3_00 F4`), trajectory variability is not
(`R3_02 §3`). Recommended first change, **same 60 bags**:

```yaml
transmission:
  robots: 20
dataset:
  trajectories: 3
```

20 stratified draws give ~95 % marginal coverage per joint versus ~40-70 % today,
and 60 distinct trajectories become 60 distinct trajectories still (because
`trajectories_per_robot: true` keys them on `(tier, index)`).

---

## 2. Reparametrize: nominal x log-uniform factor

### 2.1 Why

The current schema is seven absolute intervals with no recorded provenance. It
cannot answer "where did 15000 come from?", it cannot express "these are all the
same gearbox family so their stiffnesses are correlated", and it does not port to a
second robot without re-guessing seven pairs of numbers.

Replace with a **nominal vector** (which has provenance and can be cited) and a
**multiplicative uncertainty factor** (which has a justification — see `R3_04`):

```yaml
transmission:
  robots: 20
  sampling: stratified
  # Per-joint nominal joint-side torsional stiffness [Nm/rad].  Provenance is
  # recorded in docs/PARAMETER_PROVENANCE.md; every value must have an entry
  # there naming its source and its anchor class.
  stiffness_nominal: [2.4e4, 2.4e4, 1.6e4, 1.6e4, 9.0e3, 5.5e3, 5.5e3]
  # Multiplicative uncertainty, log-uniform: k_j = nominal_j * exp(U(-ln r, +ln r)).
  # r = 2 is not a guess; see docs/PARAMETER_PROVENANCE.md section "Why factor 2".
  stiffness_factor: [0.5, 2.0]
  # Fraction of the uncertainty that is a single arm-wide scale rather than
  # per-joint.  A real arm is built from one gearbox family, so its joint
  # stiffnesses are correlated; independent per-joint draws generate physically
  # implausible arms (a very stiff A6 under a very soft A2).
  stiffness_common_fraction: 0.5
  damping_ratio: [0.05, 0.2]
  rotor_inertia_nominal: [1.0, 1.0, 0.5, 0.5, 0.25, 0.15, 0.15]
  rotor_inertia_factor: [0.5, 2.0]
  inertia_samples: 512
```

Keep the old `stiffness: [[min, max], ...]` form accepted for one release, mapped
internally to `nominal = sqrt(min*max)`, `factor = sqrt(max/min)`. Emit a
`DeprecationWarning` naming the equivalent new block, exactly as `load_config`
already does for the removed `tiers` ladder.

### 2.2 Correlated draw

```python
def _sample_stiffness(nominal, factor, common_fraction, count, rng):
    """Draw ``count`` stiffness vectors as nominal x log-uniform factor.

    A fraction ``common_fraction`` of the log-uncertainty is a single scalar
    shared by every joint of a robot, the rest is drawn per joint.  This encodes
    that an arm is built from one gearbox family: a robot is globally stiffer or
    softer than nominal, with joint-to-joint scatter on top of that.
    """
    span = np.log(factor[1]) - np.log(factor[0])
    centre = 0.5 * (np.log(factor[0]) + np.log(factor[1]))
    common = _stratified_log_uniform_scalar(count, rng) * span * common_fraction
    per_joint = _stratified_unit(count, len(nominal), rng) * span * (1.0 - common_fraction)
    return nominal * np.exp(centre + common[:, None] + per_joint)
```

Stratify both the common factor and the per-joint residuals. `common_fraction: 0.0`
recovers the fully independent behaviour; `1.0` gives a one-dimensional family.
Default `0.5`.

---

## 3. Rotor inertia (F5)

### 3.1 The problem

```yaml
rotor_inertia: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]  # reflected [kg m^2], as armature; to refine
```

Three separate issues:

1. **It is not sampled.** `TransmissionSampling.rotor_inertia` is documented as "it
   is not sampled", and `sample_robots` passes it through unchanged. It is exactly
   as non-observable from the URDF as the stiffness, so the argument for
   randomizing `k` applies to it verbatim.
2. **It is confounded with `k`.** The transmission mode is `sqrt(k/J_eff)` with
   `J_eff = J_r J_l / (J_r + J_l)`, and the damping is derived as
   `d = 2 zeta sqrt(k J_eff)`. Fixing `J_r` and varying `k` explores a
   one-dimensional slice of a two-dimensional degeneracy. A model that learns to
   map data to `k` under a fixed `J_r` will not transfer to a robot with a
   different `J_r`.
3. **The value is very likely wrong.** Independent published identifications for
   this robot class report, at the joint level:
   - LBR iiwa 14, base joint: controlled motor inertia **1.03 kg m²**, link inertia
     5.6 kg m², stiffness 18500 Nm/rad (Disney Research, measured on a real iiwa).
   - DLR SARA, joint 5: motor inertia **0.339 kg m²**, link-side inertia
     0.56 kg m², stiffness 9000 Nm/rad.

   A flat 0.1 kg m² is 3-10x low. Since `J_eff = J_r J_l/(J_r + J_l)` and, at
   A1-A4, `J_l >> J_r`, we have `J_eff ~ J_r`: the error goes **straight into** the
   mode frequency (as `sqrt`) and into the derived damping.

Note the one place where the flat 0.1 is *conservative*: at A7, `J_l = 3e-4 << J_r`
so `J_eff ~ J_l` and the mode is set by the link. Raising `J_r` there changes almost
nothing. So the fix costs nothing at the wrist and corrects a real error at the base.

### 3.2 Fix

- Add `rotor_inertia_nominal` (per joint, tapering from base to wrist) and
  `rotor_inertia_factor`, sampled exactly like stiffness, from the same robot stream.
- Suggested nominal, to be replaced by measured values as they arrive and to be
  entered in `docs/PARAMETER_PROVENANCE.md` with its source:
  `[1.0, 1.0, 0.5, 0.5, 0.25, 0.15, 0.15] kg m²` — anchored on 1.03 at the base
  (measured, iiwa) and 0.339 at a mid joint (measured, SARA, one size class down),
  tapering monotonically.
- Sample `J_r` and `k` **independently** (not with a shared factor): they come from
  different physical components (the gear's torsional compliance versus the motor's
  rotor plus gear ratio squared), and correlating them would hide the degeneracy
  rather than cover it.

### 3.3 Consequence to check

`elastic_time_step` is `min(max_time_step, transmission.required_time_step())`.
Raising `J_r` at the base **lowers** those modes and relaxes the step; it does
nothing at A7, which is where the step actually comes from. So the expected net
effect on runtime is neutral. Assert it (`R3_08 §A`) rather than assume it.

---

## 4. Damping ratio

Keep the interval as-is (`[0.05, 0.2]`, uniform). Two notes:

- Until `R3_02` lands, `zeta` is unobservable and this is nuisance variation. Do
  **not** widen it in the meantime — a wider unobservable range only adds label
  noise.
- Once the probe exists, consider stratifying `zeta` too, for the same reason as
  stiffness, and recording the realized per-joint span in the manifest.

---

## 5. Manifest additions

`generate` already writes `transmission_sampling: asdict(config.transmission)`.
Extend the manifest with a **realized coverage report**, which is the artifact a
reviewer will ask for:

```python
"sampling_coverage": {
    "stiffness": {
        "declared_log_span": [...],      # per joint
        "realized_log_span": [...],      # per joint
        "coverage_fraction": [...],      # realized / declared
        "method": "stratified" | "iid" | "sobol",
    },
    "rotor_inertia": {...},
    "damping_ratio": {...},
    "payload": {...},   # when R3_01 lands
}
```

Print the minimum coverage fraction at the end of `generate` and **warn below 0.8**:

```
warning: joint A2 stiffness coverage 0.41 of its declared range; raise `robots`
         or switch transmission.sampling to `stratified`.
```

---

## 6. Acceptance

- Stratified sampling with `robots: 20` gives per-joint coverage fraction `>= 0.90`
  on every joint, versus the measured 0.40-0.75 today.
- Joint-to-joint rank correlation of the sampled stiffness is not significantly
  different from zero at `common_fraction: 0.0`, and is strongly positive at
  `common_fraction: 1.0`.
- The legacy `stiffness: [[min, max], ...]` form still loads, warns, and produces
  the same robots as before for the same seed (regression guard for old configs).
- `rotor_inertia` is per-robot and appears in the `rotor_inertia__<joint>` columns
  with distinct values across tiers (today every row carries 0.1).
- `elastic_time_step` does not decrease by more than 10 % on average after the
  rotor-inertia change.

Tests in `R3_08 §A`.
