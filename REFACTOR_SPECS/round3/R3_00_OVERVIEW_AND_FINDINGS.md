# Round 3 — Identification-dataset audit: observability, excitation, sampling, defensibility

> **Audience:** a human or an agent implementing mechanically in this repository.
> Read `R3_00` first, then the topic file for the change you are making, then
> `R3_07_IMPLEMENTATION_TASKS.md` (ordered, atomic) and `R3_08_VERIFICATION.md`.
> Rounds 1 and 2 (`REFACTOR_SPECS/*.md`, `REFACTOR_SPECS/round2/`) are implemented
> and unrelated to this round; nothing here touches the sim2real calibration path.

## 1. Scope

This round audits the **simulation-only identification dataset** path
(`scripts/generate_identification_dataset.py` → `src/elastic_sim/dataset.py`,
`excitation.py`, `torque_runners.py`, `config/identification/*.yaml`,
`docs/IDENTIFICATION_DATASET.md`) as it stands at commit `d34a452`
(2026-09-18, *"enhanced variability on dataset generation"*).

The dataset is consumed by `dynamic_model_nn`
(`C:\Users\adria\Projects\dynamic_model_nn`, HEAD `45c5078`, 2026-09-08) to train
learned dynamic models. This round therefore also covers the **producer/consumer
contract** across the two repositories.

The reference command under audit:

```bash
python scripts/generate_identification_dataset.py --backends mujoco --no-rigid
```

## 2. What that command produces today

Resolved from `config/identification/kuka_lbr_iiwa_14_r820_table.yaml`:
6 elastic robots x 1 backend x 10 trajectories x 1 friction sample.

| Quantity | Value | Where it comes from |
|---|---|---|
| Bags | **60** | `iter_conditions` = trajectories x friction x tiers x backends |
| Distinct trajectories | **60** | `dataset.trajectories_per_robot: true` |
| Samples per bag | **5001** | 10 s (`base_frequency 0.1`, `n_periods 1`) at `sample_time_step 0.002` |
| Rows | **~300 060** | 60 x 5001 |
| Columns | **103** | see `rollout_frame` |
| CSV size | **~0.5-0.7 GB** | 103 x 300k float fields in text |
| Integration step | **~6.2e-5 - 7.8e-5 s** | `elastic_time_step`, set by the A7 mode (640-810 Hz) |
| Physics steps | **~9.2e6** | ~154k per rollout x 60 |
| Wall time | **~20-60 min**, single-threaded | Python-bound: one Pinocchio RNEA call **per physics step** |

Dropping Newton (`--backends mujoco`) costs nothing scientifically on the elastic
path: `run_newton_elastic_torque` falls back to `SolverMuJoCo`, which is not an
independent physics model (already documented). It roughly halves wall time.

The six sampled robots for `seed: 20260917` are deterministic; reproduced exactly
(stiffness in Nm/rad, joints A1..A7):

```
e00 k = [32170 25873 23972 23342 10875  4363  5676]
e01 k = [19870 27251 15393 10096  8919  5714  6110]
e02 k = [34599 19237 10852 18003  8808  3387  5200]
e03 k = [33565 19785 24969 24811  5509  8159  7023]
e04 k = [29376 23697 14580 12240  5981  9801  4821]
e05 k = [30722 19141 15149 13849  8021  6506  7743]
```

## 3. Findings

Severity: **[H]** invalidates a scientific claim the dataset is meant to support ·
**[M]** materially reduces dataset value · **[L]** cost / hygiene.

| # | Finding | Severity | Spec |
|---|---|---|---|
| F1 | **The learning target is invariant to the randomized parameters.** `ft` is the spring torque, which for this model *is* `rnea(q_link, dq_link, ddq_link)` with the **nominal, fixed** URDF inertias. The `(q, dq, ddq) -> ft` map is byte-for-byte the same physics for all six robots. | **[H]** | `R3_01` |
| F2 | **No payload / tool variation.** Link-side torque RMS is `[1.3, 31, 1.7, 9.7, 0.2, 0.2, 0.0] Nm`: A7 is exactly zero, A5/A6 are noise. Three of seven target channels carry no signal, and the consumer's per-bag normalization divides them by their own std. | **[H]** | `R3_01` |
| F3 | **The excitation never excites the transmission.** 5 harmonics on a 0.1 Hz base = spectral content 0.1-0.5 Hz. Transmission modes are 50-170 Hz (A1-A6) and 640-900 Hz (A7). Consequence: **the damping ratio `zeta` is unobservable** (7 randomized nuisance dimensions with no signature in the data) and `k` enters only as the quasi-static deflection `tau/k`. | **[H]** | `R3_02` |
| F4 | **Six i.i.d. draws cover the declared intervals poorly.** Per-joint realized spans: A2 `19141-27251` of `15000-35000` (0.15 of 0.37 decades); A7 `4821-7743` of `3000-10000` (0.21 of 0.52 decades). | **[M]** | `R3_03` |
| F5 | **`rotor_inertia` is frozen at 0.1 kg m² for every joint and every robot.** It is exactly as unobservable as `k`, it is the other half of `sqrt(k/J_eff)` (so `k` and `J_r` are confounded), and it is almost certainly wrong: published values for this robot class are 0.339 kg m² (SARA J5) to 1.03 kg m² (iiwa base). | **[M]** | `R3_03` |
| F6 | **All 60 trajectories sit at one point in excitation-hyperparameter space.** Every candidate is amplitude-maximized against the same `velocity_fraction 0.75` / `max_acceleration 4.0` caps, so the velocity and acceleration distributions are near-identical bag to bag. Variability comes almost entirely from `centre_jitter` (gravity load). | **[M]** | `R3_02` |
| F7 | **No held-out robot.** `iter_conditions` puts trajectory in the outer loop, so the consumer's contiguous 50/50 split places **every** robot in both halves. Generalization to an unseen stiffness cannot be measured. | **[H]** | `R3_05` |
| F8 | **Consumer silently truncates the target.** `dynamic_model_nn/dataset.py` still matches `all(f"ft{i}" in df.columns for i in range(6))` and builds a 6-wide `F`. A 7-DoF dataset loses `ft6`. `docs/IDENTIFICATION_DATASET.md` claims this was fixed; **it was not** (verified against that repo's working tree). | **[H]** | `R3_05` |
| F9 | **Parameter ranges are asserted, not derived.** The YAML comment says "order-of-magnitude values ... not datasheet values". That is honest but not defensible at review. There is no recorded provenance, no nominal-plus-uncertainty structure, and no procedure for a second robot (UR10). | **[M]** | `R3_04` |
| F10 | **Storage and runtime waste.** 103 columns of which 28 duplicate (`q_link*` == `q*`, `dq_link*` == `dq*`) and 42 are per-row constants; CSV text format; RNEA evaluated per physics step instead of per control period. | **[L]** | `R3_06` |

## 4. The one-paragraph version

The pipeline is mechanically excellent — the torque-driven rollouts, the backend
parity work, the mode-resolving time step, the regressor-conditioned excitation and
the acceptance test that recovers base parameters to 1e-4 are all correct and
well-documented. What is missing is **observability**: the dataset randomizes
parameters the data cannot see (`zeta` entirely, `k` only quasi-statically), holds
fixed a parameter that matters as much as the ones it varies (`rotor_inertia`),
leaves the one axis that would make the distal joints informative untouched
(payload), and does not structure the file so that generalization to an unseen
robot can be measured. Fixing those four things does not require new physics — all
four are configuration-and-plumbing changes in `dataset.py`, `excitation.py` and
the YAML.

## 5. Recommended order of work

Order is by *value per unit of effort*, and each stage leaves the repo in a working state.

1. **`R3_05` first** — F8 is a one-line fix in the consumer and F7 is a re-ordering
   in `iter_conditions`. Both are cheap and both invalidate results until fixed.
   Do not generate a large dataset before these land.
2. **`R3_03`** — stratified sampling, `rotor_inertia` randomization, and the
   `nominal x factor` reparametrization. Pure sampling changes, no physics.
3. **`R3_01`** — payload randomization. Requires the URDF-injection helper; this is
   the largest single piece of new code in the round.
4. **`R3_02`** — high-frequency excitation probe and per-trajectory regime
   randomization. Requires a re-check of the integration step and of the
   acceleration/velocity limits.
5. **`R3_04`** — provenance and justification. Documentation plus a small config
   schema change; can be done in parallel with any of the above.
6. **`R3_06`** — performance and storage. Optional, but pays for itself as soon as
   robot counts rise.

## 6. Cross-repository note

Changes in `R3_05` touch `C:\Users\adria\Projects\dynamic_model_nn`. Treat that
repo's `dataset.py` as part of this round's surface; `R3_05` gives the exact edit.
Do not regenerate datasets against an unfixed consumer.
