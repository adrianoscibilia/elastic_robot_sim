# R3_07 — Implementation tasks (ordered, atomic)

Execute in order. After each edited `.py`, run
`python -c "import ast; ast.parse(open(PATH).read())"`. Tests are in `R3_08`.
Cross-refs: `R3_01` (payload), `R3_02` (excitation), `R3_03` (sampling),
`R3_04` (justification), `R3_05` (splits/consumer), `R3_06` (performance).

Stages are independent: finishing Stage A leaves the repo working and better.
Do not start a long dataset build before Stage A is done.

---

# STAGE A — Contract and splits (`R3_05`). Do first.

## TASK A1 — `dynamic_model_nn/dataset.py`: general per-joint target block

Repository: `C:\Users\adria\Projects\dynamic_model_nn`.

▶ In `CustomDataset._load_dataframe`, INSERT a branch **before** the
`("fx","fy","fz","tx","ty","tz")` branch:

```python
if all(f"ft{i}" in df.columns for i in range(self.dof)) and self.dof != 6:
    ft = np.column_stack([df[f"ft{i}"].to_numpy(dtype=np.float32) for i in range(self.dof)])
    self.has_ft_sensor = True
elif ...   # existing chain unchanged
```

▶ ADD `self.n_target_channels = ft.shape[1]` after the chain.
**Acceptance:** a 7-DoF CSV gives `F.shape[1] == 7`; a 6-channel wrench CSV
(`dof == 6`) is unchanged. `R3_08 §D1`.

## TASK A2 — `dynamic_model_nn/dataset.py`: honour a declared split column

▶ ADD `declared_split_loaders(dataset, batch_size, **kw)` next to the existing
contiguous and bag-level splitters (`R3_05 §2.2`). It reads a `split` column when
present and falls back to the contiguous 50/50 split otherwise.
▶ ADD `self.split_labels` to `CustomDataset`, populated from the `split` column when
present, else `None`.
**Acceptance:** historical CSVs (no `split` column) produce byte-identical loaders.
`R3_08 §D2`.

## TASK A3 — `src/elastic_sim/dataset.py`: `SplitPolicy` and a `split` column

▶ ADD:

```python
@dataclass(frozen=True)
class SplitPolicy:
    mode: str = "contiguous"       # contiguous | holdout_trajectories | holdout_robots
    test_robots: int = 0
    val_robots: int = 0
    test_trajectories: int = 0
    val_trajectories: int = 0

    def __post_init__(self) -> None:
        if self.mode not in ("contiguous", "holdout_trajectories", "holdout_robots"):
            raise ValueError(f"unknown split mode {self.mode!r}")
```

▶ ADD `split: SplitPolicy` to `DatasetConfig` (default `SplitPolicy()`), parse
`dataset.split` in `load_config`, add `"split"` to the accepted `data_cfg` keys.
▶ ADD `assign_splits(config) -> dict[str, str]` mapping bag name -> label. For
`holdout_robots`, reserve the **extreme strata** (softest `test_robots // 2` and
stiffest `test_robots - test_robots // 2` by mean log stiffness), not the last N by
index — `R3_05 §2.2`.
▶ In `rollout_frame`, ADD a `split` column. In `generate`, ADD `"split"` to each
record and a top-level `manifest["split"]` block.
**Acceptance:** `set(train) & set(test) == set()`; every bag labelled exactly once.
`R3_08 §D3`.

## TASK A4 — `write_dataset`: contract sidecar and uniformity assertion

▶ ASSERT per bag that `np.diff(t)` is constant to 1e-12 and equals
`sample_time_step`; raise naming the offending bag.
▶ WRITE `<dataset>.contract.json` with the schema block from `R3_05 §1.3`.
**Acceptance:** `R3_08 §D4`.

## TASK A5 — `docs/IDENTIFICATION_DATASET.md`: correct the consumer claim

▶ REPLACE the *Consumer requirement* paragraph with the true state of affairs and a
pointer to `R3_05`. ▶ ADD the normalization warning (`R3_05 §3.1`) and the split
documentation. **Do not leave a claim in the docs that the code does not hold.**

---

# STAGE B — Sampling (`R3_03`). No physics changes.

## TASK B1 — `dataset.py`: stratified / Sobol draws

▶ ADD `_stratified_log_uniform(low, high, count, rng)` (`R3_03 §1.1`).
▶ ADD `sampling: str = "stratified"` to `TransmissionSampling`, accepted values
`iid | stratified | sobol`. Use `scipy.stats.qmc.Sobol` when `sobol` and scipy is
importable; raise a clear error naming the fallback otherwise.
▶ REWRITE `sample_robots` to dispatch on `sampling`, keeping `iid` bit-identical to
today for the same seed.
**Acceptance:** `iid` reproduces the six robots listed in `R3_00 §2` exactly;
`stratified` raises per-joint coverage above 0.9 at `robots: 20`. `R3_08 §A1`.

## TASK B2 — `dataset.py` + YAML: `nominal x factor` reparametrization

▶ ADD to `TransmissionSampling`: `stiffness_nominal`, `stiffness_factor`,
`stiffness_common_fraction`, `rotor_inertia_nominal`, `rotor_inertia_factor`.
▶ KEEP `stiffness: [[min, max], ...]` accepted; map to
`nominal = sqrt(min*max)`, `factor = sqrt(max/min)` and emit a `DeprecationWarning`
naming the new block (mirror the existing `tiers` deprecation message style).
▶ ADD `_sample_stiffness(...)` with the shared-factor decomposition (`R3_03 §2.2`).
**Acceptance:** legacy configs load and, with `sampling: iid`, produce identical
robots. `R3_08 §A2`.

## TASK B3 — `dataset.py`: randomize rotor inertia

▶ Draw `rotor_inertia` per robot from `rotor_inertia_nominal x
rotor_inertia_factor`, **independently** of the stiffness draw (`R3_03 §3.2`).
▶ `Tier.rotor_inertia` already accepts a per-joint tuple; no signature change.
▶ UPDATE the YAML nominal to `[1.0, 1.0, 0.5, 0.5, 0.25, 0.15, 0.15]`.
**Acceptance:** `rotor_inertia__<joint>` columns differ across tiers;
`elastic_time_step` does not fall more than 10 % on average. `R3_08 §A3`.

## TASK B4 — `dataset.py`: coverage report in the manifest

▶ ADD `sampling_coverage` to the manifest (`R3_03 §5`) and a printed warning when
any joint's coverage fraction is below 0.8.
**Acceptance:** `R3_08 §A4`.

## TASK B5 — `run_identification_simulation.py`: manifest guard

▶ When `--tier eNN` is used together with `--manifest PATH`, read `robots`,
`sampling` and `seed` from the manifest and error on disagreement with the current
config, naming both values. This closes the class of silent mismatch that
stratification introduces (`R3_03 §1.2a`).

---

# STAGE C — Payload (`R3_01`). Largest piece of new code.

## TASK C1 — NEW `src/elastic_sim/payload.py`

▶ CREATE `Payload`, `payload_asset`, `_last_link_name`, `_mirror_siblings` per
`R3_01 §2.3`. Use `text[:text.rindex("</robot>")] + injected + text[...]`, not
`str.replace`.
▶ `_last_link_name` must return the child link of the **last active joint**, derived
from `discover_urdf_joints` filtered to `asset.joint_names`. Assert the name is
present in the URDF text.
▶ Compose with `generic_mujoco_runner._materialized_mujoco_urdf` for mesh
resolution rather than duplicating it.
**Acceptance:** `R3_08 §C1`.

## TASK C2 — `dataset.py`: `PayloadSampling` and `sample_payloads`

▶ ADD the dataclass and sampler per `R3_01 §2.4`, stream key `(int(seed), 2)`.
▶ Parse `payload:` in `load_config`; add `"payload"` to the accepted top-level keys.
▶ The rigid tier always gets an empty `Payload()`.
**Acceptance:** `R3_08 §C2`.

## TASK C3 — `dataset.py`: thread the payload through `run_condition`

▶ ADD `payload: Payload | None = None` to `run_condition`; wrap its body in
`with payload_asset(asset, payload) as asset_p:` and use `asset_p` throughout.
▶ REPLACE the hoisted `link_inertia` with the payload-keyed `_envelope` cache
(`R3_01 §2.5`).
▶ Keep excitation payload-free — option (a) in `R3_01 §2.6` — and say so in
`generate`'s docstring.
**Acceptance:** `R3_08 §C3`.

## TASK C4 — output schema: payload columns and deflection channels

▶ ADD `payload_mass`, `payload_offset_x/y/z`, `payload_size` to `rollout_frame`
(NaN on the rigid tier) and to each manifest record.
▶ ADD `defl0..defl{n-1} = q_motor{i} - q_link{i}` (`R3_01 §3`).
▶ UPDATE the output-contract table in `docs/IDENTIFICATION_DATASET.md`.
**Acceptance:** `R3_08 §C4`.

---

# STAGE D — Excitation (`R3_02`). Re-check limits and step after each step here.

## TASK D1 — `excitation.py`: explicit harmonic indices

▶ ADD an `indices=None` parameter to `evaluate_series` and `project_coefficients`,
defaulting to `arange(1, H+1)` so every existing call is unchanged.
**Acceptance:** existing excitation tests pass bit-identically. `R3_08 §B1`.

## TASK D2 — `excitation.py`: modal probe

▶ ADD `probe_harmonics: tuple[int, ...] = ()` and
`probe_acceleration_fraction: float = 0.2` to `FourierExcitationConfig`, with
validation: indices strictly increasing, all `> n_harmonics`, and
`max(probe_harmonics) * base_frequency < 0.4 / time_step`.
▶ SPLIT `_fit_to_limits` into a two-budget scaling (`R3_02 §2.4`).
▶ SCORE the regressor condition number on the **main harmonics only**
(`R3_02 §2.6`); this is required, not optional — `condition_stride=10` aliases the
probe.
▶ RECORD `probe_harmonics` and the realized probe top frequency in the trajectory
metadata.
**Acceptance:** `R3_08 §B2`, `§B3`.

## TASK D3 — `excitation.py` + `dataset.py`: per-trajectory regime randomization

▶ ADD an `excitation.regime` block (`R3_02 §3.2`).
▶ In `generate`, derive a per-trajectory `FourierExcitationConfig` via
`dataclasses.replace` from stream `default_rng((seed, 3, trajectory_seed(...)))`.
▶ MIRROR the identical derivation in `run_identification_simulation.py`.
▶ **First iteration: randomize `max_acceleration` and `velocity_fraction` only.**
Leave `base_frequency` fixed until Stage A's declared split is in place, so bag
lengths stay uniform (`R3_02 §3.2`).
▶ ADD `exc_base_frequency`, `exc_max_acceleration`, `exc_velocity_fraction`,
`exc_probe_top_hz` columns and manifest fields.
**Acceptance:** trajectory digests reproduce between the two scripts. `R3_08 §B4`.

## TASK D4 — YAML: turn the probe on

▶ ADD `probe_harmonics: [400, 700, 1000, 1300, 1600]`,
`probe_acceleration_fraction: 0.2` and the `regime:` block to
`config/identification/kuka_lbr_iiwa_14_r820_table.yaml` with the explanatory
comments from `R3_02 §2.7` and `§3.2`.

---

# STAGE E — Justification artifacts (`R3_04`). Parallelizable with B/C/D.

## TASK E1 — NEW `docs/PARAMETER_PROVENANCE.md`

▶ CREATE with the structure in `R3_04 §B1`, filled for
`kuka_lbr_iiwa_14_r820`. Every nominal in the shipped YAML gets a row.

## TASK E2 — config: provenance letters

▶ ADD `stiffness_provenance` / `rotor_inertia_provenance` (letters from
`{M, P, C, D, E}`), validated in `load_config`, copied into the manifest, printed by
`generate` as a one-line summary.
**Acceptance:** `R3_08 §D5`.

## TASK E3 — NEW `scripts/report_parameter_bounds.py`

▶ CREATE per `R3_04 §B3`: writes `bounds.csv`, `bounds.md`, `modes.png`.

## TASK E4 — NEW `scripts/run_range_sensitivity.py`

▶ CREATE per `R3_04 §B4`: generates the S0-S3 datasets, writes an index JSON,
prints (does not run) the consumer's training commands.

## TASK E5 — docs

▶ `docs/IDENTIFICATION_DATASET.md`: replace "They are not datasheet values" with a
pointer to `PARAMETER_PROVENANCE.md` plus the framing paragraph from `R3_04 §A1`.
▶ `config/identification/*.yaml`: shrink the stiffness comment to a two-line
pointer.

---

# STAGE F — Performance (`R3_06`). Optional; do before raising `robots` past ~25.

## TASK F1 — hoist address lookups to index arrays

▶ In `run_mujoco_torque`, `run_mujoco_elastic_torque` and both Newton runners,
build integer index arrays once and use fancy indexing (`R3_06 §2.2`).
**Acceptance:** bit-identical output. `R3_08 §E1`.

## TASK F2 — preallocate and record only the written steps

▶ Preallocate the output arrays; record every `k`-th step with
`k = round(sample_time_step / time_step)` plus endpoints; skip interpolation in
`rollout_frame` when `k` is exact (`R3_06 §2.3`).
**Acceptance:** `R3_08 §E2`.

## TASK F3 — `control_decimation`

▶ ADD `simulation.control_decimation: int = 1`; hold the torque between controller
updates; assert `1 / (time_step * decimation) > 10 * probe_top_hz`; record in the
manifest (`R3_06 §2.1`).
**Acceptance:** `R3_08 §E3`.

## TASK F4 — `--jobs N`

▶ Parallelize the bag loop with a process pool; force `jobs=1` under `--visualize`;
send config, not models (`R3_06 §2.5`).
**Acceptance:** `--jobs 4` is bit-identical to `--jobs 1`. `R3_08 §E4`.

## TASK F5 — storage

▶ `write_dataset` dispatches on suffix (`.csv` / `.parquet`); add
`dataset.metadata_columns: inline | sidecar`; write floats as float32 / `%.7g`
(`R3_06 §3.2`).
**Acceptance:** `R3_08 §E5`.

---

# Definition of done for the round

```bash
uv run pytest tests/ -q                                    # all green
uv run python scripts/report_parameter_bounds.py            # bounds table, no bound violated
uv run python scripts/generate_identification_dataset.py \
    --backends mujoco --no-rigid --robots 20 --trajectories 3
```

and the printed summary shows:

- every joint's sampling coverage `>= 0.9`;
- provenance letters with no `E` on stiffness or rotor inertia;
- `ft4..ft6` RMS all `> 0.5 Nm`;
- held-out robots named, with no overlap;
- a probe peak within 10 % of the predicted mode on A1-A6.
