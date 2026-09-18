# R3_05 — Train/test splits and the `dynamic_model_nn` contract

Addresses findings **F7** (no held-out robot; every robot lands in both halves of
the consumer's split) and **F8** (the consumer silently truncates a 7-DoF target to
six channels).

**Do this file first.** Both findings invalidate results, both fixes are small, and
neither is worth discovering after a multi-hour dataset build.

---

## 1. F8 — the target is being truncated

### 1.1 Evidence

`docs/IDENTIFICATION_DATASET.md` currently states, under *Consumer requirement*:

> `dynamic_model_nn` originally accepted a target of only three or six channels
> while its models emit one per joint, so a 7-DoF arm failed in the loss. Its
> `dataset.py` now accepts a general `ft0..ft{dof-1}` block, checked before the
> fixed six-channel form so a 7-channel target is not silently truncated.

That is **not true of the current state of the consumer repository.**
`C:\Users\adria\Projects\dynamic_model_nn` at HEAD `45c5078` (2026-09-08), working
tree, `dataset.py` around line 310:

```python
if all(name in df.columns for name in ("fx", "fy", "fz", "tx", "ty", "tz")):
    ft = np.column_stack([df[name].to_numpy(dtype=np.float32)
                          for name in ("fx", "fy", "fz", "tx", "ty", "tz")])
    self.has_ft_sensor = True
elif all(f"ft{i}" in df.columns for i in range(6)):          # <-- fixed 6
    ft = np.column_stack([df[f"ft{i}"].to_numpy(dtype=np.float32) for i in range(6)])
    self.has_ft_sensor = True
elif all(name in df.columns for name in ("fx", "fy", "fz")):
    ...
```

There is no general `ft0..ft{dof-1}` branch anywhere in the file. A 7-DoF dataset
matches the `range(6)` branch, `self.F` is built 7-column-short, **`ft6` is dropped
without a warning**, and `F` is 6-wide against models that emit 7.

The file is listed as modified (` M dataset.py`) in that repo's `git status`, so the
fix may have been started and lost, or the doc may have been written ahead of the
code. Either way it is not there now.

### 1.2 Fix (in `dynamic_model_nn/dataset.py`)

Insert a general branch **before** the fixed six-channel one, and make the
six-channel branch unreachable for datasets whose `dof != 6`:

```python
# One target channel per joint: ft0..ft{dof-1}.  Checked before the fixed
# six-channel wrench form, so a 7-DoF arm is not silently truncated to six
# channels (which is what happened before: ft6 was dropped and F came back
# 6-wide against models that emit one channel per joint).
if all(f"ft{i}" in df.columns for i in range(self.dof)) and self.dof != 6:
    ft = np.column_stack([df[f"ft{i}"].to_numpy(dtype=np.float32) for i in range(self.dof)])
    self.has_ft_sensor = True
elif all(name in df.columns for name in ("fx", "fy", "fz", "tx", "ty", "tz")):
    ...
elif all(f"ft{i}" in df.columns for i in range(6)):
    ...
```

The `self.dof != 6` guard keeps the historical six-channel *wrench* semantics for
genuinely 6-channel datasets (where `ft0..ft5` means fx..tz, not per-joint torque),
which is the only case the old branch was ever right for. Add an explicit
`n_target_channels` attribute and assert it equals the model's output width at
construction time in the training entry points, so a future mismatch fails loudly
rather than broadcasting.

### 1.3 Producer-side guard

Do not rely on the consumer being fixed. Add to `write_dataset`:

```python
# Guard the known consumer truncation: dynamic_model_nn matches ft0..ft5 before
# any general ft0..ft{dof-1} block, so a 6-DoF-looking target on a 7-DoF arm is
# silently cut.  Fail here rather than after a training run.
if n_dof != 6 and manifest.get("n_dof") != n_dof:
    raise ValueError(...)
```

and write an explicit `target_channels: n_dof` key into the manifest, plus a
`README`-style header comment in the CSV is not possible (pandas), so instead emit a
sibling `<dataset>.contract.json`:

```json
{
  "schema": "elastic_sim.identification/2",
  "n_dof": 7,
  "input_columns": ["q0..q6", "dq0..dq6", "tau0..tau6"],
  "target_columns": ["ft0..ft6"],
  "target_semantics": "link-side joint torque [Nm]",
  "requires_consumer": "dynamic_model_nn dataset.py with the general ft0..ft{dof-1} branch",
  "split": {...}
}
```

`R3_08 §D` adds a test that loads a generated 7-DoF CSV through the consumer's
`CustomDataset` (skipped if that repo is not importable) and asserts
`dataset.F.shape[1] == 7`. That is the only test that actually closes F8.

---

## 2. F7 — no held-out robot

### 2.1 Evidence

`iter_conditions` yields, in order:

```python
for trajectory_index in range(config.n_trajectories):
    for friction_index in range(config.n_friction_samples):
        for tier in config.tiers:
            for backend in config.backends:
```

with the documented rationale:

> "Trajectory index varies slowest and backend fastest, so consecutive bags differ
> by condition rather than by regime. `dynamic_model_nn` splits train/test as a
> contiguous half of the file, and a dataset ordered by tier would put whole tiers
> on one side of that split."

That reasoning is correct *for its stated goal* — avoiding an accidental
distribution shift between the halves. But it has the side effect that **every robot
appears in both halves**. Concretely, with 10 trajectories x 6 robots the split
falls between `t4_e05` and `t5_e00`: trajectories 0-4 of every robot train,
trajectories 5-9 of every robot test.

So the test set measures *interpolation to a new trajectory on a known robot*. It
cannot measure *generalization to an unseen stiffness*, which is the entire point of
randomizing the stiffness.

### 2.2 Fix: make the split an explicit, declared artifact

Do not solve this by re-ordering bags and hoping the consumer's 50 % lands in the
right place. Declare the split in the manifest and have the consumer read it.

**Producer side.** Add a `split:` block to the YAML:

```yaml
dataset:
  ...
  # How bags are assigned to train / validation / test.  "contiguous" reproduces
  # the historical behaviour (a contiguous half of the file, every robot in both
  # halves, so only trajectory generalization is measured).  "holdout_robots"
  # reserves whole robots, which is the only split that measures generalization
  # to an unseen transmission - the reason the transmission is randomized at all.
  split:
    mode: holdout_robots          # contiguous | holdout_trajectories | holdout_robots
    test_robots: 4                # last N sampled robots, by index
    val_robots: 2
```

`generate` then writes, per bag record, a `split: "train" | "val" | "test"` field,
and a top-level manifest block:

```json
"split": {
  "mode": "holdout_robots",
  "train": ["e00", "e01", ...],
  "val": ["e14", "e15"],
  "test": ["e16", "e17", "e18", "e19"],
  "rationale": "held-out robots measure generalization to an unseen transmission"
}
```

Also add a `split` **column** to the CSV, so a consumer that only reads the CSV can
honour it with `df[df.split == "train"]` and never needs the manifest.

**Choose held-out robots at the extremes, not at random.** With stratified sampling
(`R3_03`), reserve the softest and the stiffest strata for test. That turns the test
set into a *measured extrapolation margin* rather than another interpolation sample,
and it is what makes the `R3_04 §A4` sensitivity table meaningful.

**Consumer side.** `dynamic_model_nn/dataset.py` already has the machinery — there
is a trajectory-level splitter next to the contiguous one:

```python
split = min(len(bags) - 1, max(1, int(len(bags) * train_fraction)))
train_indices = [index for bag in bags[:split] for index in dataset.bag_to_indices[bag]]
```

Add a third loader that reads the `split` column when present:

```python
def declared_split_loaders(dataset, batch_size, ...):
    """Honour a `split` column written by the producer.

    Falls back to the contiguous 50/50 split when the column is absent, so
    historical CSVs load unchanged.
    """
```

and make it the default whenever the column exists. This also removes the
variable-bag-length hazard that `R3_02 §3.2` flags for regime randomization: with a
declared split, bag lengths no longer decide where the boundary falls.

### 2.3 Keep the anti-clustering property

Re-ordering is still worth doing *within* each split so that a consumer which
ignores the split column does not get a pathological ordering. Keep
`iter_conditions` as it is; just assign `split` by tier membership rather than by
position. Nothing about the emission order needs to change.

---

## 3. Two consumer behaviours the producer must document

Neither is a bug, but both change what the dataset means, and neither is written
down anywhere today.

### 3.1 Per-bag normalization erases magnitude

`CustomDataset._normalize` can normalize **per bag** (`stats_by_bag`):

```python
bag_mean = values.mean(dim=0)
bag_std = values.std(dim=0, unbiased=False).clamp_min(1e-8)
normalized[idx] = (values - bag_mean) / bag_std
```

Applied to the target, this removes each bag's scale. On a dataset whose bags differ
*only* by transmission parameters, that discards a large part of the between-robot
information — the model sees the same normalized shape for a stiff and a soft robot.
It also drives the F2 pathology: a channel whose true RMS is ~0 (`ft6` with no
payload) gets divided by ~0 and its numerical noise is rescaled to unit variance.

**Document it in `docs/IDENTIFICATION_DATASET.md`** under the output contract, and
state the recommended setting: **global normalization for the target**, per-bag only
for inputs if at all. Once `R3_01`'s payload lands, `ft4..ft6` carry real signal and
the second half of the problem disappears; the first half does not.

### 3.2 `ddq` is recomputed, so bag uniformity is load-bearing

Already documented ("`ddq` is deliberately not emitted"), but `R3_02 §3.2`'s regime
randomization makes bags different *lengths*. Savitzky-Golay needs a uniform step
**within** a bag, which still holds. Add an explicit line saying so, and add a
producer-side assertion in `write_dataset` that every bag's `t` diff is constant to
1e-12 and equals `sample_time_step`.

---

## 4. Acceptance

- Consumer: a generated 7-DoF CSV loads with `dataset.F.shape[1] == 7`.
- Consumer: a genuine 6-channel wrench CSV (`ft0..ft5`, `dof == 6`) still loads with
  the historical semantics.
- Producer: the CSV carries a `split` column; every bag has exactly one value;
  `set(test_robots) & set(train_robots) == {}`.
- Producer: the manifest `split` block names the held-out robots and no bag record
  contradicts the column.
- Producer: every bag's time step is uniform and equals `sample_time_step`.
- Consumer: `declared_split_loaders` reproduces the contiguous split exactly when the
  column is absent.

Tests in `R3_08 §D`.
