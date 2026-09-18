# R3_08 — Verification (round 3)

Add to `tests/`. Run `uv run pytest tests/ -q`. Tests needing a simulator backend or
the consumer repository must `pytest.importorskip(...)` so the suite still runs
without heavy deps, following the existing convention in
`tests/test_identification_dataset.py`.

Each section maps to a stage in `R3_07`.

---

## A. Parameter sampling (`R3_03`, Stage B)

### A1 — stratification covers the declared range; `iid` is unchanged

`tests/test_transmission_sampling.py`

```python
import numpy as np
from elastic_sim.dataset import TransmissionSampling, sample_robots

INTERVALS = ((1.5e4, 3.5e4), (1.5e4, 3.5e4), (1.0e4, 2.5e4), (1.0e4, 2.5e4),
             (5.0e3, 1.5e4), (3.0e3, 1.0e4), (3.0e3, 1.0e4))

def _coverage(robots, intervals):
    k = np.array([r.stiffness for r in robots])
    lo, hi = np.array(intervals).T
    realized = np.log(k.max(axis=0) / k.min(axis=0))
    declared = np.log(hi / lo)
    return realized / declared

def test_iid_reproduces_the_shipped_robots():
    """Regression guard: the documented seed/index promise must not move."""
    sampling = TransmissionSampling(robots=6, stiffness=INTERVALS, sampling="iid")
    k0 = np.array(sample_robots(sampling, 20260917)[0].stiffness)
    expected = [32170., 25873., 23972., 23342., 10875., 4363., 5676.]
    assert np.allclose(k0, expected, rtol=2e-4)

def test_iid_undercovers_and_stratified_does_not():
    iid = sample_robots(TransmissionSampling(robots=6, stiffness=INTERVALS, sampling="iid"), 20260917)
    strat = sample_robots(TransmissionSampling(robots=20, stiffness=INTERVALS, sampling="stratified"), 20260917)
    assert _coverage(iid, INTERVALS).min() < 0.6      # documents the finding
    assert _coverage(strat, INTERVALS).min() > 0.90   # documents the fix

def test_stratified_does_not_correlate_joints():
    robots = sample_robots(TransmissionSampling(robots=64, stiffness=INTERVALS, sampling="stratified"), 7)
    k = np.log(np.array([r.stiffness for r in robots]))
    corr = np.corrcoef(k.T)
    off = corr[~np.eye(len(INTERVALS), dtype=bool)]
    assert np.abs(off).max() < 0.35, "strata must be permuted independently per joint"
```

### A2 — `nominal x factor` round-trips the legacy form

```python
def test_legacy_interval_form_maps_to_nominal_and_factor(recwarn):
    legacy = TransmissionSampling(robots=6, stiffness=INTERVALS, sampling="iid")
    modern = TransmissionSampling(
        robots=6, sampling="iid",
        stiffness_nominal=tuple(np.sqrt(lo * hi) for lo, hi in INTERVALS),
        stiffness_factor=(1.0 / np.sqrt(3.5e4 / 1.5e4), np.sqrt(3.5e4 / 1.5e4)),
        stiffness_common_fraction=0.0,
    )
    a = np.array([r.stiffness for r in sample_robots(legacy, 11)])
    b = np.array([r.stiffness for r in sample_robots(modern, 11)])
    assert np.allclose(a[:, 0], b[:, 0], rtol=1e-9)   # A1/A2 share the interval
```

### A3 — rotor inertia is sampled and does not wreck the step

```python
def test_rotor_inertia_varies_across_robots_without_shrinking_the_step(asset):
    from elastic_sim.dataset import DEFAULT_CONFIG, load_config, elastic_time_step
    config = load_config(DEFAULT_CONFIG)
    steps, rotors = [], []
    for tier in config.tiers:
        if tier.is_rigid:
            continue
        spec = tier.transmission(7, link_inertia)
        rotors.append(spec.rotor_inertia)
        steps.append(elastic_time_step(spec, config))
    assert np.std(np.array(rotors), axis=0).max() > 0.0, "rotor inertia must be sampled"
    assert np.mean(steps) > 0.9 * BASELINE_MEAN_STEP
```

### A4 — coverage report present and warns

```python
def test_manifest_reports_sampling_coverage(tmp_path, small_config, asset):
    frame, manifest, _ = generate(small_config, asset, verbose=False)
    cov = manifest["sampling_coverage"]["stiffness"]
    assert len(cov["coverage_fraction"]) == 7
    assert cov["method"] in ("iid", "stratified", "sobol")
```

---

## B. Excitation (`R3_02`, Stage D)

### B1 — explicit indices are backward compatible

```python
def test_evaluate_series_indices_default_is_unchanged():
    a = np.random.default_rng(0).normal(size=(3, 5)); b = a[::-1].copy()
    t = np.linspace(0, 10, 501); off = np.zeros(3)
    q1 = exc.evaluate_series(a, b, off, 2 * np.pi * 0.1, t)
    q2 = exc.evaluate_series(a, b, off, 2 * np.pi * 0.1, t, indices=np.arange(1, 6))
    assert all(np.array_equal(x, y) for x, y in zip(q1, q2))
```

### B2 — the probe puts energy at the transmission mode

This is the test that closes F3. **Write it both ways** so the contrast is recorded.

```python
@pytest.mark.slow
def test_probe_excites_the_transmission_mode(asset):
    """Run the same robot twice, probe off and probe on, and FFT the deflection.

    Probe OFF is the state documented in R3_00 F3: no energy above ~0.5 Hz, so
    nothing at the 50-170 Hz modes.  Probe ON must put a peak there.
    """
    pytest.importorskip("mujoco")
    spectra = {}
    for probe in ((), (400, 700, 1000, 1300, 1600)):
        result = _one_elastic_rollout(asset, probe_harmonics=probe)   # helper
        deflection = np.asarray(result["q_motor"]) - np.asarray(result["q_link"])
        freq = np.fft.rfftfreq(len(deflection), d=0.002)
        spectra[probe] = (freq, np.abs(np.fft.rfft(deflection, axis=0)))

    freq, off = spectra[()]
    assert off[freq > 1.0].max() / off[freq <= 1.0].max() < 1e-3, "probe-off baseline"

    freq, on = spectra[(400, 700, 1000, 1300, 1600)]
    modes = _transmission_modes(asset)          # TransmissionSpec.natural_frequency()
    for joint in range(6):                      # A7's 640-900 Hz mode is above Nyquist
        band = (freq > 1.0)
        peak = freq[band][np.argmax(on[band, joint])]
        assert abs(peak - modes[joint]) / modes[joint] < 0.10, f"joint A{joint + 1}"
```

### B3 — the probe makes `zeta` observable, and respects limits

```python
@pytest.mark.slow
def test_damping_ratio_becomes_observable_with_the_probe(asset):
    """Two robots identical but for zeta must produce different data.

    With the probe off this must FAIL to find a difference - that failure is
    the documented evidence for R3_00 F3.  Assert both directions.
    """
    # build tier_a (zeta=0.05) and tier_b (zeta=0.20), same k, same trajectory
    # probe off: relative RMS difference of ft < 1e-3   (unobservable)
    # probe on : relative RMS difference of ft > 1e-2   (observable)

def test_probe_respects_limits_and_endpoint_rest(asset):
    config = exc.FourierExcitationConfig(n_harmonics=5, base_frequency=0.1,
                                         time_step=0.002, max_acceleration=4.0,
                                         probe_harmonics=(400, 700, 1000),
                                         probe_acceleration_fraction=0.2)
    traj = exc.optimize_excitation(asset, config, seed=3, n_candidates=4)
    assert np.abs(traj.acceleration).max() <= 4.0 + 1e-9
    assert np.abs(traj.velocity[0]).max() < 1e-9 and np.abs(traj.velocity[-1]).max() < 1e-9
    assert np.abs(traj.acceleration[0]).max() < 1e-9

def test_probe_does_not_spoil_the_regressor_condition(asset, model):
    """Conditioning is scored on the main harmonics, so it must barely move."""
    # optimize with and without the probe, same seed; assert |c_on/c_off - 1| < 0.10

def test_probe_top_frequency_is_below_the_output_nyquist():
    with pytest.raises(ValueError, match="alias"):
        exc.FourierExcitationConfig(base_frequency=0.1, time_step=0.002,
                                    probe_harmonics=(400, 4000))
```

### B4 — regime randomization reproduces across the two scripts

```python
def test_regime_randomization_spreads_the_dynamic_regime(asset):
    # 20 trajectories with regime.enabled -> std(peak |ddq|) / mean(peak |ddq|) > 0.30
    # 20 trajectories without            -> the same ratio < 0.05

def test_single_rollout_script_reproduces_the_dataset_trajectory(asset, small_config):
    """The most likely regression in Stage D: two derivations of the same seed."""
    from scripts.run_identification_simulation import resolve_trajectory   # or its helper
    _, manifest, _ = generate(small_config, asset, verbose=False)
    record = manifest["records"][0]
    rebuilt = resolve_trajectory(small_config, asset, tier=record["tier"],
                                 index=record["trajectory"])
    assert rebuilt.digest() == record["trajectory_digest"]
```

---

## C. Payload (`R3_01`, Stage C)

### C1 — injection is consistent across every consumer of the asset

This is the test that catches the failure mode `R3_01 §2.2` warns about.

```python
def test_payload_reaches_pinocchio_and_mujoco_identically(asset):
    pin_mod = pytest.importorskip("pinocchio"); mujoco = pytest.importorskip("mujoco")
    from elastic_sim.payload import Payload, payload_asset
    payload = Payload(mass=5.0, offset=(0.0, 0.0, 0.10), size=0.15)
    with payload_asset(asset, payload) as asset_p:
        # Pinocchio CRBA M_77 must rise by the analytic amount
        # MuJoCo inverse dynamics on a random (q, dq, ddq) must agree with
        # Pinocchio's rnea to 1e-6 Nm - the same bar the bare asset is held to.

def test_empty_payload_is_the_identity(asset):
    from elastic_sim.payload import Payload, payload_asset
    with payload_asset(asset, Payload()) as same:
        assert same is asset

def test_payload_attaches_to_the_last_active_joint_child(asset):
    """Not the URDF's last link, and not a trailing fixed ee frame."""
```

### C2 — sampling streams are independent

```python
def test_payload_stream_does_not_perturb_the_robot_stream():
    """Adding payloads must not renumber or move the sampled robots."""
    without = sample_robots(SAMPLING, 20260917)
    _ = sample_payloads(PAYLOAD_SAMPLING, 20260917, count=6)
    with_payloads = sample_robots(SAMPLING, 20260917)
    assert all(np.array_equal(a.stiffness, b.stiffness) for a, b in zip(without, with_payloads))
```

### C3 — the controller still cancels (the de-tuning guard)

```python
@pytest.mark.slow
def test_feedback_stays_small_with_a_payload(asset, small_config):
    """A payload that reached the simulator but not Pinocchio shows up here."""
    # run one elastic bag with a 5 kg payload
    # assert record["feedback_ratio"] < 0.05, same bar as the payload-free suite
```

### C4 — the distal channels stop being empty, and the effort limit is respected

```python
@pytest.mark.slow
def test_payload_gives_the_distal_channels_signal(asset, small_config):
    # payload-free : ft4, ft5, ft6 RMS ~ [0.2, 0.2, 0.0] Nm
    # 5 kg payload : all three > 0.5 Nm
    # and peak |tau| < 0.8 * URDF effort (200 Nm) on every joint

def test_payload_relaxes_rather_than_tightens_the_integration_step(asset):
    """Raising J_link at the wrist lowers the A7 mode; assert, do not assume."""
```

---

## D. Splits and consumer contract (`R3_05`, Stage A)

### D1 — the consumer keeps all seven target channels

```python
def test_consumer_loads_seven_target_channels(tmp_path):
    sys.path.insert(0, DYNAMIC_MODEL_NN)      # skip if not importable
    CustomDataset = pytest.importorskip("dataset").CustomDataset
    csv = _write_synthetic_dataset(tmp_path, dof=7, rows=200)
    data = CustomDataset(csv, normalize=False)
    assert data.dof == 7
    assert data.F.shape[1] == 7, "ft6 must not be dropped by the six-channel branch"

def test_consumer_keeps_six_channel_wrench_semantics(tmp_path):
    csv = _write_synthetic_dataset(tmp_path, dof=6, rows=200)   # ft0..ft5 = wrench
    data = CustomDataset(csv, normalize=False)
    assert data.F.shape[1] == 6
```

### D2 — declared splits are honoured and fall back cleanly

```python
def test_declared_split_loader_falls_back_when_no_column(tmp_path):
    # a CSV without `split` must reproduce the historical contiguous 50/50 indices
```

### D3 — held-out robots do not leak

```python
def test_holdout_robots_do_not_appear_in_train(small_config_holdout, asset):
    frame, manifest, _ = generate(small_config_holdout, asset, verbose=False)
    train = set(frame.loc[frame.split == "train", "tier"])
    test = set(frame.loc[frame.split == "test", "tier"])
    assert train and test and not (train & test)
    assert set(manifest["split"]["test"]) == test

def test_holdout_reserves_the_extreme_strata(small_config_holdout, asset):
    """Test robots must be the softest and stiffest, not the last N by index."""
```

### D4 — uniformity and the contract sidecar

```python
def test_every_bag_has_a_uniform_time_step(small_config, asset, tmp_path):
    frame, manifest, _ = generate(small_config, asset, verbose=False)
    for bag, group in frame.groupby("bag"):
        step = np.diff(group["t"].to_numpy())
        assert np.allclose(step, small_config.sample_time_step, atol=1e-12), bag

def test_contract_sidecar_is_written(tmp_path, small_config, asset):
    csv, *_ = write_dataset(*generate(small_config, asset, verbose=False)[:2],
                            tmp_path / "d.csv")
    contract = json.loads(csv.with_suffix(".contract.json").read_text())
    assert contract["n_dof"] == 7 and contract["target_columns"] == ["ft0..ft6"]
```

### D5 — provenance is validated

```python
def test_provenance_letters_are_validated(tmp_path):
    with pytest.raises(ValueError, match="provenance"):
        load_config(_yaml_with(stiffness_provenance=["P", "X", "C", "C", "C", "E", "E"]))

def test_every_shipped_nominal_has_a_provenance_row():
    """A nominal with no row in docs/PARAMETER_PROVENANCE.md must not ship."""
    doc = (REPO / "docs" / "PARAMETER_PROVENANCE.md").read_text(encoding="utf-8")
    for joint in ("A1", "A2", "A3", "A4", "A5", "A6", "A7"):
        assert f"| {joint} | stiffness" in doc
        assert f"| {joint} | rotor_inertia" in doc
```

---

## E. Performance and storage (`R3_06`, Stage F)

```python
def test_index_array_hoisting_is_bit_identical(asset, short_trajectory):
    """Refactor only: every recorded value must be unchanged."""

def test_direct_step_recording_matches_interpolation_without_a_probe(asset):
    """k = round(sample_time_step / time_step) recording vs np.interp: < 1e-9 rad."""

def test_direct_step_recording_differs_with_a_probe(asset):
    """And it must differ once there is 40-160 Hz content - that is the point."""

def test_control_decimation_preserves_the_analytic_residual(asset):
    """Rigid tier vs Pinocchio inverse dynamics stays below 1e-7 Nm at decimation 8."""

def test_parallel_generation_is_bit_identical(small_config, asset):
    """--jobs 4 == --jobs 1."""

def test_parquet_round_trips(tmp_path, small_config, asset):
    frame, manifest, comparison = generate(small_config, asset, verbose=False)
    path, *_ = write_dataset(frame, manifest, tmp_path / "d.parquet", comparison)
    assert pd.read_parquet(path).equals(frame)
```

---

## F. What the existing suite already locks in — do not regress it

`tests/test_identification_dataset.py` currently guarantees, and every change in
this round must leave intact:

- MuJoCo / Pinocchio agreement to 1e-6;
- the transmission mode using the rotor/link reduced inertia, damping reproducing
  the sampled ratio;
- robot sampling staying inside its intervals, log-uniform, seed-stable;
- backend comparison pairing bags and flagging a divergent backend;
- the regressor reproducing inverse dynamics; 43 rigid and 57 friction-augmented
  base parameters;
- excitation endpoint conditions and limit compliance;
- the optimized trajectory beating the point-to-point baseline;
- the recorded torque being the applied torque;
- feedback under 5 % of feedforward;
- Newton / MuJoCo agreement to 1e-4 rad; near-rigid convergence;
- **the acceptance test: base parameters recovered from a generated rollout to
  better than 1e-4 relative error.**

The last one is the round's non-negotiable invariant. If any Stage C or D change
moves it, the change is wrong — not the test.
