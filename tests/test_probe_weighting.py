"""RR_20 T3: ``FourierExcitationConfig.probe_weighting``.

``dq = sum a_k cos + b_k sin``, so line ``k``'s acceleration amplitude is
``k w hypot(a_k, b_k)``. The default (``position``) draw is velocity-flat and
its acceleration grows as ``k``; ``acceleration`` makes it flat.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from elastic_sim.assets import AssetRegistry
from elastic_sim.excitation import FourierExcitationConfig, sample_candidate

#: the RR_20 iiwa comb on the 0.2 Hz grid, 2.0 ... 23.2 Hz
IIWA_P1 = (10, 12, 13, 15, 16, 18, 21, 24, 27, 30, 34, 38, 44, 50, 56, 63, 72, 80, 91, 103, 116)


@pytest.fixture(scope="module")
def iiwa():
    return AssetRegistry.for_repository().load("kuka_lbr_iiwa_14_r820")


def _config(weighting: str) -> FourierExcitationConfig:
    return FourierExcitationConfig(n_harmonics=5, base_frequency=0.2, n_periods=2, time_step=1e-3,
                                   max_acceleration=4.0, velocity_fraction=0.8, centre_jitter=1.0,
                                   probe_harmonics=IIWA_P1, probe_acceleration_fraction=0.25,
                                   probe_weighting=weighting)


def _line_rms(asset, config, seeds=100):
    time = np.arange(0.0, config.duration + 0.5 * config.time_step, config.time_step)
    omega = 2.0 * np.pi * config.base_frequency
    k = np.asarray(config.probe_harmonics, dtype=float)
    amplitude = []
    for seed in range(seeds):
        a, b, _ = sample_candidate(asset, config, np.random.default_rng(seed), time)
        amplitude.append(k * omega * np.hypot(a[:, 5:], b[:, 5:]))
    return np.sqrt(np.mean(np.asarray(amplitude) ** 2, axis=0))  # joints x lines


def test_acceleration_weighting_is_flat_and_position_is_not(iiwa):
    flat = _line_rms(iiwa, _config("acceleration"))
    rising = _line_rms(iiwa, _config("position"))
    for joint in range(flat.shape[0]):
        assert flat[joint].max() / flat[joint].min() < 1.6
        # velocity-flat: low lines ~ k_min / k_max of the top line's acceleration
        assert rising[joint, :3].mean() / rising[joint, -3:].mean() < 0.25
    # the same budget, so the lowest lines gain several-fold (RR_20 S3.1)
    assert np.all(flat[:, :3].mean(axis=1) > 3.0 * rising[:, :3].mean(axis=1))


def test_acceleration_weighting_keeps_the_boundary_constraints(iiwa):
    config = _config("acceleration")
    time = np.arange(0.0, config.duration + 0.5 * config.time_step, config.time_step)
    k = np.asarray(IIWA_P1, dtype=float)
    a, b, _ = sample_candidate(iiwa, config, np.random.default_rng(3), time)
    # each block independently at rest at t = 0: dq(0) = sum a, ddq(0) ~ sum k b
    assert np.abs(a[:, 5:].sum(axis=1)).max() < 1e-12
    assert np.abs(b[:, 5:] @ k).max() < 1e-9
    assert np.abs(a[:, :5].sum(axis=1)).max() < 1e-12


def test_the_seed_fixes_the_draw(iiwa):
    time = np.arange(0.0, 10.0 + 0.5e-3, 1e-3)
    flat = sample_candidate(iiwa, _config("acceleration"), np.random.default_rng(7), time)
    again = sample_candidate(iiwa, _config("acceleration"), np.random.default_rng(7), time)
    assert all(np.array_equal(x, y) for x, y in zip(flat, again))


def test_position_is_the_default_and_unknown_weightings_are_refused():
    assert FourierExcitationConfig().probe_weighting == "position"
    with pytest.raises(ValueError, match="probe_weighting"):
        replace(_config("position"), probe_weighting="velocity")


def test_metadata_records_only_a_non_default_weighting(iiwa):
    pytest.importorskip("pinocchio")
    from elastic_sim.excitation import optimize_excitation

    short = dict(n_periods=1, time_step=2e-3)
    flat = optimize_excitation(iiwa, replace(_config("acceleration"), **short), seed=1, n_candidates=2)
    plain = optimize_excitation(iiwa, replace(_config("position"), **short), seed=1, n_candidates=2)
    assert flat.metadata["probe_weighting"] == "acceleration"
    assert "probe_weighting" not in plain.metadata  # a position trajectory keeps its metadata bytes
    assert flat.metadata["probe_top_hz"] == pytest.approx(23.2)
