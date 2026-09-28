"""Round 5 closure (`R5_10`): instability guard, speed work, backend audit.

Cheap checks unmarked; the one that integrates physics is marked ``slow``.
The speed work (T-3) is held to "same bytes out" by the byte-identical
parquet check in the report, not here; these tests pin the pieces that make
it safe: the early-exit collision check returns the same verdict, the
Fourier-basis cache returns the same bits, and the trajectory cache is reused
across matched rows.
"""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.path.join(_REPO, "src"))
sys.path.insert(0, os.path.join(_REPO, "scripts"))

from elastic_sim.assets import AssetRegistry
from elastic_sim.backend_comparison import ComparisonThresholds, compare_backends
from elastic_sim.excitation import evaluate_series
from elastic_sim.kinematics import MAX_BISECTION_LEVELS, PortableKinematics


@pytest.fixture(scope="module")
def registry():
    return AssetRegistry.for_repository(_REPO)


@pytest.fixture(scope="module")
def iiwa(registry):
    return registry.load("kuka_lbr_iiwa_14_r820_table")


# ---------------------------------------------------------------------------
# T-2: validate_path can no longer hang on a diverged path
# ---------------------------------------------------------------------------

def test_validate_path_refuses_an_unphysical_jump(iiwa):
    kinematics = PortableKinematics(iiwa)
    home = np.asarray(iiwa.metadata["default_configuration"], dtype=float)
    jump = 0.05 * 2 ** (MAX_BISECTION_LEVELS + 1)
    with pytest.raises(ValueError, match="not physical"):
        kinematics.validate_path(np.vstack([home, home + jump]), max_joint_step=0.05)
    with pytest.raises(ValueError, match="non-finite"):
        kinematics.validate_path(np.vstack([home, home * np.nan]), max_joint_step=0.05)


def test_state_out_of_range_flags_a_runaway_joint(registry):
    from elastic_sim.dataset import _state_out_of_range

    asset = registry.load("ur10_table")
    n = len(asset.joint_names)
    time = np.linspace(0.0, 1.0, 11)
    fine = {"time": time, "q_link": np.zeros((11, n)), "q_motor": np.zeros((11, n))}
    assert _state_out_of_range(asset, fine) is None
    runaway = dict(fine, q_motor=fine["q_motor"].copy())
    runaway["q_motor"][7, 2] = 1.0e3
    record = _state_out_of_range(asset, runaway)
    assert record["kind"] == "joint_range" and record["joint"] == asset.joint_names[2]
    assert record["time"] == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# T-3: the speed work returns the same answers
# ---------------------------------------------------------------------------

def test_early_exit_collision_check_gives_the_same_verdict(iiwa):
    kinematics = PortableKinematics(iiwa)
    home = np.asarray(iiwa.metadata["default_configuration"], dtype=float)
    penetrating = np.array([0.0, -2.09, 0.0, 1.219, 0.0, 0.0, 0.0])
    path = np.repeat(home[None, :], 1001, axis=0)
    ramp = np.linspace(0.0, 1.0, 40)
    path[480:520] = home + np.outer(ramp, penetrating - home)
    path[520:560] = penetrating + np.outer(ramp, home - penetrating)
    full = kinematics.validate_path(path, margin=0.01, max_joint_step=0.05)
    early = kinematics.validate_path(path, margin=0.01, max_joint_step=0.05, stop_at_first_invalid=True)
    assert full.valid is early.valid is False
    assert early.checked_configurations < full.checked_configurations
    clear = np.repeat(home[None, :], 1001, axis=0)
    assert (kinematics.validate_path(clear, margin=0.01, stop_at_first_invalid=True)
            == kinematics.validate_path(clear, margin=0.01))


def test_fourier_basis_cache_is_bit_identical():
    rng = np.random.default_rng(3)
    a, b = rng.normal(size=(4, 7)), rng.normal(size=(4, 7))
    offset, omega = rng.normal(size=4), 2.0 * np.pi * 0.1
    time = np.arange(0.0, 10.0 + 1e-4, 2e-3)
    indices = np.array([1, 2, 3, 4, 5, 40, 90])
    scaled = indices * omega
    phase = np.outer(time, scaled)
    sin, cos = np.sin(phase), np.cos(phase)
    expected = (
        sin @ (a / scaled).T - cos @ (b / scaled).T + offset,
        cos @ a.T + sin @ b.T,
        -(sin @ (a * scaled).T) + cos @ (b * scaled).T,
    )
    for _ in range(2):  # cold, then from the cache
        for got, want in zip(evaluate_series(a, b, offset, omega, time, indices=indices), expected):
            assert np.array_equal(got, want)


# ---------------------------------------------------------------------------
# T-4: the backend comparison reads the clean columns
# ---------------------------------------------------------------------------

def _pair_bag(bag: str, backend: str, clean: np.ndarray, noise: np.ndarray) -> pd.DataFrame:
    n = clean.shape[1]
    frame = {"t": np.arange(len(clean)) * 0.002, "bag": bag, "backend": backend, "tier": "rigid"}
    for j in range(n):
        for noisy, exact in (("q", "q_link_clean"), ("dq", "dq_link_clean"), ("q_motor", "q_motor_clean"),
                             ("ft", "tau_link_clean"), ("tau", "tau_motor_clean")):
            frame[f"{exact}{j}"] = clean[:, j]
            frame[f"{noisy}{j}"] = clean[:, j] + noise[:, j]
    return pd.DataFrame(frame)


def test_backend_comparison_is_not_fooled_by_independent_noise():
    """`R5_09` Sec 2's sqrt(2): same physics, two noise draws (`R5_10` T-4)."""
    rng = np.random.default_rng(0)
    clean = np.sin(np.linspace(0.0, 6.0, 500))[:, None] * np.ones((1, 3)) * 1e-2
    frame = pd.concat([
        _pair_bag("t0_rigid_f0_mujoco", "mujoco", clean, rng.normal(0.0, 0.1, clean.shape)),
        _pair_bag("t0_rigid_f0_newton", "newton", clean, rng.normal(0.0, 0.1, clean.shape)),
    ], ignore_index=True)
    row = compare_backends(frame, (), ComparisonThresholds()).iloc[0]
    assert row["columns"] == "clean" and row["pass"]
    assert row["ft_relative_rms"] == 0.0
    # Without clean columns the same pair reads ~sqrt(2) and fails, which is
    # the historical behaviour the fallback keeps for old datasets.
    measured_only = frame.drop(columns=[c for c in frame.columns if "_clean" in c])
    old = compare_backends(measured_only, (), ComparisonThresholds()).iloc[0]
    assert old["columns"] == "measured" and not old["pass"]
    assert old["ft_relative_rms"] == pytest.approx(np.sqrt(2.0), rel=0.15)


# ---------------------------------------------------------------------------
# T-2 + T-3 on the real reproduction (slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_iiwa_rigid_pd_bag_is_excluded_listed_and_does_not_hang(monkeypatch, iiwa):
    """The weekend's F-1 bag, end to end, plus the trajectory cache across modes."""
    import elastic_sim.dataset as dataset
    import diagnose_controller_modes as dcm

    config = dataset.load_config(_REPO / "config/identification/kuka_lbr_iiwa_14_r820_table_round5.yaml")
    config = dcm._shrink(config, trajectories=1, robots=1, backends=("mujoco",))
    import elastic_sim.dataset_worklist as worklist

    calls = []
    original = worklist.optimize_excitation
    monkeypatch.setattr(worklist, "optimize_excitation",
                        lambda *a, **k: calls.append(1) or original(*a, **k))
    cache: dict = {}
    digests = []
    for mode in ("pd", "velocity_pi"):
        mode_config = replace(config, controller=replace(config.controller, mode=mode))
        with pytest.warns(UserWarning, match="unstable"):
            frame, manifest, _ = dataset.generate(mode_config, iiwa, verbose=False, jobs=1,
                                                  trajectory_cache=cache)
        unstable = manifest["unstable_bags"]
        assert [u["bag"] for u in unstable] == ["t0_rigid_f0_mujoco"]
        assert unstable[0]["kind"] == "mjWARN_BADQACC" and unstable[0]["joint"].startswith("iiwa_A")
        assert "t0_rigid_f0_mujoco" not in set(frame["bag"])
        assert [r["bag"] for r in manifest["records"]] == ["t0_e00_f0_mujoco"]
        digests.append(manifest["trajectory_digests"])
    assert digests[0] == digests[1]
    assert len(calls) == len(cache) == 2, "each trajectory optimized once across both modes"

