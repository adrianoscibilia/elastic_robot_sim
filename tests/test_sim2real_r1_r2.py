"""S2R_08 P-4b: the runner hooks (T-3.1), the static equilibrium, R1 (T-3.3),
R2 (T-3.4), the F-9 spectra.

Short synthetic bags (a smooth multi-sine about a pose) keep the unmarked
tests to seconds; the round-6 bit-identity regression and the stiff G-3
check are ``slow``.
"""

from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.path.join(_REPO, "src"))
sys.path.insert(0, os.path.join(_REPO, "tests"))
sys.path.insert(0, os.path.join(_REPO, "scripts"))

from elastic_sim.sim2real import replay_r1, replay_r2, spectra
from elastic_sim.sim2real.config import ConfigError, R1Settings, R1Windows, load_config
from elastic_sim.sim2real.plant import PlantBase, PlantParameters
from elastic_sim.sim2real.real_runs import ReplayBag

BASE_CONFIG = _REPO / "config" / "sim2real" / "kuka_lbr_iiwa_14_r820_base.yaml"
POSE = np.array([0.3, 0.6, -0.2, -1.2, 0.4, 0.8, 0.1])
#: T-3.1 regression: a short seeded round-6 iiwa drive bag (test_round6._run),
#: digested before the hooks were added (S2R_09 part B).
ROUND6_DIGESTS = {"e00": "08e8f758fdd8a45a1886628a47832f239d5cfa9fd58632cbc4ab0e6037b044df",
                  "rigid": "fd612d2d7a5193d13bfd82e11b0ce797920f5a799ab8bd5a3415bc182ce25adc"}


@pytest.fixture(scope="module")
def config():
    return load_config(BASE_CONFIG)


@pytest.fixture(scope="module")
def base(config):
    return PlantBase.from_config(config)


def _bag(duration: float = 1.0, *, amplitude: float = 0.05, kind: str = "excitation", seed: int = 0) -> ReplayBag:
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, duration + 0.5e-3, 1e-3)
    phase = rng.uniform(0, 2 * np.pi, 7)
    window = np.sin(np.pi * np.clip(t / duration, 0, 1))[:, None] ** 2   # starts and ends at rest
    q = POSE + amplitude * window * (np.sin(2 * np.pi * 0.5 * t[:, None] + phase)
                                     + 0.3 * np.sin(2 * np.pi * 2.0 * t[:, None] + 2 * phase))
    q = np.round(q / replay_r1.ENCODER_QUANTUM) * replay_r1.ENCODER_QUANTUM   # the encoder grid
    zeros = np.zeros_like(q)
    return ReplayBag(run_id="S", bag=f"{kind}_0", kind=kind, t=t, q=q, dq=zeros, tau_cmd=zeros, ft=zeros,
                     cmd_echo=q.copy())


# ---------------------------------------------------------------------------
# T-3.1: hooks and equilibrium
# ---------------------------------------------------------------------------


def test_initial_state_with_no_fields_is_the_default_rollout(base):
    """``InitialState()`` (every field None) and ``model_setup`` that sets
    theta0 reproduce the default rollout bit for bit."""
    from elastic_sim.sim2real.plant import BodyMap
    from elastic_sim.torque_runners import InitialState, TransmissionSpec, run_mujoco_elastic_torque

    bag = _bag(0.05)
    plant = PlantParameters.nominal(base)
    reference, _ = replay_r1.spline_reference(bag.t, bag.q, base.joint_names, time_step=1e-4)
    transmission = TransmissionSpec.from_damping_ratio(plant.transmission.stiffness, 0.1,
                                                       plant.transmission.rotor_inertia, np.ones(7))

    def run(**hooks):
        controller = replay_r1.MotorForcing(reference, transmission.rotor_inertia, plant.motor_friction)
        return run_mujoco_elastic_torque(base.asset, reference, controller, transmission, time_step=1e-4,
                                         friction=plant.motor_friction, check_step=False, **hooks)

    default = run()
    hooked = run(initial_state=InitialState(),
                 model_setup=lambda mj, model: plant.apply_to_mujoco(mj, model, BodyMap.build(mj, model, base)))
    for key in ("q_link", "q_motor", "tau_link", "tau_motor"):
        assert np.array_equal(default[key], hooked[key]), key


def test_static_equilibrium_matches_the_rigid_gravity(base):
    """R1's start: residual <= 1e-9 N m (T-3.1), and the spring torque is
    the rigid chain's gravity at the deflected link pose (R0's model, with
    the intermediate bodies) -- the two engines agree on the statics."""
    from elastic_sim.sim2real.replay_r0 import R0Bag, R0Engine

    bag = _bag(0.2, amplitude=0.0, kind="hold")
    plant = PlantParameters.nominal(base)
    task = replay_r1.R1Task(replay_r1.prepare_bags([bag], dataclasses.replace(load_config(BASE_CONFIG).replay.r0,
                                                                               hold_decimate=1))[0],
                            0, len(bag.t), 0, len(bag.t), False)
    out = replay_r1.rollout(task, plant, R1Settings())
    record = out["diagnostics"]["equilibrium"]
    assert record["kind"] == "static" and record["residual_max_nm"] <= 1e-9
    spring = np.asarray(record["spring_torque"])
    delta = spring / plant.transmission.stiffness
    q_link = bag.q[:1] - delta
    zero = np.zeros_like(q_link)
    engine = R0Engine(base, [R0Bag("S", "h", "hold", "", np.arange(1), bag.t[:1], q_link, zero, zero, zero)],
                      deflection=False)
    _, tau, _ = engine.simulate(plant)
    assert np.allclose(spring, tau[0], atol=1e-6), spring - tau[0]


@pytest.mark.slow
def test_round6_bag_is_bit_identical_after_the_hooks():
    """S2R_01 T-3.1 done-when: a fixed small round-6 bag (seeded) is
    bit-identical before and after the hooks."""
    import hashlib

    from test_round6 import CONFIGS, _run

    from elastic_sim.assets import AssetRegistry
    from elastic_sim.dataset import load_config as load_dataset_config

    registry = AssetRegistry.for_repository(_REPO)
    config = load_dataset_config(_REPO / CONFIGS["iiwa_drive"], check_bounds=False)
    asset = registry.load(config.asset)
    for tier, expected in ROUND6_DIGESTS.items():
        _, result, frame = _run(config, asset, tier)
        digest = hashlib.sha256()
        for key in sorted(result):
            value = result[key]
            if isinstance(value, np.ndarray) or (isinstance(value, list) and value
                                                 and isinstance(value[0], (np.ndarray, float))):
                digest.update(key.encode())
                digest.update(np.ascontiguousarray(np.asarray(value, dtype=float)).tobytes())
        digest.update(frame.select_dtypes("number").to_numpy(dtype=float).tobytes())
        assert digest.hexdigest() == expected, tier


# ---------------------------------------------------------------------------
# R1
# ---------------------------------------------------------------------------


def test_discrepancy_spline_fits_to_the_noise_level():
    rng = np.random.default_rng(1)
    t = np.arange(0.0, 2.0, 1e-3)
    clean = 0.3 * np.sin(2 * np.pi * 0.7 * t) + 0.01 * np.sin(2 * np.pi * 20 * t)
    sigma = 2e-6
    spline, _ = replay_r1.discrepancy_spline(t, clean + sigma * rng.standard_normal(t.size), sigma)
    fit = spline(t)
    assert np.sqrt(np.mean((fit - clean) ** 2)) < sigma            # it removes noise ...
    assert np.max(np.abs(fit - clean)) < 10 * sigma                 # ... and keeps the 20 Hz motion


def test_r1_forces_the_measured_motor_motion(base, config):
    """T-3.3: the motor follows the reference to <= 10 encoder quanta; ft_sim
    is the observation model of tau_s; the result is on R0's rows."""
    bag = _bag(1.0)
    engine = replay_r1.R1Engine(base, replay_r1.prepare_bags([bag], config.replay.r0), config.replay.r1)
    plant = PlantParameters.from_theta(base, {"sensor.ft.iiwa_A2.gain": 1.1, "sensor.ft.iiwa_A2.offset@S": 0.3},
                                       runs=("S",))
    result = engine.replay(plant)[0]
    check = replay_r1.tracking_check([result])
    assert check["pass"] and check["worst_quanta"] <= 10.0
    assert np.array_equal(result.rows, np.arange(0, len(bag.t), config.replay.r0.decimate))
    assert np.allclose(result.ft_sim[:, 1], 1.1 * result.tau_s[:, 1] + 0.3)
    assert result.mode == "R1" and result.extra["equilibrium"]["residual_max_nm"] <= 1e-9


def test_r1_windows_tile_the_bag(base, config):
    bag = replay_r1.prepare_bags([_bag(3.0)], config.replay.r0)[0]
    settings = dataclasses.replace(config.replay.r1, windows=R1Windows(length_s=1.2, stride_s=0.7, burn_in_s=0.5))
    engine = replay_r1.R1Engine(base, [bag], settings)
    scored = np.concatenate([np.arange(t.score_from, t.score_to) for t in engine.tasks])
    assert np.array_equal(scored, np.arange(len(bag.t)))                  # every row once
    assert engine.tasks[0].quasi_static is False and all(t.quasi_static for t in engine.tasks[1:])
    with pytest.raises(ValueError, match="tile"):
        replay_r1.R1Engine(base, [bag], dataclasses.replace(
            settings, windows=R1Windows(length_s=1.0, stride_s=0.8, burn_in_s=0.5)))


@pytest.mark.slow
def test_r1_stiff_limit_matches_r0_below_10_hz(base, config):
    """G-3's engine check on a synthetic bag: with K x 100, R1 equals R0 on
    R1's own kinematics below 10 Hz (S2R_09 part B)."""
    from scipy.signal import butter, sosfiltfilt

    from elastic_sim.sim2real.replay_r0 import R0Bag, R0Engine

    bag = _bag(1.5)
    r0 = dataclasses.replace(config.replay.r0, decimate=1)
    engine = replay_r1.R1Engine(base, replay_r1.prepare_bags([bag], r0), config.replay.r1, stiffness_scale=100.0)
    plant = PlantParameters.nominal(base)
    result = engine.replay(plant)[0]
    stiff = dataclasses.replace(plant, transmission=dataclasses.replace(
        plant.transmission, stiffness=plant.transmission.stiffness * 100.0))
    kin = R0Bag("S", "x", "excitation", "", result.rows, result.t, result.extra["q_ref"], result.dq, result.ddq,
                result.ft)
    _, tau, _ = R0Engine(base, [kin]).simulate(stiff)
    sos = butter(4, 10.0, fs=1000.0, output="sos")
    low = sosfiltfilt(sos, result.tau_s, axis=0) - sosfiltfilt(sos, tau, axis=0)
    assert np.max(np.sqrt(np.mean(low[200:-200] ** 2, axis=0))) <= 0.05


def test_r1_config_keys(tmp_path):
    def stage(**r1):
        path = tmp_path / "stage.yaml"
        path.write_text(yaml.safe_dump({"schema": "elastic_sim.sim2real/1", "extends": str(BASE_CONFIG),
                                        "replay": {"r1": r1}}))
        return path

    loaded = load_config(stage(reference_filter="savgol", extras="plant"))
    assert loaded.replay.r1.reference_filter == "savgol" and loaded.replay.r1.extras == "plant"
    with pytest.raises(ConfigError, match="reference_filter"):
        load_config(stage(reference_filter="gcv"))
    with pytest.raises(ConfigError, match="extras"):
        load_config(stage(extras="all"))


# ---------------------------------------------------------------------------
# R2
# ---------------------------------------------------------------------------


def test_command_reference_is_the_echo_delayed():
    t = np.arange(0.0, 0.5, 1e-3)
    command = np.column_stack([np.sin(2 * np.pi * 2 * t + k) for k in range(7)])
    from elastic_sim.sim2real.controllers import command_reference

    reference = command_reference(t, command, [f"j{k}" for k in range(7)], delay_cycles=3, period=1e-3,
                                  time_step=2.5e-4)
    for k in (10, 100, 400):
        q, dq, _ = reference(k * 1e-3)
        assert np.allclose(q, command[k - 3], atol=1e-12)
    q, dq, ddq = reference(1e-3)          # before the first delayed sample: the first command, at rest
    assert np.allclose(q, command[0]) and not dq.any() and not ddq.any()


def test_r2_runs_the_closed_loop(base, config):
    """T-3.4 at nominal theta and a hand-set bandwidth: no divergence; the
    simulated tracking error is reported next to the real one."""
    bag = _bag(1.0)
    shifted = np.vstack([bag.q[:2], bag.q[:-2]])        # a robot 2 ms behind its command
    bag = dataclasses.replace(bag, q=shifted)
    engine = replay_r2.R2Engine(base, replay_r2.prepare_bags([bag], config.replay.r0), config.replay.r2)
    plant = PlantParameters.from_theta(base, replay_r2.nominal_controller_theta(base, bandwidth_hz=10.0),
                                       runs=("S",))
    results = engine.replay(plant)
    assert len(results) == 1 and "error" not in engine.last_outputs[0]
    tracking = results[0].extra["tracking"]
    assert all(np.isfinite(tracking["sim_rms"])) and max(tracking["sim_rms"]) < 5e-3
    assert np.all(np.asarray(tracking["real_rms"]) > 0)
    assert results[0].extra["equilibrium"]["residual_max_nm"] <= 1e-9


# ---------------------------------------------------------------------------
# F-9 spectra
# ---------------------------------------------------------------------------


def test_explained_fraction_and_band_powers():
    rng = np.random.default_rng(3)
    n = 200_000   # ~200 Welch segments: chance coherence ~ 1/200
    signal = rng.standard_normal((n, 2))
    residual = np.column_stack([2.0 * signal[:, 0], rng.standard_normal(n)])
    fraction = spectra.explained_fraction([residual], [signal])
    assert np.all(fraction[1:, 0] > 0.99) and np.all(fraction[1:, 1] < 0.05)
    f, psd = spectra.psd_sum([residual])
    total = spectra.band_powers(f, psd).sum(axis=0) / n
    assert np.allclose(total, residual.var(axis=0), rtol=0.05)


def test_order_fraction_finds_the_ripple():
    t = np.arange(0.0, 10.0, 1e-3)
    q = np.column_stack([0.5 * np.sin(2 * np.pi * 0.2 * t)] * 2)
    rng = np.random.default_rng(4)
    ripple = 0.2 * np.sin(320.0 * q[:, 0] + 0.7)
    residual = np.column_stack([ripple + 0.1 * rng.standard_normal(t.size), 0.1 * rng.standard_normal(t.size)])
    fraction = spectra.order_fraction([residual], [q])
    assert fraction[0] > 0.5 and fraction[1] < 0.01
