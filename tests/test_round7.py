"""FR_04 S-1..S-4: payload modes, the no-payload identity checks, audit columns,
transmission priors, the two-mass reference filter and its baselines.

Cheap checks unmarked; the builds marked ``slow`` (a shrunk FMRR config on a
short excitation, both payload modes).
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.path.join(_REPO, "src"))
sys.path.insert(0, os.path.join(_REPO, "scripts"))

from elastic_sim.dataset import load_config
from elastic_sim.dataset_config import PayloadSampling
from elastic_sim.dataset_worklist import attached_payload
from elastic_sim.payload import Payload
from elastic_sim.round7 import AUDIT_FAMILIES, audit_columns, transmission_priors
from elastic_sim.two_mass import burn_in_seconds, damping_from_ratio, filter_sequence, mean_diagonal

R6 = "config/identification/fmrr_tecnobody_round6.yaml"
R7 = {
    "fmrr": ("config/identification/fmrr_tecnobody_round7_nopayload.yaml", R6),
    "ur10": ("config/identification/ur10_table_round7_nopayload.yaml", "config/identification/ur10_table_round6.yaml"),
    "iiwa_drive": ("config/identification/kuka_lbr_iiwa_14_r820_table_drive_round7_nopayload.yaml",
                   "config/identification/kuka_lbr_iiwa_14_r820_table_round6_drive.yaml"),
    "iiwa_bus": ("config/identification/kuka_lbr_iiwa_14_r820_table_bus_round7_nopayload.yaml",
                 "config/identification/kuka_lbr_iiwa_14_r820_table_round6_bus.yaml"),
}


def _write(tmp_path: Path, raw: dict) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# S-1: payload.mode
# ---------------------------------------------------------------------------

def test_payload_mode_defaults_to_random_and_parses(tmp_path):
    raw = yaml.safe_load((_REPO / R6).read_text(encoding="utf-8"))
    assert load_config(_REPO / R6, check_bounds=False).payload.mode == "random"
    raw["payload"]["mode"] = "fixed"
    raw["payload"]["fixed"] = {"mass": 1.5, "offset": [0.0, 0.0, -1.08], "size": 0.1}
    config = load_config(_write(tmp_path, raw), check_bounds=False)
    assert (config.payload.mode, config.payload.fixed_mass, config.payload.fixed_offset) == ("fixed", 1.5, (0.0, 0.0, -1.08))
    raw["payload"]["mode"] = "sometimes"
    with pytest.raises(ValueError, match="payload.mode"):
        load_config(_write(tmp_path, raw), check_bounds=False)
    raw["payload"]["mode"] = "fixed"
    raw["payload"].pop("fixed")
    with pytest.raises(ValueError, match="payload.fixed"):
        load_config(_write(tmp_path, raw), check_bounds=False)


@pytest.mark.parametrize("variant", list(R7))
def test_round7_configs_differ_from_round6_only_in_payload_mode_and_output(variant):
    r7, r6 = (yaml.safe_load((_REPO / path).read_text(encoding="utf-8")) for path in R7[variant])
    assert r7["payload"].pop("mode") == "none"
    assert r7["dataset"].pop("output") != r6["dataset"].pop("output")
    assert r7 == r6
    config = load_config(_REPO / R7[variant][0], check_bounds=False)
    assert config.payload.mode == "none" and config.payload.enabled


def test_attached_payload_by_mode():
    drawn = Payload(mass=2.0, offset=(0.0, 0.0, 0.1), size=0.1)
    config = load_config(_REPO / R6, check_bounds=False)
    elastic = next(t for t in config.tiers if not t.is_rigid)
    rigid = next(t for t in config.tiers if t.is_rigid)
    assert attached_payload(config, drawn, elastic) is drawn
    none = replace(config, payload=replace(config.payload, mode="none"))
    assert attached_payload(none, drawn, elastic).is_empty
    fixed = replace(config, payload=replace(config.payload, mode="fixed", fixed_mass=1.0, fixed_size=0.08))
    body = attached_payload(fixed, drawn, elastic)
    assert (body.mass, body.size) == (1.0, 0.08)
    assert attached_payload(fixed, Payload(), rigid).is_empty


def test_identity_checker_passes_equal_and_names_what_differs():
    from round7_checks import compare

    robot = {"name": "e00", "stiffness": [1.0, 2.0], "damping": [0.1, 0.2], "damping_ratio": [0.5, 0.5],
             "rotor_inertia": [1.0, 1.0], "mode_frequency_hz": [3.0, 4.0], "time_step": 1e-3,
             "payload": {"mass": 1.0, "offset": [0, 0, 0], "size": 0.1}}
    record = {"bag": "t0_e00_f0_mujoco", "tier": "e00", "split": "train", "trajectory": 0, "payload": robot["payload"],
              "viscous": [1.0, 1.0], "coulomb": [0.5, 0.5], "trajectory_digest": "abc", **{k: robot[k] for k in
              ("stiffness", "damping", "damping_ratio", "rotor_inertia", "mode_frequency_hz", "time_step")}}
    r6 = {"trajectory_digests": {"e00/t0": "abc"}, "robots": [robot], "records": [record]}
    r7 = json.loads(json.dumps(r6))
    r7["payload_mode"] = {"mode": "none", "attached": None}
    r7["records"][0]["payload"] = None
    assert compare(r6, r7)["ok"]
    r7["robots"][0]["damping"] = [0.1, 0.2000001]
    r7["records"][0]["viscous"] = [1.0, 1.1]
    report = compare(r6, r7)
    assert not report["ok"]
    differing = {c["check"]: c["differing"] for c in report["checks"]}
    assert any("e00.damping" in d for d in differing.values())
    assert any("t0_e00_f0_mujoco.viscous" in d for d in differing.values())


# ---------------------------------------------------------------------------
# The reference filter
# ---------------------------------------------------------------------------

def test_burn_in_rule():
    # max(3 / (0.05 * 2 pi * 10), 2 / 10) = 0.9549 -> 1.0 (rounded up to 0.05)
    assert burn_in_seconds(0.05, 10.0) == 1.0
    assert burn_in_seconds(1.0, 2.0) == 1.0       # 2 / f wins: 0.239 vs 1.0
    assert burn_in_seconds(0.05, 3.0 / (0.05 * 2 * math.pi * 0.6)) == 0.6   # exact multiple stays


@pytest.mark.parametrize("coupling", ["diagonal", "full"])
@pytest.mark.parametrize("integrator", ["trapezoidal", "backward_euler"])
def test_filter_is_exact_on_a_quasi_static_input(coupling, integrator):
    rng = np.random.default_rng(0)
    n, steps = 3, 400
    tau = np.tile(rng.normal(size=n) * 20.0, (steps, 1))
    inertia = np.tile(np.diag([2.0, 1.0, 0.5]), (steps, 1, 1))
    out = filter_sequence(tau, inertia, np.array([1e4, 5e3, 2e3]), np.array([10.0, 5.0, 2.0]), 1e-3,
                          coupling=coupling, integrator=integrator)
    assert np.max(np.abs(out["tau_s"] - tau)) < 1e-9 * np.max(np.abs(tau))


def _chirp_response(integrator: str, coupling: str, h: float = 1e-3) -> float:
    """Relative RMSE of the filter's tau_s against a tight-tolerance solution of the same oscillator."""
    from scipy.integrate import solve_ivp
    from scipy.signal import chirp

    m, f_mode, zeta, end = 1.0, 2.0, 0.05, 8.0
    k = m * (2 * np.pi * f_mode) ** 2
    d = 2 * zeta * np.sqrt(k * m)
    time = np.arange(0.0, end + h / 2, h)
    drive = lambda t: 10.0 * chirp(t, 0.2, end, 5.0) * np.minimum(t, 1.0) ** 3  # noqa: E731  starts at rest
    exact = solve_ivp(lambda t, x: [x[1], (drive(t) - d * x[1] - k * x[0]) / m], (0, end), [0.0, 0.0],
                      t_eval=time, rtol=1e-11, atol=1e-13, method="DOP853")
    reference = k * exact.y[0] + d * exact.y[1]
    inertia = np.full((len(time), 1, 1), m) if coupling == "full" else np.full((len(time), 1), m)
    out = filter_sequence(drive(time)[:, None], inertia, np.array([k]), np.array([d]), h,
                          coupling=coupling, integrator=integrator)["tau_s"][:, 0]
    return float(np.sqrt(np.mean((out - reference) ** 2)) / np.sqrt(np.mean(reference ** 2)))


@pytest.mark.parametrize("coupling", ["diagonal", "full"])
def test_trapezoidal_filter_reproduces_the_two_mass_chirp_response(coupling):
    assert _chirp_response("trapezoidal", coupling) < 1e-3


def test_backward_euler_misses_the_modal_tolerance():
    """Why the default is trapezoidal: first order, ~omega h / 2 (FR_04 implementation report)."""
    assert _chirp_response("backward_euler", "diagonal") > 1e-2


def test_damping_from_ratio_and_mean_diagonal():
    inertia = np.stack([np.diag([1.0, 4.0]), np.diag([3.0, 4.0])])
    assert np.allclose(mean_diagonal(inertia), [2.0, 4.0])
    assert np.allclose(damping_from_ratio([100.0, 25.0], [0.5, 0.1], [1.0, 4.0]), [10.0, 2.0])


# ---------------------------------------------------------------------------
# S-2 / S-3 contract blocks
# ---------------------------------------------------------------------------

def test_audit_columns_are_declared_never_inputs():
    block = audit_columns(3)
    assert block["columns"] == [f"{family}0..{family}2" for family in AUDIT_FAMILIES]
    assert "never a model input" in block["use"]


def test_transmission_priors_are_the_filter_side_ranges(tmp_path):
    from elastic_sim.assets import AssetRegistry

    asset = AssetRegistry.for_repository(_REPO).load("fmrr_tecnobody")
    k = np.array([[4000.0, 3000.0, 2000.0], [8000.0, 6000.0, 4000.0]])
    m = np.array([70.0, 17.0, 9.0])
    zeta = np.array([[0.05, 0.1, 0.5], [0.1, 0.2, 1.0]])
    robots = [{"name": f"e0{i}", "stiffness": k[i].tolist(), "damping": (2 * zeta[i] * np.sqrt(k[i] * m)).tolist(),
               "plant_link_inertia_median": m.tolist(), "mode_frequency_hz": [9.0, 9.0, 9.0]} for i in range(2)]
    manifest = {"joint_names": ["joint_y", "joint_x", "joint_z"], "robots": robots, "sample_time_step": 0.004,
                "transmission_sampling": {"stiffness_nominal": [5000.0, 4000.0, 3000.0], "stiffness_factor": [0.5, 2.0],
                                          "damping_ratio": [0.4, 1.2]}}
    priors = transmission_priors(manifest, asset)
    first = priors["joints"][0]
    assert first["stiffness_range"] == [2500.0, 10000.0]
    assert np.allclose(first["damping_ratio_range"], [0.05, 0.1])
    assert np.allclose(first["mode_frequency_hz_range"], np.sqrt(k[:, 0] / m[0]) / (2 * np.pi))
    assert first["units"]["stiffness"] == "N/m" and first["damping_ratio_declared"] == [0.4, 1.2]
    f_min = float(np.sqrt(4000.0 / 70.0) / (2 * np.pi))
    assert priors["burn_in_s"] == burn_in_seconds(0.05, f_min)
    robots[0].pop("plant_link_inertia_median")
    with pytest.raises(ValueError, match="plant_link_inertia_median"):
        transmission_priors(manifest, asset)


# ---------------------------------------------------------------------------
# Builds (slow): payload.mode none has round 6's robots and trajectories
# ---------------------------------------------------------------------------

def _small(config, mode: str):
    from diagnose_controller_modes import _shrink

    config = _shrink(config, trajectories=1, robots=3, backends=("mujoco",))
    excitation = replace(config.excitation, base_frequency=0.5, probe_harmonics=(8, 16, 32))
    return replace(config, excitation=excitation, candidates=4, payload=replace(config.payload, mode=mode))


@pytest.fixture(scope="module")
def fmrr_builds(tmp_path_factory):
    from diagnose_controller_modes import _load_asset
    from elastic_sim.dataset import generate, write_dataset

    base = load_config(_REPO / R6)
    asset = _load_asset(base.asset)
    cache: dict = {}
    out = {}
    for mode in ("random", "none"):
        config = _small(base, mode)
        frame, manifest, comparison = generate(config, asset, verbose=False, jobs=2, trajectory_cache=cache)
        path = tmp_path_factory.mktemp(mode) / f"fmrr_{mode}.parquet"
        write_dataset(frame, manifest, path, comparison, metadata_columns=config.metadata_columns)
        out[mode] = (path, frame, manifest)
    return asset, out


@pytest.mark.slow
def test_payload_none_build_has_round6_robots_and_trajectories(fmrr_builds):
    from round7_checks import compare

    _, builds = fmrr_builds
    (_, frame6, manifest6), (_, frame7, manifest7) = builds["random"], builds["none"]
    report = compare(manifest6, manifest7)
    assert report["ok"], report
    elastic = frame7["tier"] != "rigid"
    assert (frame7.loc[elastic, "payload_mass"] == 0.0).all()
    assert (frame6.loc[elastic, "payload_mass"] > 0.0).all()
    for robot in manifest7["robots"]:
        assert robot["payload"] is None and robot["sizing_payload"]["mass"] > 0.0
    # Same robots, a different plant: the target changes, the motor-side reference does not.
    assert np.array_equal(frame6["q_ref_clean0"].to_numpy(), frame7["q_ref_clean0"].to_numpy())
    assert not np.array_equal(frame6["ft_clean2"].to_numpy(), frame7["ft_clean2"].to_numpy())


@pytest.mark.slow
def test_contract_blocks_and_cross_test_fixture(fmrr_builds):
    import pandas as pd

    from elastic_sim.round7 import augment_contract, bag_baselines

    asset, builds = fmrr_builds
    path, frame, manifest = builds["none"]
    # The 2 s test bags are shorter than the FMRR prior's burn-in (8.25 s): keep its rule, shorten the head.
    priors = {**transmission_priors(manifest, asset), "burn_in_s": 0.5}
    augment_contract(path, frame, manifest, asset, priors=priors)
    contract = json.loads(path.with_suffix(".contract.json").read_text(encoding="utf-8"))
    # S-2: audit columns present and declared, none of them an input.
    assert contract["audit_columns"]["columns"] == ["q_link0..q_link2", "dq_link0..dq_link2", "defl0..defl2"]
    for family in AUDIT_FAMILIES:
        assert all(f"{family}{i}" in frame.columns for i in range(3))
        assert not any(column.startswith(family) for column in contract["input_columns"])
    # S-3
    assert contract["transmission_priors"]["burn_in_s"] == priors["burn_in_s"] > 0
    # S-4: per split, kept rows only, the oracle at least as good as the rigid model.
    block = contract["baselines"]["two_mass"]
    assert block["exclude_head_s"] == priors["burn_in_s"]
    test = block["splits"]["test"]
    assert 0 < test["n_rows"] < test["n_rows_unmasked"]
    assert test["rigid_elastic_oracle"]["rms"] < test["rigid"]["rms"]
    for suffix in ("_full", "_be", "_full_be"):                    # every coupling x integrator beside the default
        assert test[f"rigid_elastic_oracle{suffix}"]["rms"] > 0 and f"rigid_elastic_nominal{suffix}" in test
    # The fixture reproduces itself and the contract's numbers for its bag.
    fixture = np.load(path.with_name(path.stem + ".two_mass_reference.npz"))
    again = filter_sequence(fixture["tau_r"], fixture["inertia"], fixture["stiffness"], fixture["damping"],
                            float(fixture["time_step"]))
    assert np.array_equal(again["tau_s"], fixture["tau_s_diagonal_trapezoidal"])
    table, _ = bag_baselines(frame, manifest, contract, asset, priors, keep=[str(fixture["bag"])])
    kept = fixture["kept"]
    sse = ((fixture["tau_s_diagonal_trapezoidal"][kept] - fixture["target"][kept]) ** 2).sum(axis=0)
    assert np.allclose(table["rigid_elastic_oracle_sse"].iloc[0], sse, rtol=1e-12)
    assert isinstance(pd.DataFrame(table), pd.DataFrame)
