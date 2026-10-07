"""RR_12 C-1 (production references), C-2 (complete validation), C-3
(portable provenance) and C-4 (checkpoint resolution)."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from erd_recording.contract import (
    differentiation_from_reference,
    load_reference,
    portable_path,
    resolve_reference,
)
from erd_recording.evaluate_cli import resolve_checkpoint
from erd_recording.validate import (
    check_identification_freshness_ur,
    check_noise,
    check_reference_tracking,
    check_session_health_iiwa,
    check_session_health_ur,
    finalize_checks,
    fit_start_time,
    not_applicable,
    quintic_hermite,
)

IIWA_JOINTS = tuple(f"iiwa_A{i}" for i in range(1, 8))
UR_JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint",
             "wrist_3_joint")
REPO = Path(__file__).resolve().parents[4]
PRODUCTION = {
    "iiwa": (REPO / "data/identification/r6-full-20260929-0049/iiwa_drive/"
                    "kuka_lbr_iiwa_14_r820_table_round6_drive.contract.json",
             "c64f7aef2e1e0f4625c1fea7d72ef052f8e045a0e09b257c2ebf04225bb5ea25"),
    "ur10": (REPO / "data/identification/r6-ur10-20261005-1126/ur10/ur10_table_round6.contract.json",
             "cc8867e909b0e9804df29b4645cabb954991fe99c7c0d2e6d0639b6fbf214a2c"),
}


def _production_like(tmp_path: Path, *, rate: float, joints, asset: str, synthetic: bool = False):
    """The production layout: no joint_names/rate in the contract, a sibling manifest."""
    contract = {"schema": "elastic_sim.identification/3", "n_dof": len(joints),
                "differentiation": {"sample_rate_hz": rate, "sg_window": 5, "sg_poly": 3,
                                    "sg_cutoff_hz": 0.45 * rate / 5, "probe_top_hz": 0.09 * rate}}
    if synthetic:
        contract["synthetic"] = True
    path = tmp_path / "prod.contract.json"
    path.write_text(json.dumps(contract))
    (tmp_path / "prod.manifest.json").write_text(json.dumps(
        {"asset": asset, "joint_names": list(joints), "sample_time_step": 1.0 / rate}))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_joint_names_and_rate_come_from_the_sibling_manifest(tmp_path):
    path, sha = _production_like(tmp_path, rate=1000.0, joints=IIWA_JOINTS, asset="kuka_lbr_iiwa_14_r820_table")
    reference = load_reference(path, robot="kuka_lbr_iiwa_14_r820", joint_order=IIWA_JOINTS, expected_sha256=sha)
    assert reference["_joint_names"] == list(IIWA_JOINTS)
    assert reference["_rate_hz"] == 1000.0
    assert reference["_asset"] == "kuka_lbr_iiwa_14_r820_table"


def test_sha256_pin_mismatch_refuses(tmp_path):
    path, _ = _production_like(tmp_path, rate=1000.0, joints=IIWA_JOINTS, asset="x")
    with pytest.raises(ValueError, match="sha256"):
        load_reference(path, robot="kuka_lbr_iiwa_14_r820", joint_order=IIWA_JOINTS, expected_sha256="0" * 64)


def test_real_hardware_refuses_a_synthetic_reference(tmp_path):
    path, sha = _production_like(tmp_path, rate=1000.0, joints=IIWA_JOINTS, asset="x", synthetic=True)
    with pytest.raises(ValueError, match="synthetic"):
        load_reference(path, robot="kuka_lbr_iiwa_14_r820", joint_order=IIWA_JOINTS, expected_sha256=sha,
                       hardware="real")


def test_a_production_reference_without_its_manifest_refuses(tmp_path):
    path, _ = _production_like(tmp_path, rate=1000.0, joints=IIWA_JOINTS, asset="x")
    (tmp_path / "prod.manifest.json").unlink()
    with pytest.raises(ValueError, match="manifest"):
        load_reference(path, robot="kuka_lbr_iiwa_14_r820", joint_order=IIWA_JOINTS)


def test_manifest_joint_order_mismatch_refuses(tmp_path):
    path, _ = _production_like(tmp_path, rate=1000.0, joints=tuple(reversed(IIWA_JOINTS)), asset="x")
    with pytest.raises(ValueError, match="joint order"):
        load_reference(path, robot="kuka_lbr_iiwa_14_r820", joint_order=IIWA_JOINTS)


def test_table_asset_must_carry_the_same_arm(tmp_path):
    path, _ = _production_like(tmp_path, rate=1000.0, joints=IIWA_JOINTS, asset="kuka_lbr_iiwa_14_r820_table")
    reference = load_reference(path, robot="kuka_lbr_iiwa_14_r820", joint_order=IIWA_JOINTS,
                               sim_asset="kuka_lbr_iiwa_14_r820")
    assert reference["_arm_check"]["ok"]
    with pytest.raises(ValueError, match="not the arm"):
        load_reference(path, robot="kuka_lbr_iiwa_14_r820", joint_order=IIWA_JOINTS, sim_asset="ur10")


def test_iiwa_same_rate_is_matched(tmp_path):
    path, _ = _production_like(tmp_path, rate=1000.0, joints=IIWA_JOINTS, asset="x")
    reference = load_reference(path, robot="kuka_lbr_iiwa_14_r820", joint_order=IIWA_JOINTS)
    block = differentiation_from_reference(reference, rate_real=1000.0, probe_top_real=0.0, reference_path=path)
    assert (block["sg_window"], block["sg_cutoff_hz"], block["cutoff_matched"]) == (5, 90.0, True)
    assert block["sample_rate_hz"] == 1000.0 and "rate" not in block


def test_ur10_floor_writes_the_achieved_cutoff_not_the_reference(tmp_path):
    """F-5: 45 Hz at 500 Hz -> W = odd(round(1.25)) = 1, floored to 5 at 125 Hz = 11.25 Hz."""
    path, _ = _production_like(tmp_path, rate=500.0, joints=UR_JOINTS, asset="ur10_table")
    reference = load_reference(path, robot="ur10_cb3", joint_order=UR_JOINTS)
    block = differentiation_from_reference(reference, rate_real=125.0, probe_top_real=0.0, reference_path=path)
    assert block["sg_window"] == 5 and block["window_floored"]
    assert block["sg_cutoff_hz"] == pytest.approx(11.25)
    assert block["reference_cutoff_hz"] == pytest.approx(45.0)
    assert block["cutoff_matched"] is False


@pytest.mark.parametrize("name,robot,joints,asset,window,cutoff,matched", [
    ("iiwa", "kuka_lbr_iiwa_14_r820", IIWA_JOINTS, "kuka_lbr_iiwa_14_r820", 5, 90.0, True),
    ("ur10", "ur10_cb3", UR_JOINTS, "ur10", 5, 11.25, False),
])
def test_the_real_production_references_load(name, robot, joints, asset, window, cutoff, matched):
    path, sha = PRODUCTION[name]
    if not path.is_file():
        pytest.skip(f"{path} not on this machine (T2.0-lab item 2 copies it)")
    reference = load_reference(path, robot=robot, joint_order=joints, expected_sha256=sha, hardware="real",
                               sim_asset=asset)
    rate = 1000.0 if name == "iiwa" else 125.0
    block = differentiation_from_reference(reference, rate_real=rate, probe_top_real=0.0, reference_path=path)
    assert (block["sg_window"], block["cutoff_matched"]) == (window, matched)
    assert block["sg_cutoff_hz"] == pytest.approx(cutoff)
    assert block["derived_from"].startswith("data/identification/")  # C-3: repo-relative


def test_cli_reference_replaces_the_configured_pin(tmp_path):
    config = tmp_path / "lab" / "x.yaml"
    config.parent.mkdir()
    path, sha = resolve_reference(config, "../ref/a.contract.json", "ab" * 32)
    assert path == str((tmp_path / "ref" / "a.contract.json").resolve()) and sha == "ab" * 32
    path, sha = resolve_reference(config, "../ref/a.contract.json", "ab" * 32, "/abs/b.contract.json", None)
    assert path == "/abs/b.contract.json" and sha is None


def test_portable_path_is_repo_relative(monkeypatch, tmp_path):
    monkeypatch.setenv("ERD_REPO_ROOT", str(tmp_path))
    inside = tmp_path / "data" / "x.json"
    assert portable_path(inside) == "data/x.json"
    assert portable_path("/elsewhere/y.json") == "/elsewhere/y.json"


# ---------------------------------------------------------------------------
# C-2: complete validation
# ---------------------------------------------------------------------------


def test_a_real_hardware_report_with_one_null_is_invalid():
    checks = [{"check": "uniform grid", "ok": True}, {"check": "envelopes", "ok": None, "status": "not_available"}]
    result = finalize_checks(checks, hardware="real", robot="kuka_lbr_iiwa_14_r820")
    assert result["ok"] is False and result["errors"] == ["envelopes"]
    assert finalize_checks(checks, hardware="emulator", robot="kuka_lbr_iiwa_14_r820")["ok"] is True


def test_not_applicable_only_from_the_fixed_list():
    iiwa = finalize_checks([not_applicable("identification freshness", "kuka_lbr_iiwa_14_r820")],
                           hardware="real", robot="kuka_lbr_iiwa_14_r820")
    assert iiwa["ok"] is True
    forged = {"check": "identification freshness", "ok": None, "status": "not_applicable", "reason": "skip"}
    assert finalize_checks([forged], hardware="real", robot="ur10_cb3")["ok"] is False
    with pytest.raises(KeyError):
        not_applicable("noise", "ur10_cb3")


def test_reference_tracking_recovers_the_jtc_start_time():
    t = np.arange(0, 3.0, 0.001)
    q = np.column_stack([0.3 * np.sin(2 * np.pi * 0.5 * t), 0.1 * np.cos(2 * np.pi * 1.5 * t)])
    v = np.column_stack([0.3 * np.pi * np.cos(np.pi * t), -0.1 * 3 * np.pi * np.sin(3 * np.pi * t)])
    a = np.column_stack([-0.3 * np.pi**2 * np.sin(np.pi * t), -0.1 * 9 * np.pi**2 * np.cos(3 * np.pi * t)])
    start = 0.0123456
    stamps = start + np.arange(0, 2.9, 0.001) + 0.000317  # update ticks not aligned with the plan grid
    values = quintic_hermite(t, q, v, a, stamps - start)
    fit = fit_start_time(stamps, values, t, q, v, a)
    assert fit["start_s"] == pytest.approx(start, abs=1e-8)
    assert fit["max_abs_error"] < 1e-9
    check = check_reference_tracking({"excitation_0": {"jtc": fit, "commanded": fit}})
    assert check["ok"]
    bad = check_reference_tracking({"excitation_0": {"jtc": {**fit, "max_abs_error": 1e-6}}})
    assert not bad["ok"]


def test_session_health_flags_a_non_commanding_sample():
    good = pd.DataFrame({"fri_session_state": [4] * 10, "fri_command_mode": [1] * 10})
    bad = good.copy()
    bad.loc[5, "fri_session_state"] = 3
    assert check_session_health_iiwa({"excitation_0": good})["ok"]
    assert not check_session_health_iiwa({"excitation_0": good, "excitation_1": bad})["ok"]
    ur = pd.DataFrame({"speed_scaling": [1.0, 1.0, 0.5], "target_speed_fraction": [1.0] * 3,
                       "robot_mode": [7] * 3, "safety_status": [1] * 3})
    assert not check_session_health_ur({"excitation_0": ur})["ok"]


def test_noise_table_has_the_section_5_1_fields():
    rng = np.random.default_rng(0)
    holds = [pd.DataFrame({f"{p}{j}": 0.01 * rng.normal(size=500) + 0.001 * np.arange(500)
                           for p in ("q", "dq", "tau", "ft") for j in range(2)}) for _ in range(3)]
    production = pd.DataFrame({f"{p}{j}": np.sin(np.arange(2000) / 50.0) for p in ("q", "dq", "tau", "ft")
                               for j in range(2)})
    check = check_noise(holds, production, n_dof=2, ft_offset=[0.5, -0.2])
    assert check["ok"]
    ft = check["channels"]["ft"]
    assert ft["sigma"][0] == pytest.approx(0.01, rel=0.1)  # the 0.001/sample drift is detrended away
    assert ft["sigma_over_production_rms"][0] == pytest.approx(0.01 / np.sqrt(0.5), rel=0.15)
    assert ft["offset_vs_gravity_nm"] == [0.5, -0.2]
    assert all(step > 0 for step in ft["quantisation_step"])


def test_ur_identification_freshness_uses_session_and_temperature():
    identification = {"config_digest": "abc", "friction_sweeps": [
        {"segment": f"identify_sweep_{j}_0_positive", "temperature_before_c": 30.0, "temperature_after_c": 32.0}
        for j in range(2)]}
    assert check_identification_freshness_ur(identification, run_config_digest="abc",
                                             run_temperature_c=np.array([33.0, 29.0]))["ok"]
    assert not check_identification_freshness_ur(identification, run_config_digest="abc",
                                                 run_temperature_c=np.array([37.5, 31.0]))["ok"]
    assert not check_identification_freshness_ur(identification, run_config_digest="other",
                                                 run_temperature_c=np.array([31.0, 31.0]))["ok"]


# ---------------------------------------------------------------------------
# C-4: checkpoints moved after training are found through RELOCATIONS.tsv
# ---------------------------------------------------------------------------


def test_relocated_checkpoint_is_resolved_and_hash_checked(tmp_path):
    (tmp_path / "models" / "2026-09_September").mkdir(parents=True)
    target = tmp_path / "models" / "2026-09_September" / "m_checkpoint.pt"
    target.write_bytes(b"weights")
    digest = hashlib.sha256(b"weights").hexdigest()
    table = pd.DataFrame({"date": ["2026-10-06"], "old_path": ["models/m_checkpoint.pt"],
                          "new_path": ["models/2026-09_September/m_checkpoint.pt"], "sha256": [digest]})
    resolved = resolve_checkpoint("models/m_checkpoint.pt", tmp_path, table)
    assert resolved["checkpoint"] == str(target) and resolved["relocated_from"] == "models/m_checkpoint.pt"
    table.loc[0, "sha256"] = "0" * 64
    with pytest.raises(ValueError, match="sha256"):
        resolve_checkpoint("models/m_checkpoint.pt", tmp_path, table)
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint("models/absent.pt", tmp_path, table)
