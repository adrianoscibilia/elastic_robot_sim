"""RR_23 C3-C5: the loader's key checks, the P2 chirp law and torque rule, the
TE sweep fit/skip rule, the contract fields, and the C7 hand-off entry."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from erd_recording.config import ConfigError, load_lab_config

LAB = Path(__file__).resolve().parents[1] / "config" / "lab"


def _reviewed(tmp_path: Path, name: str, edit=None) -> Path:
    """A draft lab file with both `reviewed` flags set (the owner's step), optionally edited."""
    raw = yaml.safe_load((LAB / name).read_text(encoding="utf-8"))
    raw["limits"]["reviewed"] = True
    raw["scene"]["reviewed"] = True
    if edit is not None:
        edit(raw)
    path = tmp_path / name
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def iiwa(tmp_path_factory):
    return load_lab_config(_reviewed(tmp_path_factory.mktemp("iiwa"), "iiwa_probe.yaml"))


@pytest.fixture(scope="module")
def ur10(tmp_path_factory):
    return load_lab_config(_reviewed(tmp_path_factory.mktemp("ur10"), "ur10_probe.yaml"))


# ---------------------------------------------------------------- C2, C3 ----


def test_the_three_probe_files_load_and_read_their_blocks(iiwa, ur10, tmp_path):
    assert iiwa.modal.pose_names == ("home", "forearm_up", "upright")
    assert set(iiwa.modal.joints) == {"iiwa_A1", "iiwa_A2", "iiwa_A4"}
    assert iiwa.te_sweeps.speeds == (0.1, 0.2, 0.4, 0.8)
    assert ur10.modal.extra_poses[0].index == 1 and ur10.modal.extra_poses[0].max_rung == 0.5
    assert ur10.modal.pose_name(1) == "lean_fwd"
    j150 = load_lab_config(_reviewed(tmp_path, "iiwa_probe_j150.yaml"))
    assert j150.modal is None and j150.te_sweeps is None
    assert set(j150.limits.jerk.values()) == {150.0}
    assert j150.excitation.probe.acceleration_fraction >= 0.40
    assert j150.excitation.seed == 20261102


def test_j150_differs_from_the_j100_file_only_where_c2_says():
    j100 = yaml.safe_load((LAB / "iiwa_probe.yaml").read_text())
    j150 = yaml.safe_load((LAB / "iiwa_probe_j150.yaml").read_text())
    for key in ("modal", "te_sweeps"):
        j100.pop(key)
    for key in ("jerk",):
        assert j150["limits"].pop(key) != j100["limits"].pop(key)
    assert j150["excitation"].pop("seed") != j100["excitation"].pop("seed")
    assert j150["excitation"]["probe"].pop("acceleration_fraction") != \
        j100["excitation"]["probe"].pop("acceleration_fraction")
    assert j150 == j100
    for name in ("iiwa_probe.yaml", "iiwa_probe_j150.yaml"):
        text = (LAB / name).read_text()
        assert ",ho" not in text
        assert len(re.findall(r"\n  reviewed: false", text)) == 2


@pytest.mark.parametrize("edit,match", [
    (lambda raw: raw.update(modl=raw.pop("modal")), "unknown top-level key"),
    (lambda raw: raw.update(te_sweep=raw.pop("te_sweeps")), "unknown top-level key"),
    (lambda raw: raw["modal"].update(a_maxx=1.0), r"modal: unknown key"),
    (lambda raw: raw["te_sweeps"].update(plateau=1.0), r"te_sweeps: unknown key"),
    (lambda raw: raw["modal"].update(poses=[0, 7]), "pose index 7"),
    (lambda raw: raw["modal"]["joints"].update(iiwa_A9={"f0": 1, "f1": 2, "T": 3}), "not a joint"),
    (lambda raw: raw["modal"].update(law="sine"), "law must be"),
    (lambda raw: raw["modal"].update(pose_names=["home"]), "pose_names"),
])
def test_unknown_or_bad_keys_are_refused(tmp_path, edit, match):
    with pytest.raises(ConfigError, match=match):
        load_lab_config(_reviewed(tmp_path, "iiwa_probe.yaml", edit))


# ---------------------------------------------------------------- C4 law ----


def test_the_clip_regions_follow_the_amplitude_law(iiwa, ur10):
    from erd_recording.modal import bind_ranges, chirp_trajectory

    names = iiwa.joint_order
    ranges = bind_ranges(chirp_trajectory(iiwa, names, "iiwa_A1", 0, 2.0, 1e-3).analytic)
    corner = 0.95 * 100.0 / 2.0 / (2 * math.pi)       # a/w^2 = j/w^3 at w = j/a
    assert ranges["acceleration"][0][0] == pytest.approx(2.0)
    assert ranges["acceleration"][0][1] == pytest.approx(corner, rel=2e-3)
    assert ranges["jerk"][0][1] == pytest.approx(40.0)
    assert ranges["position"] == [] and ranges["velocity"] == []
    # UR10, 1.5 Hz: a/(2 pi f)^2 = 0.0225 > 0.02, so position binds up to sqrt(a/0.02)/(2 pi)
    ranges = bind_ranges(chirp_trajectory(ur10, ur10.joint_order, "shoulder_lift_joint", 0, 2.0, 8e-3).analytic)
    assert ranges["position"][0][0] == pytest.approx(1.5)
    assert ranges["position"][0][1] == pytest.approx(math.sqrt(2.0 / 0.02) / (2 * math.pi), rel=2e-3)
    # at the 0.25 rung nothing but acceleration binds below the jerk corner (now 30 Hz)
    ranges = bind_ranges(chirp_trajectory(iiwa, names, "iiwa_A2", 0, 0.5, 1e-3).analytic)
    assert ranges["acceleration"] == [[2.0, 30.0]]


def test_the_chirp_is_r9s_chirp_evaluator(iiwa):
    from elastic_sim.round8 import ChirpEvaluator, ModalSpec

    from erd_recording.modal import chirp_trajectory

    trajectory = chirp_trajectory(iiwa, iiwa.joint_order, "iiwa_A2", 1, 1.0, 1e-3)
    pose = iiwa.poses.standstill[1]
    spec = ModalSpec(joints=(("iiwa_A2", 2.0, 30.0, 20.0),), poses=(("forearm_up", tuple(pose)),), a_max=1.0,
                     position_amplitude_max=0.02, velocity_max=0.95 * 1.0, jerk_max=0.95 * 100.0, repeats=1,
                     hold_s=1.0, fade_s=1.0)
    evaluator = ChirpEvaluator(pose, 1, 2.0, 30.0, 20.0, spec)
    expected = evaluator.offset(trajectory.time)
    assert trajectory.duration == pytest.approx(22.0)
    np.testing.assert_allclose(trajectory.position[:, 1] - pose[1], expected, rtol=0, atol=1e-15)
    others = np.delete(trajectory.position, 1, axis=1)
    np.testing.assert_array_equal(others, np.tile(np.delete(np.asarray(pose), 1), (len(trajectory.time), 1)))
    # held before and after, faded to zero
    hold = int(round(1.0 / 1e-3))
    assert np.all(expected[:hold] == 0) and np.all(expected[-hold:] == 0)


def test_the_grid_jerk_keeps_the_cap_where_the_c1_fade_spikes_an_oversampled_difference(iiwa):
    from erd_recording.modal import chirp_peaks, chirp_trajectory

    evaluator = chirp_trajectory(iiwa, iiwa.joint_order, "iiwa_A1", 2, 2.0, 1e-3).analytic
    peaks = chirp_peaks(evaluator, 1e-3)
    assert peaks["jerk"] <= 100.0
    assert peaks["acceleration"] <= 2.0 + 1e-3
    assert peaks["position"] <= 0.02


# ------------------------------------------------------- C4 torque rule ----


def test_q_is_ten_until_a_previous_rung_was_measured():
    from erd_recording.modal import torque_rule

    dynamics = {"g_j": -5.0, "m_jj": 11.23}
    first = torque_rule(dynamics, 2.0, 330.0, 0.6, None)
    assert first["Q_used"] == 10.0 and first["refused"]           # 5 + 224.6 = 0.70 of 330
    assert first["predicted_peak_nm"] == pytest.approx(5.0 + 11.23 * 2.0 * 10.0)
    measured = torque_rule(dynamics, 2.0, 330.0, 0.6, {"Q": 4.0, "rung": 0.5, "run": "r050"})
    assert measured["Q_used"] == 4.0 and not measured["refused"]
    assert "r050" in measured["Q_source"]
    floor = torque_rule(dynamics, 2.0, 330.0, 0.6, {"Q": 0.3, "rung": 0.5, "run": "r050"})
    assert floor["Q_used"] == 1.0


def test_measured_q_reads_the_previous_runs_quicklook(tmp_path):
    from erd_recording.modal import ModalError, measured_q

    def run(name, chirps):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "quicklook.json").write_text(json.dumps({"chirps": chirps}))
        return folder

    low = run("r025", {"a": {"joint": "j", "pose_name": "home", "rung": 0.25, "a": 0.5, "Q_measured": 3.0,
                             "peak_dynamic_nm": 10.0}})
    high = run("r050", {"a": {"joint": "j", "pose_name": "home", "rung": 0.5, "a": 1.0, "Q_measured": 2.0,
                              "peak_dynamic_nm": 14.0},
                        "b": {"joint": "j", "pose_name": "home", "rung": 0.5, "a": 1.0, "Q_measured": 2.5,
                              "peak_dynamic_nm": 18.0}})
    result = measured_q([low, high])
    assert result[("j", "home")]["Q"] == 2.5          # the higher rung, its worst repeat
    assert result[("j", "home")]["run"] == "r050"
    (tmp_path / "empty").mkdir()
    with pytest.raises(ModalError, match="quicklook"):
        measured_q([tmp_path / "empty"])


class _World:
    def validate_path(self, q, margin=0.0, max_joint_step=0.01):
        return type("R", (), {"valid": True, "closest_pair": None})()


def _build(config, options):
    from elastic_sim.assets import AssetRegistry

    from erd_recording.modal import build_modal_segments
    from erd_recording.planning import PlanSegment, quintic_segment

    asset = AssetRegistry.for_repository().load(config.description.sim_asset)
    dt = 1.0 / config.rate_hz

    def approach(start, end, sid, kind):
        trajectory = quintic_segment(start, end, velocity_limit=0.25, acceleration_limit=0.5, time_step=dt)
        return PlanSegment(sid, kind, trajectory, trajectory.digest())

    return build_modal_segments(config, _World(), asset, asset.joint_names, dt, 0.0, options,
                                approach=approach, segment_type=PlanSegment)


def test_ur10_rung_1_refuses_and_lists_what_q_10_cannot_carry(ur10, tmp_path):
    from erd_recording.modal import ModalOptions

    segments, report = _build(ur10, ModalOptions(rung=1.0))
    refused = {(r["joint"], r["pose_name"]) for r in report["refused"]}
    assert refused == {("shoulder_lift_joint", "lean_back"), ("shoulder_lift_joint", "fore_up"),
                       ("elbow_joint", "home")}
    chirps = [s for s in segments if s.kind == "modal_chirp"]
    assert len(chirps) == 3 * 3                       # 3 accepted chirps x 3 repeats
    assert not any("lean_fwd" in s.segment_id for s in segments)   # extra pose: rung <= 0.5 only
    assert segments[0].kind == "modal_approach" and segments[-1].segment_id == "modal_return"
    # a measured 0.5 rung lowers Q and lets the lift chirp in
    run = tmp_path / "ur_modal_lb_fu_r050"
    run.mkdir()
    (run / "quicklook.json").write_text(json.dumps({"chirps": {"x": {
        "joint": "shoulder_lift_joint", "pose_name": "fore_up", "rung": 0.5, "a": 1.0, "Q_measured": 3.0,
        "peak_dynamic_nm": 33.7}}}))
    _, report = _build(ur10, ModalOptions(rung=1.0, only=(("shoulder_lift_joint", "fore_up"),), previous=(str(run),)))
    (chirp,) = report["chirps"]
    assert not chirp["refused"] and chirp["Q_used"] == 3.0


def test_ladder_rungs_are_commissioning_except_an_extra_pose_at_its_max_rung(ur10):
    from erd_recording.modal import ModalOptions, chirp_targets

    targets = chirp_targets(ur10, ModalOptions(rung=0.5))
    by_pose = {(t["joint"], t["pose_name"]): t for t in targets}
    assert by_pose[("shoulder_lift_joint", "home")]["kind"] == "modal_chirp_commissioning"
    assert by_pose[("shoulder_lift_joint", "home")]["repeats"] == 1
    assert by_pose[("shoulder_lift_joint", "lean_fwd")]["kind"] == "modal_chirp"
    assert by_pose[("shoulder_lift_joint", "lean_fwd")]["repeats"] == 3
    assert ("shoulder_lift_joint", "lean_fwd") in {(t["joint"], t["pose_name"])
                                                   for t in chirp_targets(ur10, ModalOptions(rung=0.25))}
    assert all(t["kind"] == "modal_chirp" and t["repeats"] == 3 for t in chirp_targets(ur10, ModalOptions(rung=1.0)))
    only = chirp_targets(ur10, ModalOptions(rung=0.25, only=(("shoulder_lift_joint", "home"),), repeats=2))
    assert [(t["joint"], t["pose_name"], t["repeats"]) for t in only] == [("shoulder_lift_joint", "home", 2)]


def test_the_contract_fields_and_modal_columns(iiwa):
    from erd_recording.contract import add_segment_columns, segments_block
    from erd_recording.modal import ModalOptions

    segments, report = _build(iiwa, ModalOptions(rung=1.0, only=(("iiwa_A2", "home"),)))
    details = report["segments"]
    sid = "modal_iiwa_A2_home_1"
    assert set(details[sid]) >= {"joint", "pose_name", "pose_q", "f0", "f1", "T", "a_max", "repeat", "rung",
                                 "M_jj", "dg_dq_jj", "Q_used"}
    assert details[sid]["M_jj"] == pytest.approx(2.99, abs=0.01)
    assert details[sid]["dg_dq_jj"] == pytest.approx(-25.3, abs=0.2)     # RR_20 S1 table
    block = segments_block([sid], kinds={sid: "modal_chirp"}, details=details)
    assert block[0]["kind"] == "modal_chirp" and block[0]["split"] == "modal_reference"
    assert block[0]["repeat"] == 1 and block[0]["a_max"] == 2.0
    frame = pd.DataFrame({"bag": [sid, sid, "modal_iiwa_A2_home_0"], "t": [0.0, 1e-3, 0.0]})
    out = add_segment_columns(frame, details)
    assert list(out["modal_joint"]) == ["iiwa_A2"] * 3
    assert list(out["modal_repeat"]) == [1, 1, 0]
    assert list(out["modal_f1"]) == [30.0] * 3
    excitation = pd.DataFrame({"bag": ["excitation_0"], "t": [0.0]})
    assert list(add_segment_columns(excitation, details).columns) == ["bag", "t"]


# ------------------------------------------------------------ C5 sweeps ----


def test_the_sweep_plateau_fits_or_is_skipped(iiwa):
    from erd_recording.modal import sweep_plateau

    ramp, plateau = sweep_plateau(1.0, 0.4, acceleration=4.0, jerk=100.0, plateau_s=3.0, dt=1e-3)
    assert plateau == pytest.approx(1.0 / 0.4 - ramp, abs=1e-3)       # A2 at 0.4: the range binds
    assert 2.0 < plateau < 3.0
    assert sweep_plateau(1.6, 0.1, acceleration=4.0, jerk=100.0, plateau_s=3.0, dt=1e-3)[1] == 3.0
    # 0.8 rad/s: A1 (1.6 rad) gets < 2 s
    assert sweep_plateau(1.6, 0.8, acceleration=4.0, jerk=100.0, plateau_s=3.0, dt=1e-3)[1] < 2.0


def test_te_sweeps_go_both_ways_inside_the_caps(iiwa):
    from erd_recording.modal import build_te_sweep_segments
    from erd_recording.planning import PlanSegment, quintic_segment

    def approach(start, end, sid, kind):
        trajectory = quintic_segment(start, end, velocity_limit=0.25, acceleration_limit=0.5, time_step=1e-3)
        return PlanSegment(sid, kind, trajectory, trajectory.digest())

    segments, report = build_te_sweep_segments(iiwa, _World(), iiwa.joint_order, 1e-3, 0.0,
                                               approach=approach, segment_type=PlanSegment)
    skipped = {(s["joint"], s["speed"]) for s in report["skipped"]}
    assert skipped == {("iiwa_A1", 0.8), ("iiwa_A2", 0.8), ("iiwa_A4", 0.8)}
    sweeps = [s for s in segments if s.kind == "calibration_sweep"]
    assert len(sweeps) == 2 * 9
    arrays = iiwa.limits.as_arrays(iiwa.joint_order)
    for segment in sweeps:
        info = report["segments"][segment.segment_id]
        j = iiwa.joint_order.index(info["joint"])
        velocity = segment.trajectory.velocity[:, j]
        assert np.sign(velocity.sum()) == (1 if info["direction"] == "positive" else -1)
        assert np.max(np.abs(segment.trajectory.acceleration)) <= arrays["acceleration"][j] + 1e-9
        lo, hi = iiwa.te_sweeps.joints[info["joint"]]
        assert segment.trajectory.position[:, j].min() >= lo - 1e-9
        assert segment.trajectory.position[:, j].max() <= hi + 1e-9
        plateau = (segment.trajectory.time > info["plateau_start_s"]) & (segment.trajectory.time <= info["plateau_end_s"])
        assert np.allclose(np.abs(velocity[plateau]), abs(info["speed"]))
        assert info["split"] == "calibration"


# --------------------------------------------------------------- C6a, C7 ----


def test_transmission_reads_the_probe_bins(iiwa, tmp_path):
    """A measured q that is the plan delayed by 3 samples and halved on the
    top lines gives T = 1 below and 0.5 above."""
    from erd_recording.quicklook import transmission

    rate, duration = 1000.0, 10.0
    t = np.arange(int(duration * rate)) / rate
    lines = np.array([2.0, 6.0, 20.0])
    plan = sum(0.001 * np.sin(2 * np.pi * f * t + f) for f in lines)[:, None] * np.ones((1, 7))
    measured = sum((1.0 if f < 10 else 0.5) * 0.001 * np.sin(2 * np.pi * f * (t - 3e-3) + f) for f in lines)

    class _Segment:
        kind = "excitation"
        segment_id = "excitation_0"
        trajectory = type("T", (), {"position": plan, "metadata": {"n_periods": 2, "base_frequency": 0.2}})()

    class _Recording:
        raw = pd.DataFrame({f"q{i}": measured for i in range(7)})

        def refined_start(self, *args):
            return 0

    plan_dir = tmp_path / "plan"
    plan_dir.mkdir()
    (plan_dir / "plan.manifest.json").write_text(json.dumps({"probe": {"lines_hz": lines.tolist()}}))
    result = transmission(iiwa, _Recording(), plan_dir, {"excitation_0": _Segment()})
    np.testing.assert_allclose(result["median"]["iiwa_A2"], [1.0, 1.0, 0.5], atol=1e-6)
    assert result["below_criterion"]["iiwa_A2"] == []


def test_handoff_entry_and_append(iiwa, tmp_path, monkeypatch):
    from erd_recording.handoff import HandoffError, append, entry_for

    monkeypatch.setenv("ERD_REPO_ROOT", str(tmp_path))
    run = tmp_path / "data" / "iiwa_probe_all_01"
    (run / "dataset").mkdir(parents=True)
    dataset = run / "dataset" / "kuka_lbr_iiwa_14_r820.parquet"
    dataset.write_bytes(b"parquet bytes")
    (run / "dataset" / "kuka_lbr_iiwa_14_r820.contract.json").write_text(json.dumps(
        {"segments": [{"segment_id": "excitation_0", "kind": "excitation", "split": "test"}]}))
    (run / "validation.json").write_text(json.dumps({"status": "valid"}))
    entry = entry_for(iiwa, run, "test")
    assert entry["path"] == "data/iiwa_probe_all_01/dataset/kuka_lbr_iiwa_14_r820.parquet"
    assert entry["robot"] == "iiwa" and entry["id"] == "iiwa_probe_all_01" and len(entry["sha256"]) == 64
    with pytest.raises(HandoffError, match="not \\['modal_reference'\\]"):
        entry_for(iiwa, run, "modal_reference")
    real_files = tmp_path / "real_files.yaml"
    real_files.write_text("# header\n# files:\n#   - robot: iiwa\nfiles: []\n")
    append(entry, real_files)
    loaded = yaml.safe_load(real_files.read_text())
    assert loaded["files"] == [entry]
    with pytest.raises(HandoffError, match="already listed"):
        append(entry, real_files)
    (run / "validation.json").write_text(json.dumps({"status": "invalid"}))
    with pytest.raises(HandoffError, match="invalid"):
        entry_for(iiwa, run, "test")
    standstill = tmp_path / "data" / "iiwa_standstill_02" / "bag_standstill"
    standstill.mkdir(parents=True)
    (standstill / "metadata.yaml").write_text("x")
    entry = entry_for(iiwa, standstill.parent, "calibration")
    assert entry["format"] == "standstill_bag"
