"""RR_20 T1-T3: the real probe comb (base remap, cap, loader) and its weighting key."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from erd_recording.config import ConfigError, check_probe_comb, load_lab_config
from erd_recording.probe import ProbeComb, probe_bound_hz, remap_probe_harmonics

LAB = Path(__file__).resolve().parents[1] / "config" / "lab"
REPO = Path(os.environ["ERD_REPO_ROOT"])

#: `resolve_probe_design` of the r7 configs (32 log lines at 0.1 Hz), frozen
#: here so the remap tests need neither pinocchio nor the configs.
IIWA_R7 = (20, 23, 26, 29, 33, 37, 42, 47, 53, 60, 68, 77, 87, 99, 112, 126, 143, 161, 182, 206, 233, 264,
           298, 337, 381, 431, 487, 551, 623, 704, 796, 900)
UR10_R7 = (15, 16, 18, 20, 23, 25, 28, 32, 35, 40, 44, 49, 55, 62, 69, 77, 86, 96, 107, 119, 133, 149, 166,
           186, 208, 232, 259, 289, 323, 361, 403, 450)


def _hz(harmonics, base):
    return [round(h * base, 6) for h in harmonics]


def test_iiwa_r7_on_the_b_base_is_the_rr20_comb():
    lab = remap_probe_harmonics(IIWA_R7, 0.1, 0.2, 5, probe_bound_hz(25.0, 1000.0))
    # RR_20 S2 T2, "expected resolved combs": 21 lines 2.0 ... 23.2 Hz
    assert _hz(lab, 0.2) == [2.0, 2.4, 2.6, 3.0, 3.2, 3.6, 4.2, 4.8, 5.4, 6.0, 6.8, 7.6, 8.8, 10.0, 11.2, 12.6,
                             14.4, 16.0, 18.2, 20.6, 23.2]
    assert all(h * 0.2 <= 25.0 for h in lab)
    assert all(isinstance(h, int) and h > 5 for h in lab)


def test_ur10_r7_on_the_b_base_stays_below_the_mounting_mode():
    lab = remap_probe_harmonics(UR10_R7, 0.1, 0.2, 5, probe_bound_hz(10.0, 125.0))
    assert len(lab) == 16
    assert lab[0] * 0.2 == pytest.approx(1.6)  # 1.5 Hz is not on the 0.2 Hz grid
    assert lab[-1] * 0.2 == pytest.approx(9.6)
    assert lab[-1] * 0.2 <= 10.0 < 10.7


def test_the_a_base_path_is_the_old_capped_list():
    bound = probe_bound_hz(90.0, 1000.0)
    old = tuple(int(h) for h in IIWA_R7 if h * 0.1 <= bound)  # the pre-RR_20 planner rule
    assert remap_probe_harmonics(IIWA_R7, 0.1, 0.1, 5, bound) == old
    bound = probe_bound_hz(11.0, 125.0)
    assert remap_probe_harmonics(UR10_R7, 0.1, 0.1, 5, bound) == tuple(h for h in UR10_R7 if h * 0.1 <= bound)


def test_the_bound_is_the_sg_bound_or_the_cap():
    assert probe_bound_hz(None, 125.0) == pytest.approx(11.25)
    assert probe_bound_hz(10.0, 125.0) == 10.0
    assert probe_bound_hz(200.0, 1000.0) == pytest.approx(90.0)


def test_lines_on_the_main_harmonics_are_dropped():
    # 0.5 and 1.0 Hz on a 0.2 Hz base are harmonics 2.5 -> 2 and 5: inside n_harmonics 5
    assert remap_probe_harmonics((5, 10, 20), 0.1, 0.2, 5, 50.0) == (10,)


def test_check_probe_comb_refuses_an_empty_or_too_high_comb():
    empty = ProbeComb(harmonics=(), base_frequency=0.2, bound_hz=2.0, training_base_frequency=0.1,
                      training_harmonics=(30,))
    with pytest.raises(ValueError, match="no probe line"):
        check_probe_comb(empty, "x", ValueError)
    high = ProbeComb(harmonics=(60,), base_frequency=0.2, bound_hz=10.0, training_base_frequency=0.1,
                     training_harmonics=(120,))
    with pytest.raises(ValueError, match="exceeds the bound"):
        check_probe_comb(high, "x", ValueError)


# ---------------------------------------------------------------------------
# Loader (T1, T3)
# ---------------------------------------------------------------------------


def _reviewed(name: str) -> dict:
    raw = yaml.safe_load((LAB / f"{name}.yaml").read_text(encoding="utf-8"))
    raw["limits"]["reviewed"] = True
    raw["scene"]["reviewed"] = True
    return raw


def _write(tmp_path: Path, raw: dict) -> Path:
    path = tmp_path / "lab.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


@pytest.mark.parametrize("name,cap,fraction", [("iiwa_probe", 25.0, 0.25), ("ur10_probe", 10.0, 0.5)])
def test_the_capped_probe_drafts_load(tmp_path, name, cap, fraction):
    as_checked_in = yaml.safe_load((LAB / f"{name}.yaml").read_text(encoding="utf-8"))
    assert as_checked_in["limits"]["reviewed"] is False  # the owner re-reviews
    config = load_lab_config(_write(tmp_path, _reviewed(name)))
    probe = config.excitation.probe
    assert probe.enabled and probe.max_top_hz == cap
    assert probe.weighting == "acceleration"
    assert probe.acceleration_fraction == fraction
    assert config.excitation.base_frequency == 0.2


def test_unknown_probe_key_is_refused(tmp_path):
    raw = _reviewed("iiwa_probe")
    raw["excitation"]["probe"]["weigthing"] = "acceleration"
    with pytest.raises(ConfigError, match=r"unknown key\(s\) \['weigthing'\]"):
        load_lab_config(_write(tmp_path, raw))


def test_bad_weighting_is_refused(tmp_path):
    raw = _reviewed("iiwa_probe")
    raw["excitation"]["probe"]["weighting"] = "velocity"
    with pytest.raises(ConfigError, match="weighting must be one of"):
        load_lab_config(_write(tmp_path, raw))


def test_weighting_defaults_to_position(tmp_path):
    raw = _reviewed("iiwa_probe")
    del raw["excitation"]["probe"]["weighting"]
    assert load_lab_config(_write(tmp_path, raw)).excitation.probe.weighting == "position"


def test_an_explicit_comb_above_the_cap_is_refused_by_the_loader(tmp_path):
    # ur10_table.yaml: an explicit (no probe_design) 3 -> 150 Hz comb; a 2 Hz cap leaves nothing
    raw = _reviewed("ur10_probe")
    raw["excitation"]["probe"]["from_training_config"] = str(REPO / "config/identification/ur10_table.yaml")
    raw["excitation"]["probe"]["max_top_hz"] = 2.0
    with pytest.raises(ConfigError, match="no probe line"):
        load_lab_config(_write(tmp_path, raw))
    raw["excitation"]["probe"]["max_top_hz"] = 10.0
    assert load_lab_config(_write(tmp_path, raw)).excitation.probe.enabled


def test_a_missing_training_config_is_refused(tmp_path):
    raw = _reviewed("iiwa_probe")
    raw["excitation"]["probe"]["from_training_config"] = str(tmp_path / "nope.yaml")
    with pytest.raises(ConfigError, match="does not exist"):
        load_lab_config(_write(tmp_path, raw))


# ---------------------------------------------------------------------------
# Planner resolution on the real r7 configs (needs pinocchio, like test_planning)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,top,lines", [("iiwa_probe", 23.2, 21), ("ur10_probe", 9.6, 16)])
def test_the_planner_resolves_the_r7_comb_on_the_lab_grid(tmp_path, name, top, lines):
    pytest.importorskip("pinocchio")
    from erd_recording.planning import resolve_probe

    config = load_lab_config(_write(tmp_path, _reviewed(name)))
    comb = resolve_probe(config, config.rate_hz)
    assert len(comb.harmonics) == lines
    assert comb.top_hz == pytest.approx(top)
    assert comb.base_frequency == 0.2
    assert all(abs(f / 0.2 - round(f / 0.2)) < 1e-9 for f in comb.lines_hz)
