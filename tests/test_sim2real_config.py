"""S2R_01 T-0.1: the ``elastic_sim.sim2real/1`` schema."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import yaml

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.path.join(_REPO, "src"))

from elastic_sim.sim2real.config import REPO_ROOT, ConfigError, load_config

BASE = _REPO / "config" / "sim2real" / "kuka_lbr_iiwa_14_r820_base.yaml"


def _write(path: Path, mapping: dict) -> Path:
    path.write_text(yaml.safe_dump(mapping), encoding="utf-8")
    return path


def _stage(tmp_path: Path, **extra) -> Path:
    return _write(tmp_path / "stage.yaml", {"schema": "elastic_sim.sim2real/1", "extends": str(BASE), **extra})


def test_base_config_loads_with_defaults():
    config = load_config(BASE)
    assert config.asset == "kuka_lbr_iiwa_14_r820"
    assert set(config.runs) == {"A", "B"}
    assert config.runs["A"].path == REPO_ROOT / "data/real_robot/kuka_lbr_iiwa_14_r820/20261014/iiwa_real_all_01"
    assert config.replay.mode == "R0" and config.replay.r0.sg_window == 31 and config.replay.r0.decimate == 5
    assert config.optimizer.method == "least_squares" and config.loss.prior_weight is None
    assert config.jobs is None


def test_extends_merges_mappings_and_replaces_lists(tmp_path):
    config = load_config(_stage(tmp_path, replay={"mode": "R1", "r0": {"decimate": 2}},
                                segments=["hold"], optimizer={"method": "cma", "cma": {"sigma0": 0.1}}))
    assert config.replay.mode == "R1"
    assert config.replay.r0.decimate == 2 and config.replay.r0.sg_window == 31
    assert config.segments == ("hold",)
    assert config.optimizer.cma == {"sigma0": 0.1, "popsize": None}
    assert config.runs["B"].sha256.startswith("91743567")


def test_split_globs_assign_bags():
    split = load_config(BASE).split
    assert split.split_of("B", "excitation_3") == "train"
    assert split.split_of("B", "excitation_8") == "val"
    assert split.split_of("A", "excitation_8") == "test"
    assert split.split_of("B", "identify_hold_2") == "train"
    assert split.split_of("A", "identify_hold_2") == "zero"  # S2R_06 A-13: A's session zero
    assert split.split_of("A", "approach_0") is None
    assert split.runs_of("train", ("A", "B")) == ("B",) and split.runs_of("zero", ("A", "B")) == ("A",)


def test_selection_and_identifiability_keys(tmp_path):
    config = load_config(_stage(tmp_path, selection="end_point", identifiability={"d_post_min": 0.8}))
    assert config.selection == "end_point"
    assert config.identifiability.d_post_min == 0.8 and config.identifiability.mode == "label"
    assert not config.identifiability.freeze_then_refit
    assert load_config(BASE).selection == "val" and load_config(BASE).identifiability.d_post_min == 0.9
    assert load_config(BASE).identifiability.mode == "label" and load_config(BASE).identifiability.partner_min == 0.5
    # S2R_08: the S2R_06 switch maps onto mode, with a deprecation note.
    with pytest.warns(DeprecationWarning, match="freeze_then_refit"):
        legacy = load_config(_stage(tmp_path, identifiability={"freeze_then_refit": True}))
    assert legacy.identifiability.mode == "freeze_refit" and legacy.identifiability.freeze_then_refit
    with pytest.warns(DeprecationWarning):
        assert load_config(_stage(tmp_path, identifiability={"freeze_then_refit": False})).identifiability.mode == \
            "label"
    with pytest.raises(ConfigError, match="mode"):
        load_config(_stage(tmp_path, identifiability={"mode": "freeze"}))
    with pytest.raises(ConfigError, match="selection"):
        load_config(_stage(tmp_path, selection="test"))
    with pytest.raises(ConfigError, match="d_post_min"):
        load_config(_stage(tmp_path, identifiability={"d_post_min": 1.5}))


def test_bag_in_two_splits_is_refused(tmp_path):
    config = load_config(_stage(tmp_path, split={"train": ["B:excitation_*"], "val": ["B:excitation_9"]}))
    with pytest.raises(ConfigError, match="more than one split"):
        config.split.split_of("B", "excitation_9")


@pytest.mark.parametrize("extra, key", [
    ({"replays": {}}, "replays"),
    ({"replay": {"r0": {"sg_windw": 5}}}, "sg_windw"),
    ({"replay": {"r1": {"windows": {"length_s": 2, "stride_s": 1, "burn": 1}}}}, "burn"),
    ({"optimizer": {"cma": {"sigma": 0.1}}}, "sigma"),
    ({"loss": {"channels": {"torque": 1.0}}}, "torque"),
    ({"runs": {"C": {"path": "x", "sha": "y"}}}, "sha"),
    ({"compute": {"workers": 3}}, "workers"),
])
def test_unknown_keys_are_named(tmp_path, extra, key):
    with pytest.raises(ConfigError, match=rf"unknown .* keys in .*: {key}"):
        load_config(_stage(tmp_path, **extra))


@pytest.mark.parametrize("extra, message", [
    ({"replay": {"mode": "R3"}}, "replay.mode"),
    ({"optimizer": {"method": "adam"}}, "optimizer.method"),
    ({"split": {"test": ["C:excitation_*"]}}, "names no run"),
    ({"split": {"test": ["excitation_*"]}}, "<run>:<bag-glob>"),
    ({"segments": ["approach"]}, "segments entry"),
    ({"replay": {"r0": {"sg_window": 4}}}, "sg_window"),
    ({"runs": {"A": {"path": "x", "sha256": "abc"}}}, "sha256"),
])
def test_invalid_values_are_refused(tmp_path, extra, message):
    with pytest.raises(ConfigError, match=message):
        load_config(_stage(tmp_path, **extra))


def test_schema_and_cycles(tmp_path):
    with pytest.raises(ConfigError, match="schema"):
        load_config(_write(tmp_path / "s.yaml", {"schema": "elastic_sim.sim2real/0", "asset": "a",
                                                  "runs": {"A": {"path": "x"}}}))
    _write(tmp_path / "a.yaml", {"extends": "b.yaml"})
    _write(tmp_path / "b.yaml", {"extends": "a.yaml"})
    with pytest.raises(ConfigError, match="extends cycle"):
        load_config(tmp_path / "a.yaml")
