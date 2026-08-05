"""Tests for portable per-joint elastic simulation settings."""

import os
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(_REPO / "src"))

from elastic_sim.assets import AssetRegistry
from elastic_sim.elastic_settings import load_simulation_settings


def test_example_elastic_settings_cover_every_discovered_ur10_joint():
    asset = AssetRegistry.for_repository(_REPO).load("ur10")
    settings = load_simulation_settings(_REPO / "config/assets/ur10_elastic_example.yaml", asset)
    assert tuple(settings["transmissions"]) == asset.joint_names
    assert settings["transmissions"]["wrist_3_joint"]["stiffness"] == 800.0
    assert settings["transmissions"]["elbow_joint"]["intermediate_mass"] == 0.10


def test_settings_reject_joint_not_selected_by_asset(tmp_path: Path):
    asset = AssetRegistry.for_repository(_REPO).load("ur10")
    path = tmp_path / "bad.yaml"
    path.write_text("elastic:\n  joints:\n    missing_joint:\n      stiffness: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non-active joints"):
        load_simulation_settings(path, asset)


def test_settings_reject_another_assets_joint_configuration(tmp_path: Path):
    asset = AssetRegistry.for_repository(_REPO).load("ur10")
    path = tmp_path / "wrong-asset.yaml"
    path.write_text("asset: baxter_left\nelastic: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not 'ur10'"):
        load_simulation_settings(path, asset)
