"""S2R_01 T-2.1 / S2R_02 A-3: parameter registry v2 and the working overlay."""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.path.join(_REPO, "src"))

from elastic_sim.sim2real.config import load_config
from elastic_sim.sim2real.parameters import (OVERLAY_SCHEMA, ParameterSpec, Registry, RegistryError, UniverseEntry,
                                             check_write_target)

BASE_CONFIG = _REPO / "config" / "sim2real" / "kuka_lbr_iiwa_14_r820_base.yaml"
JOINTS = [f"iiwa_A{i}" for i in range(1, 8)]
LINKS = [f"iiwa_link_{i}" for i in range(1, 8)]


@pytest.fixture(scope="module")
def config():
    return load_config(BASE_CONFIG)


@pytest.fixture(scope="module")
def base(config):
    from elastic_sim.sim2real.plant import PlantBase

    return PlantBase.from_config(config)


@pytest.fixture(scope="module")
def registry(config, base):
    return Registry.from_config(config, base)


def test_wildcards_expand_on_the_iiwa(registry, base):
    free = registry.free_names
    assert [n for n in free if n.endswith(".mass")] == [f"inertial.{link}.mass" for link in LINKS]
    assert sum(n.startswith("inertial.") for n in free) == 7 * 5
    assert sorted(n for n in free if n.startswith("friction.link.")) == sorted(
        f"friction.link.{j}.{f}" for j in JOINTS for f in ("viscous", "coulomb"))
    # S2R_06 A-13: A is the test session, so its offsets are excluded (frozen) automatically.
    assert sorted(n for n in free if ".offset@" in n) == sorted(f"sensor.ft.{j}.offset@B" for j in JOINTS)
    assert registry.specs["sensor.ft.iiwa_A1.offset@A"].frozen
    assert registry.n_free == 35 + 14 + 7 + 7
    # Frozen in every stage (S2R_02 A-1), at the round-6 nominal.
    stiffness = registry.specs["transmission.iiwa_A1.stiffness"]
    assert stiffness.frozen and stiffness.value == 2.4e4
    assert registry.specs["friction.link.iiwa_A3.coulomb"].mean == 3.0
    # The URDF's own numbers are the inertial prior means.
    assert registry.specs["inertial.iiwa_link_2.mass"].mean == base.inertials["iiwa_link_2"].mass == 5.688888
    # The last matching pattern wins: A5-A7 offsets are bounded at +-1 N m.
    assert registry.specs["sensor.ft.iiwa_A6.offset@B"].upper == 1.0
    assert registry.specs["sensor.ft.iiwa_A2.offset@B"].upper == 2.0
    # Discrete parameters stay out of the optimizer's box.
    assert registry.specs["controller.delay_cycles"].discrete and "controller.delay_cycles" not in free


def test_encode_decode_round_trip_and_scales(registry):
    rng = np.random.default_rng(0)
    z = rng.uniform(-1.0, 1.0, registry.n_free)
    theta = registry.decode(z)
    np.testing.assert_allclose(registry.encode(theta), z, rtol=0, atol=1e-12)
    for name in registry.free_names:
        spec = registry.specs[name]
        assert spec.lower <= theta[name] <= spec.upper
    mass = registry.specs["inertial.iiwa_link_1.mass"]
    # Log scale: the box centre is the geometric mean of the bounds.
    assert mass.from_unit(0.0) == pytest.approx(math.sqrt(mass.lower * mass.upper))
    com = registry.specs["inertial.iiwa_link_1.com_z"]
    assert com.from_unit(0.0) == pytest.approx(0.5 * (com.lower + com.upper))
    assert mass.from_unit(-5.0) == mass.lower and mass.from_unit(5.0) == mass.upper
    with pytest.raises(RegistryError, match="free parameters"):
        registry.decode(z[:-1])


def test_prior_penalty_is_zero_at_the_mean_and_matches_the_residual_rows(registry):
    theta = registry.values()
    assert registry.prior_penalty(theta) == 0.0
    name = "inertial.iiwa_link_2.mass"
    spec = registry.specs[name]
    theta[name] = spec.mean * math.exp(2.0 * spec.sigma)  # two log-sigmas up
    assert registry.prior_penalty(theta) == pytest.approx(0.5 * 4.0)
    rows = registry.prior_residuals(theta)
    assert 0.5 * float(rows @ rows) == pytest.approx(registry.prior_penalty(theta))
    offset = "sensor.ft.iiwa_A1.offset@B"
    theta[offset] = 0.25
    assert registry.prior_penalty(theta) == pytest.approx(2.0 + 0.5 * (0.25 / 0.5) ** 2)


def test_refuses_initial_value_out_of_bounds_and_bad_specs(registry):
    with pytest.raises(RegistryError, match="outside its bounds"):
        registry.with_values({"inertial.iiwa_link_1.mass": 100.0})
    with pytest.raises(RegistryError, match="outside its bounds"):
        ParameterSpec("x", value=3.0, mean=1.0, sigma=0.1, scale="linear", lower=0.0, upper=2.0, frozen=False)
    with pytest.raises(RegistryError, match="needs a prior sigma"):
        ParameterSpec("x", value=1.0, mean=1.0, sigma=None, scale="linear", lower=0.0, upper=2.0, frozen=False)
    with pytest.raises(RegistryError, match="lower > 0"):
        ParameterSpec("x", value=1.0, mean=1.0, sigma=0.1, scale="log", lower=0.0, upper=2.0)
    universe = {"a.b.c": UniverseEntry(1.0, "test"), "a.b.d": UniverseEntry(None, "")}
    prior = {"a.b.*": {"scale": "linear", "bounds": [0.0, 2.0], "sigma": 0.1}}
    with pytest.raises(RegistryError, match="matches no parameter"):
        Registry.build(universe, prior, ["a.x.*"])
    with pytest.raises(RegistryError, match="no prior mean"):
        Registry.build(universe, prior, ["a.b.*"])
    with pytest.raises(RegistryError, match="both free and frozen"):
        Registry.build(universe, prior, ["a.b.c"], ["a.b.c"])
    with pytest.raises(RegistryError, match="exactly one of bounds"):
        Registry.build(universe, {"a.b.c": {"sigma": 0.1}}, ["a.b.c"])
    with pytest.raises(RegistryError, match="acts only in"):
        Registry.build(universe, {"a.b.c": {"bounds": [0, 2], "sigma": 0.1, "modes": ["R2"]}}, ["a.b.c"], mode="R0")


def test_registry_yaml_round_trip(registry):
    again = Registry.from_yaml(registry.to_yaml())
    assert again.free_names == registry.free_names
    assert again.values() == registry.values()
    for name, spec in registry.specs.items():
        assert again.specs[name] == spec


def test_overlay_round_trip_is_bit_exact(registry, tmp_path):
    rng = np.random.default_rng(7)
    theta = registry.decode(rng.uniform(-1.0, 1.0, registry.n_free))
    path = registry.write_overlay(tmp_path / "working.yaml", theta, stage="test")
    raw = yaml.safe_load(path.read_text())
    assert raw["schema"] == OVERLAY_SCHEMA and raw["kind"] == "working" and raw["stage"] == "test"
    assert list(raw["parameters"]) == list(registry.free_names)  # only the free parameters
    overlay = Registry.from_overlay(path)
    back = {name: spec.value for name, spec in overlay.specs.items()}
    assert all(back[name] == theta[name] for name in registry.free_names)  # bit for bit
    applied = registry.with_overlay(path)
    assert all(applied.values()[n] == theta[n] for n in registry.free_names)
    assert applied.specs["inertial.iiwa_link_1.mass"].mean == registry.specs["inertial.iiwa_link_1.mass"].mean
    # As the next stage's prior (S2R_01 §9): the means move to the values.
    as_prior = registry.with_overlay(path, as_prior=True)
    assert as_prior.prior_penalty(as_prior.values()) == 0.0


def test_overlay_writes_only_under_data_sim2real(registry):
    repo = _REPO
    check_write_target(repo / "data" / "sim2real" / "kuka" / "run" / "working.yaml")
    check_write_target(repo / "config" / "sim2real" / "calibrated" / "x.yaml", publish=True)
    for target in (repo / "config" / "sim2real" / "calibrated" / "x.yaml",
                   repo / "assets" / "robots" / "x.yaml",
                   repo / "config" / "identification" / "x.yaml",
                   repo / "docs" / "PARAMETER_PROVENANCE.md"):
        with pytest.raises(RegistryError, match="refusing to write"):
            check_write_target(target)
    with pytest.raises(RegistryError, match="refusing to write"):
        registry.write_overlay(repo / "config" / "sim2real" / "oops.yaml")
    assert not (repo / "config" / "sim2real" / "oops.yaml").exists()


def test_table_lists_free_parameters_with_prior_distance(registry):
    theta = registry.values()
    theta["sensor.ft.iiwa_A1.gain"] = 1.02
    table = registry.to_table(theta)
    assert table.count("\n") == registry.n_free + 1
    assert "| `sensor.ft.iiwa_A1.gain` | 1.02 | 1 | 0.01 | linear | [0.95, 1.05] | +2.00 |" in table


def test_child_prior_override_takes_precedence_over_a_later_base_pattern(tmp_path, base):
    """S2R_04 R-1: a pattern re-declared by an extending file moves to the end,
    so it beats a more specific base pattern that came after it."""
    child = tmp_path / "stage.yaml"
    child.write_text(yaml.safe_dump({"extends": str(BASE_CONFIG), "parameters": {"priors": {
        "sensor.ft.*.offset@*": {"offset": [-0.5, 0.5]}}}}))
    config = load_config(child)
    patterns = list(config.parameters.priors)
    assert patterns[-1] == "sensor.ft.*.offset@*"
    # The re-declared block still merges with the base's keys.
    assert config.parameters.priors["sensor.ft.*.offset@*"]["sigma"] == 0.5
    registry = Registry.from_config(config, base)
    for joint in ("iiwa_A2", "iiwa_A6"):  # A6 was +-1 from the later base pattern
        spec = registry.specs[f"sensor.ft.{joint}.offset@B"]
        assert (spec.lower, spec.upper) == (-0.5, 0.5)
