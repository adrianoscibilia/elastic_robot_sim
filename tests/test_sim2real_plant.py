"""S2R_01 T-2.2 / T-2.3 (gate G-5) / T-2.4 / T-2.5: one parameter set, two engines."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.path.join(_REPO, "src"))

from elastic_sim.sim2real.config import load_config
from elastic_sim.sim2real.observation import IiwaJointTorqueSensor, Ur10CurrentProxy, observation_model
from elastic_sim.sim2real.parameters import Registry
from elastic_sim.sim2real.plant import (NONPHYSICAL_LOSS, BodyMap, LinkInertial, PlantBase, PlantParameters,
                                        SensorCalibration, add_intermediate_bodies)

BASE_CONFIG = _REPO / "config" / "sim2real" / "kuka_lbr_iiwa_14_r820_base.yaml"
G5_POSES = 50


@pytest.fixture(scope="module")
def base():
    return PlantBase.from_config(load_config(BASE_CONFIG))


@pytest.fixture(scope="module")
def registry(base):
    return Registry.from_config(load_config(BASE_CONFIG), base)


def _pinocchio(base):
    from elastic_sim import identification as idn

    return idn.build_model(base.asset)


def _mujoco(base, *, elastic: bool):
    import mujoco

    from elastic_sim.generic_mujoco_runner import _build_model, _elastic_addresses, _joint_addresses
    from elastic_sim.torque_runners import TransmissionSpec, neutralize_mujoco_passive

    names = tuple(base.joint_names)
    if elastic:
        transmission = TransmissionSpec.uniform(len(names), 1.0e4)
        model, _ = _build_model(base.asset, mujoco, 1.0e-4, elastic_transmissions=transmission.as_mapping(names))
        addresses = _elastic_addresses(model, mujoco, names)
        qpos = np.asarray([addresses[n]["motor_qpos"] for n in names])
        dofs = np.asarray([addresses[n]["motor_dof"] for n in names])
        link_dofs = np.asarray([addresses[n]["elastic_dof"] for n in names])
    else:
        model, _ = _build_model(base.asset, mujoco, 1.0e-3)
        active = _joint_addresses(model, mujoco, names)
        qpos = np.asarray([a for a, _ in active])
        dofs = link_dofs = np.asarray([d for _, d in active])
    neutralize_mujoco_passive(model)
    model.dof_armature[:] = 0.0
    return mujoco, model, mujoco.MjData(model), qpos, dofs, link_dofs


def _poses(base, rng, count):
    joints = base.asset.resolve_active_joints()
    low = np.asarray([j.lower for j in joints])
    high = np.asarray([j.upper for j in joints])
    return rng.uniform(low, high, size=(count, len(joints)))


def _random_theta(registry, base, rng):
    """G-5's random theta: masses +-30 %, COM +-3 cm, inertia scale x[0.7, 1.4]."""
    theta = registry.values()
    for link, nominal in base.inertials.items():
        theta[f"inertial.{link}.mass"] = nominal.mass * rng.uniform(0.7, 1.3)
        for i, axis in enumerate("xyz"):
            theta[f"inertial.{link}.com_{axis}"] = nominal.com[i] + rng.uniform(-0.03, 0.03)
        theta[f"inertial.{link}.inertia_scale"] = rng.uniform(0.7, 1.4)
    return theta


def test_theta0_is_bit_identical_to_the_untouched_asset(base, registry):
    plant = PlantParameters.from_theta(base, registry.values())
    rng = np.random.default_rng(1)
    q, dq, ddq = (rng.uniform(-1.0, 1.0, (20, 7)) for _ in range(3))
    q = _poses(base, rng, 20)

    pin, untouched, data0 = _pinocchio(base)
    _, model, data = _pinocchio(base)
    plant.apply_to_pinocchio(pin, model)
    for k in range(len(q)):
        np.testing.assert_array_equal(pin.rnea(model, data, q[k], dq[k], ddq[k]),
                                      pin.rnea(untouched, data0, q[k], dq[k], ddq[k]))

    for elastic in (False, True):
        mujoco, ref, ref_data, qpos, dofs, _ = _mujoco(base, elastic=elastic)
        _, mod, mod_data, _, _, _ = _mujoco(base, elastic=elastic)
        plant.apply_to_mujoco(mujoco, mod, BodyMap.build(mujoco, mod, base), mod_data)
        for array in ("body_mass", "body_ipos", "body_inertia", "body_iquat"):
            np.testing.assert_array_equal(getattr(mod, array), getattr(ref, array))
        for k in range(len(q)):
            for m, d in ((ref, ref_data), (mod, mod_data)):
                d.qpos[qpos] = q[k]
                d.qvel[dofs] = dq[k]
                mujoco.mj_forward(m, d)
            np.testing.assert_array_equal(mod_data.qfrc_bias, ref_data.qfrc_bias)


@pytest.mark.parametrize("elastic", [False, True], ids=["rigid", "elastic"])
def test_gate_g5_pinocchio_mujoco_agree_under_overrides(base, registry, elastic):
    """G-5: gravity (rel. 1e-6) and mass matrix (rel. 1e-5) at 50 random poses."""
    rng = np.random.default_rng(5 + elastic)
    plant = PlantParameters.from_theta(base, _random_theta(registry, base, rng))
    assert plant.check_physical() == []
    mujoco, model, data, qpos, dofs, link_dofs = _mujoco(base, elastic=elastic)
    body_map = BodyMap.build(mujoco, model, base)
    plant.apply_to_mujoco(mujoco, model, body_map, data)
    pin, pmodel, pdata = _pinocchio(base)
    plant.apply_to_pinocchio(pin, pmodel)
    if elastic:
        assert len(body_map.intermediate) == 7
        add_intermediate_bodies(pin, pmodel, mujoco, model, body_map)
    worst_g, worst_m = 0.0, 0.0
    full = np.zeros((model.nv, model.nv))
    for q in _poses(base, rng, G5_POSES):
        data.qpos[:] = 0.0
        data.qpos[qpos] = q  # elastic build: deflection 0
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        gravity_mj = data.qfrc_bias[link_dofs]
        mujoco.mj_fullM(model, full, data.qM)
        mass_mj = full[np.ix_(dofs, dofs)]
        gravity_pin = pin.computeGeneralizedGravity(pmodel, pdata, q)
        mass_pin = pin.crba(pmodel, pdata, q)
        mass_pin = np.triu(mass_pin) + np.triu(mass_pin, 1).T
        worst_g = max(worst_g, np.linalg.norm(gravity_mj - gravity_pin) / np.linalg.norm(gravity_pin))
        worst_m = max(worst_m, np.linalg.norm(mass_mj - mass_pin) / np.linalg.norm(mass_pin))
    print(f"\nG-5 {'elastic' if elastic else 'rigid'}: worst relative error gravity {worst_g:.2e}, "
          f"mass matrix {worst_m:.2e} over {G5_POSES} poses")
    assert worst_g <= 1e-6 and worst_m <= 1e-5


def test_full_tensor_path_matches_the_uniform_scale_path(base):
    """The eigen-decomposition setter (future full-tensor parameters) gives
    the same MuJoCo body as the scale path, up to the eigensolver."""
    plant = PlantParameters.nominal(base)
    scaled = {link: LinkInertial(i.mass, i.com, 1.3 * i.inertia, urdf_scale=1.3) for link, i in plant.inertials.items()}
    full = {link: LinkInertial(i.mass, i.com, i.inertia, urdf_scale=None) for link, i in scaled.items()}
    mujoco, a, data_a, qpos, dofs, _ = _mujoco(base, elastic=False)
    _, b, data_b, _, _, _ = _mujoco(base, elastic=False)
    import dataclasses

    dataclasses.replace(plant, inertials=scaled).apply_to_mujoco(mujoco, a, BodyMap.build(mujoco, a, base), data_a)
    dataclasses.replace(plant, inertials=full).apply_to_mujoco(mujoco, b, BodyMap.build(mujoco, b, base), data_b)
    full_a, full_b = np.zeros((a.nv, a.nv)), np.zeros((b.nv, b.nv))
    for q in _poses(base, np.random.default_rng(2), 10):
        for m, d, out in ((a, data_a, full_a), (b, data_b, full_b)):
            d.qpos[qpos] = q
            mujoco.mj_forward(m, d)
            mujoco.mj_fullM(m, out, d.qM)
        # MuJoCo's importer diagonalises the URDF tensor with its own
        # iterative solver, good to ~1e-8; numpy's eigh is exact to rounding.
        assert np.linalg.norm(full_b - full_a) <= 1e-7 * np.linalg.norm(full_a)


def test_check_physical_flags_a_non_physical_tensor(base):
    plant = PlantParameters.nominal(base)
    assert plant.check_physical() == [] and plant.physical_failure() is None
    bad = dict(plant.inertials)
    # Principal moments (1, 1, 3): positive definite, but 1 + 1 < 3.
    bad["iiwa_link_3"] = LinkInertial(3.0, [0.0, 0.03, 0.12], np.diag([1.0, 1.0, 3.0]) * 1e-2)
    bad["iiwa_link_5"] = LinkInertial(-1.0, [0.0, 0.0, 0.0], np.eye(3) * 1e-3)
    import dataclasses

    broken = dataclasses.replace(plant, inertials=bad)
    problems = broken.check_physical()
    assert any("iiwa_link_3" in p and "triangle" in p for p in problems)
    assert any("iiwa_link_3" in p and "pseudo-inertia" in p for p in problems)
    assert any("iiwa_link_5" in p and "mass" in p for p in problems)
    failure = broken.physical_failure()
    assert failure["loss"] == NONPHYSICAL_LOSS and failure["reasons"] == problems


def test_export_urdf_writes_only_its_target(base, registry, tmp_path):
    rng = np.random.default_rng(3)
    plant = PlantParameters.from_theta(base, _random_theta(registry, base, rng))
    before = {p: p.stat().st_mtime_ns for p in (_REPO / "assets").rglob("*") if p.is_file()}
    target = plant.export_urdf(tmp_path / "out")
    assert target.parent == (tmp_path / "out").resolve() and sorted(tmp_path.rglob("*.urdf")) == [target]
    after = {p: p.stat().st_mtime_ns for p in (_REPO / "assets").rglob("*") if p.is_file()}
    assert after == before
    import pinocchio as pin

    exported = pin.buildModelFromUrdf(str(target))
    for index, link in enumerate(base.link_names, start=1):
        np.testing.assert_allclose(exported.inertias[index].mass, plant.inertials[link].mass, rtol=1e-15)
        np.testing.assert_allclose(exported.inertias[index].lever, plant.inertials[link].com, rtol=1e-15)
    with pytest.raises(ValueError, match="never writes under assets"):
        plant.export_urdf(_REPO / "assets" / "robots" / "x.urdf")


def test_from_theta_maps_every_group(base, registry):
    theta = registry.values()
    theta["friction.link.iiwa_A2.coulomb"] = 9.0
    theta["sensor.ft.iiwa_A4.offset@B"] = -0.3
    theta["sensor.ft.iiwa_A4.gain"] = 1.01
    theta["inertial.iiwa_link_6.inertia_scale"] = 1.5
    plant = PlantParameters.from_theta(base, theta)
    # S2R_04 A-8: the production plant's epsilon (identification.COULOMB_EPSILON).
    assert plant.link_friction.coulomb[1] == 9.0 and plant.link_friction.epsilon == 1e-3
    assert plant.extras.link_friction is plant.link_friction and plant.extras.epsilon == 1e-3
    assert plant.sensor.offset("B")[3] == -0.3 and plant.sensor.offset("A")[3] == 0.0
    assert plant.sensor.gain[3] == 1.01
    np.testing.assert_array_equal(plant.inertials["iiwa_link_6"].inertia, 1.5 * base.inertials["iiwa_link_6"].inertia)
    np.testing.assert_array_equal(plant.transmission.stiffness, base.stiffness)
    assert plant.controller is not None and plant.controller.delay_cycles == 1
    assert plant.extras.stiffness_nonlinearity is not None and plant.extras.has_transmission_error
    with pytest.raises(KeyError, match="unknown parameter groups"):
        PlantParameters.from_theta(base, {"bogus.x": 1.0})


def test_iiwa_observation_model_on_a_synthetic_signal():
    t = np.linspace(0.0, 10.0, 1001)
    tau = np.column_stack([np.sin((j + 1) * t) * (10.0 - j) for j in range(7)])
    gain = np.linspace(0.97, 1.03, 7)
    sensor = SensorCalibration(gain, {"A": np.linspace(-0.5, 0.5, 7), "B": np.zeros(7)})
    model = observation_model("kuka_lbr_iiwa_14_r820", sensor)
    assert isinstance(model, IiwaJointTorqueSensor) and model.channel == "ft"
    ft = model.observe(tau, "A")
    np.testing.assert_allclose(ft, tau * gain + np.linspace(-0.5, 0.5, 7), rtol=0, atol=1e-15)
    np.testing.assert_allclose(model.invert(ft, "A"), tau, rtol=0, atol=1e-12)
    np.testing.assert_allclose(model.residual(tau, ft, "A"), 0.0, atol=1e-15)
    np.testing.assert_allclose(model.observe(tau, "C"), tau * gain)  # unknown run: no offset
    with pytest.raises(NotImplementedError, match="Phase 7"):
        Ur10CurrentProxy(sensor)
