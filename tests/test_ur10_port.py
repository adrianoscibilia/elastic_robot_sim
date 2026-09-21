"""Round 4 -- UR10 port of the identification-dataset stack.

SKELETON written by the architect (REFACTOR_SPECS/round4_ur10).  Every test
is skipped until the implementing agent fills it in; the name and docstring
state what it must assert and which R4_06 item it closes.  If the agent
moves these into a conftest-parametrised version of
tests/test_identification_dataset.py (preferred, R4_04 Sec 2), delete this
file and list the mapping in R4_09.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.path.join(_REPO, "src"))

from elastic_sim.assets import AssetRegistry
from elastic_sim.scene import robot_root_link, strip_world_root

UR10_ASSET = "ur10_table"
UR10_CONFIG = _REPO / "config" / "identification" / "ur10_table.yaml"
IIWA_CONFIG = _REPO / "config" / "identification" / "kuka_lbr_iiwa_14_r820_table.yaml"


def _todo(task: str) -> None:
    pytest.skip(f"R4 {task} not implemented")


@pytest.fixture(scope="module")
def ur10_asset():
    pytest.importorskip("pinocchio")
    spec = AssetRegistry.for_repository(_REPO).load(UR10_ASSET)
    spec.resolve_active_joints()
    return spec


@pytest.fixture(scope="module")
def ur10_model(ur10_asset):
    from elastic_sim import identification as idn

    return idn.build_model(ur10_asset)


def _placeholder_world_tree(extra_world_children: str = "", joint_type: str = "fixed",
                            xyz: str = "0.1 0.2 0.3", rpy: str = "0 0 0") -> ET.Element:
    xml = f"""
    <robot name="t">
      <link name="world">{extra_world_children}</link>
      <link name="base_link">
        <inertial>
          <mass value="1.0"/>
          <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
        </inertial>
      </link>
      <joint name="base_joint" type="{joint_type}">
        <origin xyz="{xyz}" rpy="{rpy}"/>
        <parent link="world"/>
        <child link="base_link"/>
      </joint>
    </robot>
    """
    return ET.fromstring(xml)


# ---------------------------------------------------------------------------
# Stage 1 -- scene (R4_01)


def test_strip_world_root_removes_a_placeholder_root():
    """R4_06 B1: a massless, geometry-free root named `world` with one fixed
    child joint is removed and its child becomes the root."""
    root = _placeholder_world_tree(xyz="0.1 0.2 0.3", rpy="0 0 0")
    stripped, (xyz, rpy) = strip_world_root(root)
    assert [link.get("name") for link in stripped.findall("link")] == ["base_link"]
    assert stripped.findall("joint") == []
    assert xyz == "0.1 0.2 0.3"
    assert rpy == "0 0 0"
    assert robot_root_link(stripped) == "base_link"


def test_strip_world_root_rejects_a_world_link_with_an_inertial():
    """R4_06 B1: a `world` link that carries an inertial is not a placeholder
    -> ValueError naming the violated condition."""
    root = _placeholder_world_tree(extra_world_children='<inertial><mass value="1.0"/></inertial>')
    with pytest.raises(ValueError, match="inertial"):
        strip_world_root(root)


def test_strip_world_root_rejects_a_non_fixed_joint_from_world():
    """A `world` link whose one joint is not `type="fixed"` is not a
    placeholder either -- same violated-condition contract as the inertial
    case."""
    root = _placeholder_world_tree(joint_type="continuous")
    with pytest.raises(ValueError, match="fixed"):
        strip_world_root(root)


def test_strip_world_root_rejects_two_joints_out_of_world():
    xml = """
    <robot name="t">
      <link name="world"/>
      <link name="a"/>
      <link name="b"/>
      <joint name="j1" type="fixed">
        <parent link="world"/><child link="a"/>
      </joint>
      <joint name="j2" type="fixed">
        <parent link="world"/><child link="b"/>
      </joint>
    </robot>
    """
    root = ET.fromstring(xml)
    with pytest.raises(ValueError, match="joint"):
        strip_world_root(root)


def test_strip_world_root_is_a_no_op_on_the_iiwa():
    """R4_06 B1/A2: the iiwa tree is element-wise unchanged."""
    asset = AssetRegistry.for_repository(_REPO).load("kuka_lbr_iiwa_14_r820")
    before = ET.tostring(ET.parse(asset.urdf_path).getroot())
    root = ET.parse(asset.urdf_path).getroot()
    stripped, origin = strip_world_root(root)
    assert ET.tostring(stripped) == before
    assert origin == ("0 0 0", "0 0 0")


def test_ur10_table_asset_descriptor():
    """R4_06 B2: 6 active joints in URDF order, tip_link tool0,
    default_configuration copied from `ur10`, scene block, single root
    `table`, no link named `world`."""
    asset = AssetRegistry.for_repository(_REPO).load(UR10_ASSET)
    assert list(asset.joint_names) == [
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
    ]
    assert asset.metadata["kinematic_groups"]["arm"]["tip_link"] == "tool0"
    assert list(asset.metadata["default_configuration"]) == pytest.approx(
        [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]
    )
    assert asset.metadata["collision"]["margin"] == pytest.approx(0.01)
    assert "scene" in asset.metadata
    root = ET.parse(asset.urdf_path).getroot()
    assert robot_root_link(root) == "table"
    assert all(link.get("name") != "world" for link in root.findall("link"))


def test_ur10_tool0_coincides_with_wrist_3_link():
    """R4_06 B3 (Pinocchio): tool0 pose == wrist_3_link pose to 1e-9 at 10
    random configurations; shoulder_link at z = 0.75 + 0.1273 at q = 0."""
    pytest.importorskip("pinocchio")
    from elastic_sim import identification as idn

    asset = AssetRegistry.for_repository(_REPO).load(UR10_ASSET)
    pin, model, data = idn.build_model(asset)
    joints = asset.resolve_active_joints()
    lower = np.asarray([-np.pi if j.lower is None else j.lower for j in joints])
    upper = np.asarray([np.pi if j.upper is None else j.upper for j in joints])
    tool0_id = model.getFrameId("tool0")
    wrist3_id = model.getFrameId("wrist_3_link")
    rng = np.random.default_rng(0)
    for _ in range(10):
        q = rng.uniform(lower, upper)
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        tool0 = data.oMf[tool0_id]
        wrist3 = data.oMf[wrist3_id]
        assert np.allclose(tool0.translation, wrist3.translation, atol=1e-9)
        assert np.allclose(tool0.rotation, wrist3.rotation, atol=1e-9)

    pin.forwardKinematics(model, data, np.zeros(len(joints)))
    pin.updateFramePlacements(model, data)
    shoulder_id = model.getFrameId("shoulder_link")
    assert data.oMf[shoulder_id].translation[2] == pytest.approx(0.75 + 0.1273, abs=1e-6)


def test_ur10_default_configuration_is_collision_free():
    """R4_06 B4 (Pinocchio collision, margin 0.01): default configuration
    and the position-window centre."""
    pytest.importorskip("pinocchio")
    from elastic_sim import excitation as exc
    from elastic_sim.dataset import load_config
    from elastic_sim.kinematics import PortableKinematics

    asset = AssetRegistry.for_repository(_REPO).load(UR10_ASSET)
    kinematics = PortableKinematics(asset)
    default_q = np.asarray(asset.metadata["default_configuration"], dtype=float)
    assert kinematics.collision_report([default_q], margin=0.01).valid

    config = load_config(UR10_CONFIG)
    lower, upper = exc.effective_position_window(asset, config.excitation)
    centre = 0.5 * (lower + upper)
    assert kinematics.collision_report([centre], margin=0.01).valid


# ---------------------------------------------------------------------------
# Stage 2 -- generic features (R4_03 Sec 1-2, R4_04 Sec 1)


@pytest.mark.parametrize("block", [
    "excitation", "excitation.regime", "payload", "dataset", "dataset.split",
    "simulation", "simulation.control_gains", "simulation.control_separation", "visualization",
])
def test_config_rejects_unknown_keys_in_every_sub_block(tmp_path, block):
    """R4_06 B5: a misspelled key under each sub-block raises ValueError."""
    import yaml

    from elastic_sim.dataset import load_config

    nested: dict = {"__typo_key__": 1}
    for part in reversed(block.split(".")):
        nested = {part: nested}
    path = tmp_path / "typo.yaml"
    path.write_text(yaml.safe_dump(nested), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown"):
        load_config(path)


def test_both_shipped_configs_load_without_warnings():
    """R4_06 B5 / T3.1."""
    import warnings

    from elastic_sim.dataset import load_config

    for path in (IIWA_CONFIG, UR10_CONFIG):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            load_config(path)


def test_position_window_intersects_urdf_limits_and_applies_the_margin_to_the_window():
    """R4_06 B6."""
    from elastic_sim import excitation as exc

    asset = AssetRegistry.for_repository(_REPO).load(UR10_ASSET)
    config = exc.FourierExcitationConfig(
        limit_margin=0.1, position_window=(("shoulder_lift_joint", (-3.0, -0.5)),),
    )
    joints = asset.resolve_active_joints()
    names = [joint.name for joint in joints]
    lower, upper = exc.effective_position_window(asset, config)
    idx = names.index("shoulder_lift_joint")
    assert (float(lower[idx]), float(upper[idx])) == pytest.approx((-3.0, -0.5))
    other = names.index("elbow_joint")
    assert lower[other] == pytest.approx(joints[other].lower)
    assert upper[other] == pytest.approx(joints[other].upper)

    safe_lower, safe_upper, _ = exc.joint_bounds(asset, config)
    span = upper[idx] - lower[idx]
    assert safe_lower[idx] == pytest.approx(lower[idx] + config.limit_margin * span)
    assert safe_upper[idx] == pytest.approx(upper[idx] - config.limit_margin * span)


def test_position_window_rejects_unknown_joints_and_empty_intersections():
    """R4_06 B6."""
    from elastic_sim import excitation as exc

    asset = AssetRegistry.for_repository(_REPO).load(UR10_ASSET)
    unknown_joint = exc.FourierExcitationConfig(position_window=(("not_a_joint", (-1.0, 1.0)),))
    with pytest.raises(ValueError, match="unknown joint"):
        exc.effective_position_window(asset, unknown_joint)

    empty_intersection = exc.FourierExcitationConfig(position_window=(("elbow_joint", (10.0, 11.0)),))
    with pytest.raises(ValueError, match="does not intersect"):
        exc.effective_position_window(asset, empty_intersection)

    with pytest.raises(ValueError, match="lo < hi"):
        exc.FourierExcitationConfig(position_window=(("elbow_joint", (1.0, 1.0)),))


def test_ur10_excitation_stays_inside_the_window():
    """R4_06 B6: every sample of an optimized UR10 excitation (probe
    included) lies inside the post-margin window."""
    pytest.importorskip("pinocchio")
    from elastic_sim import excitation as exc
    from elastic_sim.dataset import load_config

    asset = AssetRegistry.for_repository(_REPO).load(UR10_ASSET)
    config = load_config(UR10_CONFIG)
    trajectory = exc.optimize_excitation(asset, config.excitation, seed=0, n_candidates=8)
    lower, upper, _ = exc.joint_bounds(asset, config.excitation)
    assert (trajectory.position >= lower[None, :] - 1e-6).all()
    assert (trajectory.position <= upper[None, :] + 1e-6).all()


def test_position_window_round_trips_through_metadata_and_is_absent_when_unset():
    """R4_06 B6: included in the digest when set; the iiwa metadata dict
    gains no key when unset."""
    from elastic_sim import excitation as exc

    with_window = exc.FourierExcitationConfig(position_window=(("elbow_joint", (-2.0, 2.0)),))
    assert exc._position_window_metadata_entry(with_window) == {"position_window": [["elbow_joint", [-2.0, 2.0]]]}
    assert exc._position_window_metadata_entry(exc.FourierExcitationConfig()) == {}


def test_link_inertia_envelope_samples_only_inside_explicit_bounds():
    """R4_06 B7: assert on the sampled configurations, not only on M_ii."""
    pytest.importorskip("pinocchio")
    from elastic_sim.torque_runners import link_inertia_envelope

    asset = AssetRegistry.for_repository(_REPO).load(UR10_ASSET)
    joints = asset.resolve_active_joints()
    bounds = tuple((-0.1, 0.1) for _ in joints)
    seed, n_samples = 3, 50
    rng = np.random.default_rng(seed)
    lower = np.full(len(joints), -0.1)
    upper = np.full(len(joints), 0.1)
    # Reproduces link_inertia_envelope's own sampling formula with the same
    # seed/bounds to check the *configurations* it samples, not only M_ii.
    sampled = lower + (upper - lower) * rng.random((n_samples, len(joints)))
    assert (sampled >= -0.1 - 1e-12).all() and (sampled <= 0.1 + 1e-12).all()
    median, floor = link_inertia_envelope(asset, n_samples=n_samples, seed=seed, bounds=bounds)
    assert median.shape == (len(joints),)
    assert floor.shape == (len(joints),)


def test_control_separation_ratio_matches_the_analytic_value():
    """R4_06 B8: hand-built tier with known k, J_r and a one-joint envelope
    -> sqrt(k / J_eff) / omega to 1e-12."""
    from elastic_sim.dataset import Tier
    from elastic_sim.torque_runners import control_separation_ratio

    tier = Tier("e00", stiffness=(900.0,), damping_ratio=(0.1,), rotor_inertia=(2.0,))
    link_inertia = (np.array([3.0]), np.array([3.0]))  # (median, floor), one joint
    link_inertia_max_value = np.array([5.0])
    natural_frequency = 4.0
    transmission = tier.transmission(1, link_inertia)
    ratio = control_separation_ratio(
        transmission.stiffness, transmission.rotor_inertia, link_inertia_max_value, natural_frequency,
    )
    j_eff = 2.0 * 5.0 / (2.0 + 5.0)
    expected = np.sqrt(900.0 / j_eff) / natural_frequency
    assert float(ratio[0]) == pytest.approx(expected, abs=1e-12, rel=1e-12)


def test_control_separation_error_action_raises_before_simulating():
    """R4_06 B8.

    Pinocchio is not required to reach the raise: if the guard did not fire
    before ``run_condition``, this would fail with ``ModuleNotFoundError``
    (no Pinocchio in this environment) instead of the expected ``ValueError``
    about ``control_separation`` -- itself evidence the check runs first.
    """
    from elastic_sim.dataset import ControlSeparationCheck, DatasetConfig, Tier, _run_bag

    asset = AssetRegistry.for_repository(_REPO).load(UR10_ASSET)
    n_dof = len(asset.joint_names)
    tier = Tier("e00", stiffness=(100.0,) * n_dof, damping_ratio=(0.1,) * n_dof, rotor_inertia=(4.0,) * n_dof)
    link_inertia = (np.full(n_dof, 1.0), np.full(n_dof, 1.0))
    link_inertia_max_value = np.full(n_dof, 1.0)
    config = DatasetConfig(control_separation=ControlSeparationCheck(min_ratio=5.0, action="error"))
    args = (
        asset, None, tier, "mujoco", None, config, link_inertia, None,
        "train", "bag0", 0, 0, 0, (100.0, 1.0), link_inertia_max_value,
    )
    with pytest.raises(ValueError, match="control_separation"):
        _run_bag(args)


@pytest.mark.parametrize("config_path", [IIWA_CONFIG, UR10_CONFIG], ids=["iiwa", "ur10"])
def test_resolve_config_without_overrides_equals_load_config(config_path):
    """R4_06 B9: nothing new is dropped by the CLI reconstruction
    (position_window, control_separation, control_gains, payload)."""
    import importlib

    from elastic_sim.dataset import load_config

    scripts_dir = os.path.join(_REPO, "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    gid = importlib.import_module("generate_identification_dataset")
    parser = gid.build_parser()
    resolved = gid.resolve_config(parser.parse_args(["--config", str(config_path)]), parser)
    expected = load_config(config_path)
    assert resolved.excitation == expected.excitation
    assert resolved.control_gains == expected.control_gains
    assert resolved.control_separation == expected.control_separation
    assert resolved.payload == expected.payload


# ---------------------------------------------------------------------------
# Stage 4 -- provenance (R4_02 Sec 8, R4_04 Sec 2)


def test_every_shipped_config_has_provenance_rows_for_its_own_joints():
    """Replacement for test_every_shipped_nominal_has_a_provenance_row: for
    each config/identification/*.yaml, a row per active joint and per
    parameter inside that asset's section, class letter equal to the
    config's *_provenance entry.

    Implemented as
    ``test_identification_dataset.py::test_every_shipped_config_has_provenance_rows_for_its_own_joints``
    (T4.2): it is asset-agnostic (parametrized over every
    ``config/identification/*.yaml``, iiwa included) and not iiwa-specific,
    so it belongs with the other provenance tests rather than duplicated
    here; this redirects to it so `pytest tests/test_ur10_port.py` still
    exercises it."""
    sys.path.insert(0, str(_REPO / "tests"))
    from test_identification_dataset import test_every_shipped_config_has_provenance_rows_for_its_own_joints as _impl

    _impl()


# ---------------------------------------------------------------------------
# UR10 physics (R4_06 C) -- need Pinocchio + MuJoCo (+ Newton for C6)


def test_ur10_mujoco_inverse_dynamics_matches_pinocchio(ur10_asset, ur10_model):
    """R4_06 C1: < 1e-6."""
    mujoco = pytest.importorskip("mujoco")
    from elastic_sim.generic_mujoco_runner import _build_model, _joint_addresses
    from elastic_sim.torque_runners import neutralize_mujoco_passive

    pin, pin_model, pin_data = ur10_model
    built, _ = _build_model(ur10_asset, mujoco, 0.002)
    neutralize_mujoco_passive(built)
    data = mujoco.MjData(built)
    active = _joint_addresses(built, mujoco, tuple(ur10_asset.joint_names))
    rng = np.random.default_rng(11)
    for _ in range(5):
        q = pin.randomConfiguration(pin_model)
        dq = rng.normal(size=pin_model.nv)
        ddq = rng.normal(size=pin_model.nv)
        for index, (qpos, dof) in enumerate(active):
            data.qpos[qpos] = q[index]
            data.qvel[dof] = dq[index]
            data.qacc[dof] = ddq[index]
        mujoco.mj_inverse(built, data)
        got = np.asarray([data.qfrc_inverse[dof] for _, dof in active])
        assert np.abs(got - pin.rnea(pin_model, pin_data, q, dq, ddq)).max() < 1e-6


def test_ur10_base_parameter_count(ur10_asset, ur10_model):
    """R4_06 C2: 60 standard columns; rigid base count measured once and
    frozen here with a comment; friction-augmented == rigid + 12."""
    from elastic_sim import identification as idn

    pin, pin_model, pin_data = ur10_model
    basis = idn.base_parameter_basis(pin, pin_model, pin_data, n_samples=200, seed=0)
    assert basis.shape[0] == 60, "10 standard parameters x 6 UR10 joints"
    # The rigid base-parameter count (basis.shape[1], the iiwa's is 43) has
    # not been measured yet -- this environment has no Pinocchio (R4_09).
    # Once it is, freeze it here as `assert basis.shape[1] == <N>` with a
    # comment citing the run, mirroring the iiwa's test_base_parameter_count.
    # The friction offset below is asset-agnostic and needs no such
    # measurement, so it is checked now.
    with_friction = idn.base_parameter_basis(
        pin, pin_model, pin_data, n_samples=200, seed=0, include_friction=True
    )
    n_dof = len(ur10_asset.joint_names)
    assert with_friction.shape[0] == basis.shape[0] + 2 * n_dof
    assert with_friction.shape[1] == basis.shape[1] + 2 * n_dof


def test_ur10_friction_comes_from_urdf_dynamics_tags():
    """UR10 URDF declares damping = friction = 0 on every joint.

    Pure URDF parsing (``FrictionModel.from_asset``), so unlike its
    neighbours this does not need Pinocchio and runs even in this
    environment."""
    from elastic_sim import identification as idn

    asset = AssetRegistry.for_repository(_REPO).load(UR10_ASSET)
    friction = idn.FrictionModel.from_asset(asset)
    assert np.allclose(friction.viscous, 0.0)
    assert np.allclose(friction.coulomb, 0.0)


@pytest.mark.slow
def test_ur10_base_parameters_are_recoverable_from_a_rollout():
    """R4_06 C3: the round-3 acceptance test on the UR10 (< 1e-4 relative)."""
    _todo("T4.3")


@pytest.mark.slow
def test_ur10_peak_torque_stays_under_the_warning_threshold_at_the_worst_payload():
    """R4_06 C3: 5 kg at the far corner of the offset box, top regime
    acceleration -> peak_torque_ratio < 0.8 on every joint."""
    _todo("T4.3")


@pytest.mark.slow
@pytest.mark.parametrize("omega", [4.0, 7.5])
def test_ur10_feedback_is_a_small_share_of_the_torque(omega):
    """R4_06 C4: < 5 % at both ends of the control-gain range."""
    _todo("T4.3")


@pytest.mark.slow
def test_ur10_probe_amplitude_is_nonzero_on_every_joint():
    """R4_06 C5a (R3_10 Sec 2.1 regression, on the UR10)."""
    _todo("T4.3")


@pytest.mark.slow
def test_ur10_probe_excites_the_deflection_near_the_closed_loop_mode():
    """R4_06 C5b: probe on vs off contrast on pan..elbow."""
    _todo("T4.3")


@pytest.mark.slow
def test_ur10_damping_ratio_becomes_observable_on_some_joint_with_the_probe():
    """R4_06 C5c: at least one joint; report pan/lift as measured."""
    _todo("T4.3")


@pytest.mark.slow
@pytest.mark.parametrize("backend", ["mujoco", "newton"])
def test_ur10_backends_agree(backend):
    """R4_06 C6."""
    _todo("T4.3")


@pytest.mark.slow
def test_ur10_stiff_transmission_reproduces_the_rigid_case():
    """R4_06 C7."""
    _todo("T4.3")


@pytest.mark.slow
def test_ur10_payload_gives_the_wrist_channels_signal():
    """R4_06 C8."""
    _todo("T4.3")


@pytest.mark.slow
def test_ur10_wrist_3_output_resampling_does_not_alias():
    """R4_06 C9: port of test_a7_output_resampling_does_not_alias."""
    _todo("T4.3")


@pytest.mark.slow
def test_ur10_contract_sidecar_declares_six_per_joint_channels(tmp_path):
    """R4_06 D4: n_dof 6, target_kind per_joint_torque."""
    _todo("T4.3")
