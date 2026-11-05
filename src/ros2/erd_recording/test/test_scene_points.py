"""Scene walls from measured points (owner 2026-11-05): ``wall_from_points`` and ``floor_box``."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from erd_recording.config import ConfigError, _load_scene
from erd_recording.scene_points import ScenePointsError, wall_box_from_points

LAB = Path(__file__).resolve().parents[1] / "config" / "lab"

IIWA_P1, IIWA_P2, IIWA_IN = [-363.89, -467.28, 66], [559.6, 193.9, 66], [-319, 459, 406]   # mm
UR_P1, UR_P2, UR_IN = [-0.5386, 0.5249], [0.3488, -0.6662], [-0.496, -0.369]               # m


def test_iiwa_wall_equals_the_reviewed_box():
    wall = wall_box_from_points(IIWA_P1, IIWA_P2, IIWA_IN, unit="mm")
    assert wall["size"] == (6.0, 0.2, 4.0)
    assert np.allclose(wall["xyz"], (0.156075, -0.217995, 1.0), atol=1e-4)
    assert np.allclose(wall["rpy"], (0.0, 0.0, 0.621356), atol=1e-6)
    assert wall["derived"]["origin_distance_m"] == pytest.approx(0.168, abs=1e-3)   # base on the allowed side


def test_ur10_guard_equals_the_reviewed_box():
    wall = wall_box_from_points(UR_P1, UR_P2, UR_IN, unit="m", offset=0.45)
    assert np.allclose(wall["xyz"], (0.346176, 0.257910, 1.0), atol=1e-4)
    assert np.allclose(wall["rpy"], (0.0, 0.0, 2.211106), atol=1e-6)
    assert wall["derived"]["origin_distance_m"] == pytest.approx(-0.118, abs=1e-3)  # base on the window side


def test_point_order_does_not_matter_and_inside_picks_the_side():
    a = wall_box_from_points(IIWA_P1, IIWA_P2, IIWA_IN, unit="mm")
    b = wall_box_from_points(IIWA_P2, IIWA_P1, IIWA_IN, unit="mm")
    assert np.allclose(a["xyz"], b["xyz"]) and np.isclose(a["rpy"][2], b["rpy"][2])
    # the box lies behind the line: its centre is on the opposite side of `inside`
    n = np.asarray(a["derived"]["normal_inside"])
    mid = np.asarray(a["derived"]["midpoint"])
    assert (np.asarray(a["xyz"][:2]) - mid) @ n == pytest.approx(-0.1)
    # local +y of the box is the allowed-side normal
    yaw = a["rpy"][2]
    assert np.allclose((-np.sin(yaw), np.cos(yaw)), n)


@pytest.mark.parametrize("kwargs, match", [
    (dict(p1=[0, 0], p2=[0.01, 0], inside=[0, 1]), "apart"),
    (dict(p1=[0, 0], p2=[1, 0], inside=[0.5, 0.001]), "from the P1-P2 line"),
    (dict(p1=[0, 0], p2=[1, 0], inside=[0, 1], unit="cm"), "unit"),
    (dict(p1=[0, 0], p2=[1, 0], inside=[0, 1], offset=-0.1), "offset"),
    (dict(p1=[0, 0], p2=[1, 0], inside=[0, 1], z=[3, -1]), "low < high"),
])
def test_bad_walls_are_refused(kwargs, match):
    with pytest.raises(ScenePointsError, match=match):
        wall_box_from_points(**kwargs)


def _scene(path: Path):
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return _load_scene(raw, path.name, raw["robot"])


@pytest.mark.parametrize("new, old", [
    ("iiwa_probe.yaml", "iiwa_fast.yaml"),
    ("iiwa_probe_j150.yaml", "iiwa_fast.yaml"),
    ("ur10_probe.yaml", "ur10_fast.yaml"),
])
def test_point_scenes_equal_the_reviewed_box_scenes(new, old):
    """After clearance and the frame transform, the scenes written from P1/P2
    are the reviewed hand-computed ones to 0.1 mm."""
    a, b = _scene(LAB / new), _scene(LAB / old)
    assert [o.id for o in a.objects] == [o.id for o in b.objects]
    for x, y in zip(a.objects, b.objects):
        assert np.allclose(x.size, y.size, atol=1e-9)
        assert np.allclose(x.xyz, y.xyz, atol=1e-4)
        assert np.allclose(x.rpy, y.rpy, atol=1e-6)
    assert {o.source_type for o in a.objects} == {"floor_box", "wall_from_points"}


def test_unknown_key_in_a_point_wall_is_refused():
    raw = yaml.safe_load((LAB / "iiwa_probe.yaml").read_text(encoding="utf-8"))
    raw["scene"]["objects"][1]["insde"] = [0, 0]
    with pytest.raises(ConfigError, match="unknown key"):
        _load_scene(raw, "x", raw["robot"])
