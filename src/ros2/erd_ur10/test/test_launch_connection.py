"""UR_01 U-1.1: the launch takes the connection from the lab file."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

LAUNCH = Path(__file__).resolve().parents[1] / "launch" / "ur10.launch.py"


@pytest.fixture(scope="module")
def launch_module():
    pytest.importorskip("launch")
    spec = importlib.util.spec_from_file_location("ur10_launch", LAUNCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FILE = SimpleNamespace(robot_ip="192.168.56.10", reverse_ip="192.168.56.1", headless_mode=False)


def test_the_file_alone_is_enough(launch_module):
    resolved = launch_module.resolve_connection({"robot_ip": None, "reverse_ip": None, "headless_mode": None}, FILE)
    assert resolved == {"robot_ip": "192.168.56.10", "reverse_ip": "192.168.56.1", "headless_mode": "false"}


def test_an_equal_argument_is_fine(launch_module):
    given = {"robot_ip": "192.168.56.10", "reverse_ip": None, "headless_mode": "False"}
    assert launch_module.resolve_connection(given, FILE)["headless_mode"] == "false"


def test_a_differing_argument_is_refused_naming_both_values(launch_module):
    with pytest.raises(RuntimeError, match=r"robot_ip:=10.0.0.2 differs from .*connection.robot_ip=192.168.56.10"):
        launch_module.resolve_connection({"robot_ip": "10.0.0.2", "reverse_ip": None, "headless_mode": None}, FILE)


def test_without_a_lab_file_the_old_defaults_apply(launch_module):
    resolved = launch_module.resolve_connection({"robot_ip": None, "reverse_ip": "1.2.3.4", "headless_mode": None}, None)
    assert resolved == {"robot_ip": "192.168.56.10", "reverse_ip": "1.2.3.4", "headless_mode": "false"}
