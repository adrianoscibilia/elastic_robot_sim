"""The speed-slider input recipe and trigger/release logic (RR_08 S2d, P4
item 9), tested against a fake RTDE connection (no live RTDE server
available in this environment, mirrors test_rtde_logger.py)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from erd_ur10.speed_slider_abort import (
    SPEED_SLIDER_RECIPE,
    build_speed_slider_connection,
    release_speed_slider,
    trigger_speed_slider,
)


def _fake_setup():
    return SimpleNamespace(**{name: None for name, _ in SPEED_SLIDER_RECIPE})


def test_trigger_sets_mask_and_fraction_then_sends():
    connection = MagicMock()
    setup = _fake_setup()
    trigger_speed_slider(connection, setup, fraction=0.5)
    assert setup.speed_slider_mask == 1
    assert setup.speed_slider_fraction == 0.5
    connection.send.assert_called_once_with(setup)


def test_release_hands_the_slider_back():
    connection = MagicMock()
    setup = _fake_setup()
    release_speed_slider(connection, setup)
    assert setup.speed_slider_mask == 0
    assert setup.speed_slider_fraction == 1.0
    connection.send.assert_called_once_with(setup)


def test_build_speed_slider_connection_configures_the_input_recipe(monkeypatch):
    fake_setup = _fake_setup()
    fake_connection = MagicMock()
    fake_connection.send_input_setup.return_value = fake_setup
    fake_connection.send_start.return_value = True

    fake_rtde_module = SimpleNamespace(RTDE=MagicMock(return_value=fake_connection))
    monkeypatch.setitem(__import__("sys").modules, "rtde.rtde", fake_rtde_module)
    monkeypatch.setitem(__import__("sys").modules, "rtde", SimpleNamespace(rtde=fake_rtde_module))

    connection, setup = build_speed_slider_connection("10.0.0.5", port=30004)

    fake_connection.connect.assert_called_once()
    fake_connection.get_controller_version.assert_called_once()
    names = [name for name, _ in SPEED_SLIDER_RECIPE]
    types = [kind for _, kind in SPEED_SLIDER_RECIPE]
    fake_connection.send_input_setup.assert_called_once_with(names, types)
    fake_connection.send_start.assert_called_once()
    assert connection is fake_connection
    assert setup is fake_setup


def test_build_speed_slider_connection_raises_on_refused_setup(monkeypatch):
    fake_connection = MagicMock()
    fake_connection.send_input_setup.return_value = None

    fake_rtde_module = SimpleNamespace(RTDE=MagicMock(return_value=fake_connection))
    monkeypatch.setitem(__import__("sys").modules, "rtde.rtde", fake_rtde_module)
    monkeypatch.setitem(__import__("sys").modules, "rtde", SimpleNamespace(rtde=fake_rtde_module))

    try:
        build_speed_slider_connection("10.0.0.5")
        raised = False
    except RuntimeError:
        raised = True
    assert raised
