"""RR_14 P-1: the contract's ``source`` follows the lab file's ``hardware``,
and ``evaluate`` refuses to label a non-real file as real validation."""

from __future__ import annotations

import pytest

from erd_recording.contract import contract_source, real_contract
from erd_recording.evaluate_cli import dataset_label


def _contract(hardware: str) -> dict:
    return real_contract(
        {"schema_version": 2}, hardware=hardware, n_dof=1, target_instrument="t", target_semantics="s",
        q_side="motor", dq_side="motor", dq_source="position_derivative", tau_instrument="tau",
        controller={}, differentiation={"sg_window": 5, "sg_poly": 3},
        baselines={"statistic": "fixture", "splits": {}}, real_block={"robot_id": "r"})


@pytest.mark.parametrize("hardware, source", [("mock", "synthetic_mock"), ("emulator", "synthetic_mock"),
                                              ("ursim", "synthetic_mock"), ("real", "real")])
def test_source_per_hardware_value(hardware, source):
    assert contract_source(hardware) == source
    contract = _contract(hardware)
    assert contract["source"] == source
    assert contract["real"]["hardware"] == hardware
    assert contract["real"]["robot_id"] == "r"


def test_unknown_hardware_is_refused():
    with pytest.raises(ValueError):
        contract_source("gazebo")


def test_evaluate_labels_a_synthetic_file_as_synthetic():
    label = dataset_label(_contract("emulator"), {"status": "synthetic", "hardware": "emulator"})
    assert label == {"dataset_source": "synthetic_mock", "dataset_hardware": "emulator",
                     "dataset_status": "synthetic"}


def test_evaluate_refuses_an_old_contract_that_says_real_for_emulator_data():
    old = _contract("real")
    old["real"].pop("hardware")  # an RR_13-era contract: source real, no hardware key
    with pytest.raises(ValueError, match="P-1"):
        dataset_label(old, {"status": "synthetic", "hardware": "emulator"})


def test_evaluate_refuses_real_validation_on_a_non_real_file():
    with pytest.raises(ValueError):
        dataset_label(_contract("ursim"), {"status": "valid", "hardware": "ursim"})
    with pytest.raises(ValueError):
        dataset_label(_contract("mock"), {"status": "synthetic", "hardware": "real"})


def test_evaluate_accepts_real_data():
    label = dataset_label(_contract("real"), {"status": "valid", "hardware": "real"})
    assert label["dataset_source"] == "real" and label["dataset_status"] == "valid"
