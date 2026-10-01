"""RR_08 synthetic references cannot certify a real recording."""
import json
from pathlib import Path

import pytest

from erd_recording.config import load_lab_config
from erd_recording.contract import differentiation_from_reference, load_reference


@pytest.mark.parametrize('robot,short,rate,dof', [
    ('kuka_lbr_iiwa_14_r820', 'iiwa', 1000, 7), ('ur10_cb3', 'ur10', 125, 6),
])
def test_synthetic_reference_matches_robot(robot, short, rate, dof):
    root = Path(__file__).resolve().parents[1] / 'config'
    config = load_lab_config(root / 'lab' / f'{short}_sim.yaml')
    path = (root / 'lab' / config.consumer.reference_contract).resolve()
    reference = load_reference(path, robot=robot, joint_order=config.joint_order)
    assert reference['synthetic'] is True
    assert reference['n_dof'] == dof
    differentiation = differentiation_from_reference(reference, rate_real=rate, probe_top_real=0,
                                                     reference_path=path)
    assert differentiation['rate'] == rate
    assert differentiation['sg_window'] % 2 == 1
    with pytest.raises(ValueError, match='joint order'):
        load_reference(path, robot=robot, joint_order=tuple(reversed(config.joint_order)))
