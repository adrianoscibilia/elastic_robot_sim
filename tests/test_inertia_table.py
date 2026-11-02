"""``identified_filter/05`` D4: the URDF chain table and the consumer's torch evaluation of it equal pinocchio."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
CONSUMER = ROOT.parent / "dynamic_model_nn" / "urdf_inertia.py"
CONFIGS = ("config/identification/kuka_lbr_iiwa_14_r820_table_drive_round8.yaml",
           "config/identification/ur10_table_round8.yaml")

torch = pytest.importorskip("torch")
pytest.importorskip("pinocchio")


def consumer():
    if not CONSUMER.exists():
        pytest.skip(f"consumer module not found at {CONSUMER}")
    spec = importlib.util.spec_from_file_location("nn_urdf_inertia", CONSUMER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("config", CONFIGS)
def test_torch_chain_inertia_equals_pinocchio_crba(config):
    from diagnose_controller_modes import _load_asset
    from elastic_sim.dataset import load_config
    from elastic_sim.inertia_table import chain_table, crba_reference

    asset = _load_asset(load_config(ROOT / config).asset)
    table = json.loads(json.dumps(chain_table(asset)))          # what a consumer reads back from the JSON file
    model = consumer().ChainInertia(table)
    q = np.random.default_rng(3).uniform(-3.0, 3.0, size=(64, model.n_dof))
    got = model(torch.tensor(q)).numpy()
    reference = crba_reference(asset, q)
    assert np.abs(got - reference).max() < 1e-10 * max(1.0, np.abs(reference).max())
    for row in table["check"]:
        assert np.allclose(model(torch.tensor([row["q"]], dtype=torch.float64))[0].numpy(), row["M"], rtol=0, atol=1e-10)
    batched = model(torch.tensor(q.reshape(8, 8, -1)))          # any leading shape
    assert np.abs(batched.numpy().reshape(64, model.n_dof, model.n_dof) - got).max() == 0.0
