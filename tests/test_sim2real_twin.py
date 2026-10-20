"""S2R_04 P3-6, S2R_08 §2.2: the R0 twin experiment (gate G-2a').

The fast test checks the twin's construction on the synthetic run; the slow
one is the full R0 twin on B's real bags with the A-12' label (seconds per
seed; T-4.6's reduced "2 bags, decimated" version would test something
else, S2R_05). It asserts G-2a' -- (i) identifiable combinations, (ii) the
offsets with d_post >= 0.9, (iii) held-out <= 1.25 x the floor, (iv)-b the
full-set posterior -- on the S2R_06 three seeds; seeds 3 and 5 run in
``scripts/sim2real/twin.py`` (wall time, S2R_08 §2.3).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.path.join(_REPO, "src"))
sys.path.insert(0, os.path.join(_REPO, "tests"))

from elastic_sim.sim2real.config import R0Settings, load_config
from elastic_sim.sim2real.parameters import Registry
from elastic_sim.sim2real.plant import IntermediateBodies, PlantBase, PlantParameters
from elastic_sim.sim2real.real_runs import load_real_run
from elastic_sim.sim2real.replay_r0 import R0Engine, prepare_bags
from elastic_sim.sim2real.twin import TEST_SEEDS, make_twin, perturbed_theta, run_twin_gate

BASE_CONFIG = _REPO / "config" / "sim2real" / "kuka_lbr_iiwa_14_r820_base.yaml"
TWIN_CONFIG = _REPO / "config" / "sim2real" / "iiwa_twin_r0.yaml"
ALL = ("excitation", "identify_excitation", "hold")


@pytest.fixture(scope="module")
def base():
    return PlantBase.from_config(load_config(BASE_CONFIG))


def test_twin_is_self_consistent_and_noisy_at_the_measured_level(tmp_path, base):
    from test_sim2real_real_runs import make_run

    from elastic_sim import identification as idn

    pin, model, data = idn.build_model(base.asset)
    path = make_run(tmp_path, base.asset,
                    lambda q: np.asarray([pin.computeGeneralizedGravity(model, data, r).copy() for r in np.atleast_2d(q)]))
    run = load_real_run(path, segments=ALL, run_id="B", cache_root=tmp_path / "cache")
    registry = Registry.from_config(load_config(BASE_CONFIG), base)
    rng = np.random.default_rng(0)
    theta = perturbed_theta(registry, base, rng, run="B")
    intermediate = IntermediateBodies.from_elastic_build(base)
    twin, injected = make_twin(run, base, theta, R0Settings(), rng, intermediate=intermediate)
    changes = injected["fixed_point_max_change"]
    assert changes[-1] < 1e-10 and all(b < a for a, b in zip(changes, changes[1:]))
    assert [b.bag for b in twin.bags] == [b.bag for b in run.bags]
    # At theta* the replay residual is the injected ft noise (q noise is ~1e-7 here).
    settings = R0Settings(decimate=1, hold_decimate=1)
    engine = R0Engine(base, prepare_bags(twin.bags, settings, time_step=1e-3), intermediate=intermediate)
    residual = np.vstack([r.residual for r in engine.replay(PlantParameters.from_theta(base, theta, runs=("B",)))])
    ratio = np.sqrt(np.mean(residual ** 2, axis=0)) / np.asarray(injected["sigma_ft"])
    assert np.all((ratio > 0.8) & (ratio < 1.3)), ratio


REAL_B = _REPO / "data/real_robot/kuka_lbr_iiwa_14_r820/20261014/iiwa_fast_all_01"


@pytest.mark.slow
@pytest.mark.parametrize("seed", TEST_SEEDS)
def test_gate_g2a_on_real_b(tmp_path, base, seed):
    """S2R_01 T-4.6 / S2R_08 §2.2, R0 part, on all of B's bags."""
    if not (REAL_B / "raw.parquet").is_file():
        pytest.skip("real run B not present")
    child = tmp_path / "twin_fast.yaml"
    child.write_text(yaml.safe_dump({
        "extends": str(TWIN_CONFIG), "output": str(tmp_path / "out"),
        "optimizer": {"method": "least_squares", "budget": 30, "seed": seed}}))
    config = load_config(child)
    real = load_real_run(REAL_B, expected_sha256=config.runs["B"].sha256, segments=ALL, run_id="B")
    _, gate = run_twin_gate(config, run="B", seed=seed, run_id="twin_fast", real=real, base=base, figures=False)
    c = gate["checks"]
    print(f"\nG-2a' seed {seed}: (i) combinations max {100 * max(c['combinations']['per_joint_relative']):.3f} %, "
          f"(ii) offsets max {c['offsets']['max_nm']:.4f} N m over {len(c['offsets']['error_nm'])}, (iii) "
          f"held-out/floor max {max(c['held_out_noise']['ratio']):.3f}, (iv)-b within 2 sd "
          f"{c['posterior']['within_2_sd']:.3f}, beyond 3 sd {len(c['posterior']['beyond_3_sd'])}; calibrated "
          f"{c['posterior']['n_calibrated']} of {c['posterior']['n_parameters']}")
    assert c["combinations"]["pass"] and c["offsets"]["pass"] and c["offsets"]["error_nm"]
    assert c["held_out_noise"]["pass"] and c["held_out_noise"]["factor"] == 1.25
    assert c["posterior"]["pass"] and c["posterior"]["mode"] == "label"
    assert c["posterior"]["n_parameters"] == c["posterior"]["n_calibrated"] + c["posterior"]["n_not_calibrated"]
    assert 0 < c["posterior"]["n_not_calibrated"] and len(c["unobserved_excluded"]["names"]) == 7
    assert gate["pass"]
