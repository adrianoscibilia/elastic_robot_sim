"""Shared helpers for the paper_v2 measurements and figures (A2L_005).

Run with the dynamic_model_nn interpreter (torch):
    ~/Projects/dynamic_model_nn/.venv/bin/python scripts/paper/<script>.py

The rollout is ``dynamic_model_nn/evaluate_on.evaluate`` with every per-row
array kept: the dataset is built exactly as there (checkpoint's own
normalisation, SG ddq from the contract), each bag the split touches is
rolled out whole from its first sample, and the scored rows are the split's
rows with ``t - t_bag_start >= burn-in``.  The r7-taufree checkpoints have
``normalize_target = False``, so every output is in N·m as returned.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ERS = Path(os.environ.get("ERS_ROOT", Path.home() / "Projects" / "elastic_robot_sim"))
DMN = Path(os.environ.get("DMN_ROOT", Path.home() / "Projects" / "dynamic_model_nn"))
PAPER = Path(os.environ.get("PAPER_ROOT", Path.home() / "Projects" / "lnn_paper"))
FIG_DIR = PAPER / "figures" / "v2"
DATA_OUT = ERS / "REFACTOR_SPECS" / "final_results" / "data"
REAL_EVAL = ERS / "REFACTOR_SPECS" / "final_results" / "tools" / "real_eval"

R7 = ERS / "data" / "identification" / "r7-nopayload"
FILES = {
    ("iiwa_drive", "test"): R7 / "iiwa_drive" / "kuka_lbr_iiwa_14_r820_table_drive_round7_nopayload.parquet",
    ("iiwa_drive", "gainshift"): R7 / "iiwa_drive" / "kuka_lbr_iiwa_14_r820_table_drive_round7_nopayload_gainshift.parquet",
    ("ur10", "test"): R7 / "ur10" / "ur10_table_round7_nopayload.parquet",
    ("ur10", "gainshift"): R7 / "ur10" / "ur10_table_round7_nopayload_gainshift.parquet",
}
# final_results/03 §8
SHA256 = {
    ("iiwa_drive", "test"): "5e6d5f8fe5a47f55d75b29749184f8f4b8f31be403e7177c1971f14043a90d62",
    ("iiwa_drive", "gainshift"): "f569c97efac7475af863dd97852b759ea275f3d49bbb6de14d8d59b052450ca6",
    ("ur10", "test"): "365daa01ccb836ca232593659eeecec458d638bacad33516276e289b33dc23a7",
    ("ur10", "gainshift"): "b9f4c46608429312309c19efa284da6c24b394ec14c5390d9ac1257100a704df",
}
BURN_IN = {"iiwa_drive": 1.65, "ur10": 2.70}
VARIANTS = ["iiwa_drive", "ur10"]
CLASSES = ["lnn_tau_res", "kalnn_tau_res", "lnn_twomass", "kalnn_twomass"]
JOINT_LABELS = {
    "iiwa_drive": ["A1", "A2", "A3", "A4", "A5", "A6", "A7"],
    "ur10": ["pan", "lift", "elbow", "w1", "w2", "w3"],
}
JOINT_NAMES = {
    "iiwa_drive": [f"iiwa_A{i}" for i in range(1, 8)],
    "ur10": ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
             "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
}
# Paper colours (A2L_005 rules)
C_MEAS, C_RIGID, C_RES, C_2M = "black", "0.55", "#E8692F", "#2A78D6"

if str(DMN) not in sys.path:
    sys.path.insert(0, str(DMN))


def deployed_checkpoints() -> pd.DataFrame:
    """``variant, model, seed, checkpoint, test_rmse`` of the deployed (best-validation) seed per class."""
    s = pd.read_csv(DATA_OUT / "model_comparison_taufree_summary.csv")
    s = s[s.variant.isin(VARIANTS) & s.model.isin(CLASSES)]
    return s[["variant", "model", "best_val_seed", "best_val_seed_checkpoint", "best_val_seed_test_rmse"]].rename(
        columns={"best_val_seed": "seed", "best_val_seed_checkpoint": "checkpoint",
                 "best_val_seed_test_rmse": "test_rmse"}).reset_index(drop=True)


def sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_dataset(checkpoint: Path, dataset_path: Path):
    """``CustomDataset`` built as ``evaluate_on.evaluate`` builds it for this checkpoint."""
    import torch
    from dataset import CustomDataset
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    normalization = blob.get("normalization")
    scheme = (normalization or {}).get("scheme", "per_bag")
    dataset = CustomDataset(
        csv_path=str(dataset_path), backend=None, normalization=scheme if scheme != "none" else "train",
        normalize=scheme != "none", normalize_target=bool((normalization or {}).get("normalize_target", True)))
    if scheme != "none":
        dataset.apply_normalization_state(normalization)
    return dataset, normalization


def rollout(checkpoint: Path, dataset_path: Path, split: str = "test", burn_in: float = 0.0, dataset=None):
    """Roll a checkpoint over every bag of ``split``; arrays over all rows of the file (NaN outside the split).

    Returns ``dict(pred, rig, res, bypass, delta, target, rows, keep, bag, t, model, dataset)``:
    ``pred`` the model output (``tau_s`` for a two-mass class), ``rig``/``res`` the
    rigid torque and residual, ``bypass = rig + res`` (the filter-bypassed
    two-mass prediction; equal to ``pred`` for a residual class), ``delta``
    the two-mass latent deflection, ``rows`` the split's row indices and
    ``keep`` the scored mask over ``rows``.
    """
    import torch
    import evaluate_on
    from utils import unpack_model_output
    checkpoint = Path(checkpoint)
    if dataset is None:
        dataset, normalization = load_dataset(checkpoint, dataset_path)
    else:
        normalization = torch.load(checkpoint, map_location="cpu", weights_only=False).get("normalization")
    assert not (normalization or {}).get("normalize_target", True), "expected physical-unit outputs"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _ = evaluate_on.load_checkpoint(checkpoint, None, None, dataset.dof)
    model.to(device).eval()
    rows = (np.arange(len(dataset)) if split == "all"
            else np.flatnonzero(np.asarray(dataset.split_labels) == split))
    N, n = len(dataset), dataset.dof
    out = {k: np.full((N, n), np.nan) for k in ("pred", "rig", "res", "delta")}
    two_mass = hasattr(model, "latent_deflection")
    with torch.no_grad():
        for indices, _ in evaluate_on._bag_spans(dataset, rows):
            index = torch.as_tensor(indices, dtype=torch.long)
            q, dq, ddq = (dataset.q[index].to(device), dataset.dq[index].to(device), dataset.ddq[index].to(device))
            pred, res = unpack_model_output(model(q, dq, ddq, None))
            out["pred"][indices] = pred.cpu().double().numpy()
            out["res"][indices] = res.cpu().double().numpy()
            if two_mass:
                out["delta"][indices] = model.latent_deflection()[0].cpu().double().numpy()
                rig, _, res2 = model._rigid_and_residual(q.unsqueeze(0), dq.unsqueeze(0), ddq.unsqueeze(0))
                out["rig"][indices] = rig[0].cpu().double().numpy()
                assert np.allclose(res2[0].cpu().double().numpy(), out["res"][indices], atol=1e-5)
            else:
                out["rig"][indices] = out["pred"][indices] - out["res"][indices]
    out["bypass"] = out["rig"] + out["res"]
    out["target"] = dataset.F.double().numpy()
    out["rows"] = rows
    out["keep"] = evaluate_on._kept_rows(dataset, rows, burn_in)
    out["bag"] = np.asarray(dataset.bag_id)
    out["t"] = np.asarray(dataset.time, dtype=float)
    out["model"] = model
    out["dataset"] = dataset
    return out


def scored(o, key):
    """``o[key]`` on the scored rows."""
    return o[key][o["rows"]][o["keep"]]


def rmse(e, axis=None):
    return np.sqrt(np.mean(np.asarray(e) ** 2, axis=axis))


def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.6, "lines.linewidth": 1.0,
        "savefig.dpi": 300,
    })
    return plt
