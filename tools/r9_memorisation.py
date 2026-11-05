#!/usr/bin/env python3
"""identified_filter/11 A0.3: does a residual memorise its training robots?  (dynamic_model_nn's environment)

Retrains one campaign stage (its own chain, unchanged but for the epochs and the output folder) and keeps every
epoch's weights; then scores each epoch on 10 train robots and on every val robot (all their bags, past the burn-in).
A train/val gap that opens with the epochs, train falling while val rises, is memorisation.

    uv run --project ../dynamic_model_nn python tools/r9_memorisation.py \\
        --chain runs/r9-campaign/chains/J11/ur10/s0_R.yaml --epochs 24 --out runs/r9-campaign/a0/J11-ur10
    uv run --project ../dynamic_model_nn python tools/r9_memorisation.py --score-only --out runs/r9-campaign/a0/J11-ur10

Writes ``<out>/epochs/epoch_NNN.pt`` and ``<out>/memorisation.csv`` (epoch, split, rmse, robots, rows).
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

SIM = Path(__file__).resolve().parents[1]
NN = (SIM / ".." / "dynamic_model_nn").resolve()
sys.path.insert(0, str(NN))

TRAIN_ROBOTS = 10


def train(chain: Path, epochs: int, out: Path) -> Path:
    """The chain's stage with ``epochs`` epochs, every epoch's weights under ``out/epochs``; returns the checkpoint."""
    import model_workflow
    import train_chain_from_config

    config = yaml.safe_load(chain.read_text(encoding="utf-8"))
    run = config["runs"][0]
    run["output_model_path"] = str(out / Path(run["output_model_path"]).name)
    run["training_overrides"].update(epochs=int(epochs), early_stopping_patience=10 * int(epochs), resume=True)
    config["defaults"]["summary_dir"] = str(out / "log")
    local = out / "chain.yaml"
    local.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (out / "epochs").mkdir(parents=True, exist_ok=True)
    save = torch.save

    def saving(obj, path, *args, **kwargs):
        # The per-epoch progress file (resume) carries the epoch's weights: keep a copy of each.
        if str(path).endswith("_progress.pt") and isinstance(obj, dict) and "model_state_dict" in obj:
            save(copy.deepcopy(obj["model_state_dict"]), out / "epochs" / f"epoch_{int(obj['epoch']):03d}.pt")
        return save(obj, path, *args, **kwargs)

    model_workflow.torch.save = saving
    seed = int(Path(chain).stem.split("_")[0][1:])
    sys.argv = ["train_chain_from_config.py", "--config", str(local), "--seeds", str(seed),
                "--summary-out", str(out / "summary.csv")]
    train_chain_from_config.main()
    return Path(run["output_model_path"]).with_name(Path(run["output_model_path"]).stem + "_checkpoint.pt")


def score(checkpoint: Path, dataset_path: Path, out: Path) -> None:
    import pandas as pd

    from dataset import CustomDataset, head_mask
    from evaluate_on import load_checkpoint
    from utils import unpack_model_output

    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    normalization = blob["normalization"]
    dataset = CustomDataset(csv_path=str(dataset_path), normalization=normalization.get("scheme", "train"),
                            normalize=True, normalize_target=bool(normalization.get("normalize_target", True)))
    dataset.apply_normalization_state(normalization)
    model, _, record = load_checkpoint(checkpoint, None, None, dataset.dof)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    burn_in = float(json_contract(dataset_path)["transmission_priors"]["burn_in_s"])
    labels, tiers = np.asarray(dataset.split_labels), np.asarray(dataset.tier)
    train_robots = sorted({t for t, s in zip(tiers, labels) if s == "train" and t != "rigid"})[:TRAIN_ROBOTS]
    groups = {"train10": lambda i: tiers[i] in train_robots,
              "val": lambda i: labels[i] == "val" and tiers[i] != "rigid"}
    rows = {name: [] for name in groups}
    for indices in dataset.bag_to_indices.values():
        indices = np.asarray(indices)
        for name, member in groups.items():
            if member(indices[0]):
                rows[name].append(indices[head_mask(dataset.time[indices], burn_in)])
    rows = {name: np.concatenate(parts) for name, parts in rows.items()}
    table = []
    for path in sorted((out / "epochs").glob("epoch_*.pt")):
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
        model.to(device).eval()
        for name, index in rows.items():
            total, count = 0.0, 0
            with torch.no_grad():
                for start in range(0, len(index), 8192):
                    chunk = torch.as_tensor(index[start:start + 8192], dtype=torch.long)
                    prediction, _ = unpack_model_output(model(dataset.q[chunk].to(device), dataset.dq[chunk].to(device),
                                                              dataset.ddq[chunk].to(device),
                                                              dataset.tau_cmd[chunk].to(device)))
                    prediction = dataset.denormalize(prediction.cpu(), "F", bag_ids=None).double()
                    target = dataset.denormalize(dataset.F[chunk], "F", bag_ids=None).double()
                    total += float(((prediction - target) ** 2).sum())
                    count += target.numel()
            table.append({"epoch": int(path.stem.split("_")[1]), "split": name, "rmse": (total / count) ** 0.5,
                          "robots": len(train_robots) if name == "train10" else int(len({tiers[i] for i in index})),
                          "rows": int(len(index)), "class_path": record["class_path"]})
            print(table[-1], flush=True)
    pd.DataFrame(table).to_csv(out / "memorisation.csv", index=False)


def json_contract(dataset_path: Path) -> dict:
    import json

    return json.loads(Path(dataset_path).with_suffix(".contract.json").read_text(encoding="utf-8"))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chain", type=Path, help="a campaign chain YAML (runs/<run>/chains/<job>/<platform>/sN_X.yaml)")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--score-only", action="store_true", help="score the saved epochs again")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.score_only:
        config = yaml.safe_load((args.out / "chain.yaml").read_text(encoding="utf-8"))
        run = config["runs"][0]
        checkpoint = Path(run["output_model_path"]).with_name(Path(run["output_model_path"]).stem + "_checkpoint.pt")
    else:
        checkpoint = train(args.chain, args.epochs, args.out)
        config = yaml.safe_load((args.out / "chain.yaml").read_text(encoding="utf-8"))
    score(checkpoint, Path(config["defaults"]["dataset_file"]), args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
