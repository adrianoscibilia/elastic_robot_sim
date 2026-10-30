#!/usr/bin/env python3
"""FG1 helper: the LNN training pipeline's own row counts and target scale.

Runs under ``dynamic_model_nn``'s interpreter and imports its training code
(``dataset.CustomDataset``, ``bag_windows``, ``head_mask``,
``sequence_models.SequenceSpecMixin``), so the counts are the pipeline's, not
a re-derivation.  ``scripts/zollo_fair_fit.py`` calls it and compares; it
prints one JSON object.

Example
-------
::

    ../dynamic_model_nn/.venv/bin/python scripts/lnn_row_counts.py --repo ../dynamic_model_nn \\
        --dataset data/identification/r7-nopayload/ur10/ur10_table_round7_nopayload_val4.parquet \\
        --time-step 0.002 --burn-in 2.7
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", required=True, help="the dynamic_model_nn checkout")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--time-step", type=float, required=True, help="the two-mass classes' time_step")
    parser.add_argument("--burn-in", type=float, required=True, help="the two-mass classes' burn_in_s")
    args = parser.parse_args()
    dataset_path = os.path.abspath(args.dataset)
    sys.path.insert(0, os.path.abspath(args.repo))
    import numpy as np
    from dataset import CustomDataset, _split_bags, bag_windows, head_mask
    from sequence_models import SequenceSpecMixin

    spec_holder = SequenceSpecMixin()
    spec_holder._init_sequence(args.time_step, args.burn_in)
    spec = spec_holder.sequence_spec()
    with contextlib.redirect_stdout(sys.stderr):
        data = CustomDataset(csv_path=dataset_path, normalize_target=False)
    labels = np.asarray(data.split_labels)
    windows = bag_windows(data, "train", spec["window_s"], spec["burn_in_s"], spec["stride_s"])
    loss_rows = sum(int(head_mask(data.time[w], spec["burn_in_s"]).sum()) for w in windows)
    val_loss = sum(int(head_mask(data.time[b], spec["burn_in_s"]).sum()) for b in _split_bags(data, "val"))
    print(json.dumps({
        "train_rows": int((labels == "train").sum()), "val_rows": int((labels == "val").sum()),
        "train_windows": len(windows), "train_window_loss_rows": loss_rows, "val_bag_loss_rows": val_loss,
        "window_length": sorted({len(w) for w in windows}), "spec": spec, "dt": float(data.dt),
        "target_scale": [float(v) for v in data.target_scale], "sg_window": int(data.sg_window),
    }))


if __name__ == "__main__":
    main()
