#!/usr/bin/env python3
"""Enlarge a production file's validation set by relabelling train robots (FR_03 T-3).

No new data: the copy differs from the source only in the ``split`` value of
the moved train tiers' rows.  Test robots and test rows are unchanged, so a
checkpoint's ``evaluate_on --split test`` is identical on both files.

Rule (deterministic).  Rank the train *elastic* tiers (never ``rigid``) by the
geometric mean over joints of their stiffness ratio to nominal, softest
first; move the tiers at evenly spaced ranks: positions i / (m + 1) of the
ranking, i = 1..m, for m moved tiers -- 1/3 and 2/3 for two, the median for
one.  Soft and stiff transmissions both land in val.

Output, under ``<out-root>/<variant>/``:

* ``<stem>_val<k>.parquet``: the copy (k = the new number of val robots);
* its ``.contract.json`` (split block updated, train/val baselines
  recomputed with the producer's per-bag diagnostics; test and all kept
  verbatim) and ``.manifest.json`` (split block and ``records[].split``);
* ``<stem>_val<k>.relabel.json``: source sha256, moved tiers, the rule, the
  new split lists.

Examples
--------
    # the plan only, nothing written
    python scripts/relabel_val.py --dataset data/identification/r6-full-20260929-0049/fmrr/fmrr_tecnobody_round6.parquet \\
        --variant fmrr --dry-run

    python scripts/relabel_val.py --dataset .../ur10_table_round6.parquet --variant ur10 --val-robots 4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import warnings
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(_REPO / "src"))
sys.path.insert(0, os.fspath(_REPO / "scripts"))

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_OUT_ROOT = "data/identification/r6-retrain"
RULE = ("rank the train elastic tiers (rigid excluded) by the geometric mean over joints of stiffness / "
        "stiffness_nominal, softest first; move the tiers at ranking positions i/(m+1), i = 1..m "
        "(index round(i/(m+1) * (n-1)) of n ranked tiers)")


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def default_val_robots(n_train_tiers: int) -> int:
    """FR_03 T-3 sizes: the arms (15 train tiers) go to 4 val robots, FMRR (8) to 3."""
    return 3 if n_train_tiers <= 8 else 4


def stiffness_ranking(manifest: dict, tiers: list[str]) -> list[tuple[str, float]]:
    """``(tier, geometric-mean stiffness ratio to nominal)``, softest first."""
    nominal = np.asarray(manifest["transmission_sampling"]["stiffness_nominal"], dtype=float)
    robots = {robot["name"]: robot for robot in manifest["robots"]}
    ranked = []
    for tier in tiers:
        stiffness = np.asarray(robots[tier]["stiffness"], dtype=float)
        ranked.append((tier, float(np.exp(np.mean(np.log(stiffness / nominal))))))
    return sorted(ranked, key=lambda item: (item[1], item[0]))


def pick_moved(ranked: list[tuple[str, float]], moves: int) -> list[int]:
    """Indices into ``ranked`` at evenly spaced ranks (1/3, 2/3 for 2; the median for 1)."""
    n = len(ranked)
    if moves <= 0:
        return []
    if moves >= n:
        raise ValueError(f"cannot move {moves} of {n} train elastic tiers to val")
    picks = [int(math.floor(i / (moves + 1) * (n - 1) + 0.5)) for i in range(1, moves + 1)]
    if len(set(picks)) != len(picks):
        raise ValueError(f"evenly spaced ranks collide for {moves} of {n} tiers: {picks}")
    return picks


def plan(dataset: Path, val_robots: int | None) -> dict:
    contract = json.loads(dataset.with_suffix(".contract.json").read_text(encoding="utf-8"))
    manifest = json.loads(dataset.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    split = contract["split"]
    train, val, test = list(split["train"]), list(split["val"]), list(split["test"])
    if manifest.get("split", {}).get("train") != train:
        raise ValueError(f"{dataset}: contract and manifest disagree on the train split")
    target = val_robots if val_robots is not None else default_val_robots(len(train))
    moves = target - len(val)
    if moves < 0:
        raise ValueError(f"{dataset} already has {len(val)} val robots (> {target})")
    elastic = [tier for tier in train if tier != "rigid"]
    ranked = stiffness_ranking(manifest, elastic)
    picks = pick_moved(ranked, moves)
    moved = [ranked[i][0] for i in picks]
    return {
        "contract": contract, "manifest": manifest,
        "ranking": [{"rank": i, "tier": t, "stiffness_ratio_geomean": r} for i, (t, r) in enumerate(ranked)],
        "moved": [{"tier": ranked[i][0], "rank": i, "of": len(ranked),
                   "stiffness_ratio_geomean": ranked[i][1]} for i in picks],
        "split": {"train": [t for t in train if t not in moved], "val": val + moved, "test": test},
        "source_split": {"train": train, "val": val, "test": test},
        "val_robots": target,
    }


def relabel_table(table: pa.Table, moved: set[str]) -> tuple[pa.Table, int]:
    """The same table with ``split = "val"`` on the moved tiers' (train) rows; nothing else touched."""
    tiers = table.column("tier").to_pylist()
    old = table.column("split")
    labels = old.to_pylist()
    changed = 0
    for row, tier in enumerate(tiers):
        if tier in moved:
            if labels[row] != "train":
                raise ValueError(f"row {row} of tier {tier} is {labels[row]!r}, expected train")
            labels[row] = "val"
            changed += 1
    new = pa.array(labels, type=pa.string()).cast(old.type)
    return table.set_column(table.schema.get_field_index("split"), table.schema.field("split"), new), changed


def recompute_baselines(output: Path, manifest: dict, splits: dict) -> dict:
    """Per-bag diagnostics of the train and val bags; their medians for the new splits.

    Bag diagnostics do not depend on the split, so the medians are what the
    producer would have written for this labelling.  Also recomputes the
    *source* train median as a check against the source contract.
    """
    import pandas as pd

    from diagnose_controller_modes import _CONTRACT_BASELINES, _load_asset
    from elastic_sim.diagnostics import dataset_diagnostics

    frame = pd.read_parquet(output)
    bag_split = frame.groupby("bag", sort=False)["split"].first()
    bag_tier = frame.groupby("bag", sort=False)["tier"].first()
    bags = [b for b in bag_split.index if bag_split[b] in ("train", "val")]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        table = dataset_diagnostics(frame, manifest, _load_asset(manifest["asset"]), bags=bags).set_index("bag")

    def entry(names: list[str]) -> dict:
        subset = table.loc[[b for b in names if b in table.index]]
        out = {"n_bags": int(len(subset))}
        for column in _CONTRACT_BASELINES:
            if column in subset.columns:
                value = float(np.nanmedian(subset[column].to_numpy(dtype=float))) if len(subset) else float("nan")
                out[column] = value if np.isfinite(value) else None
        return out

    new = {split: entry([b for b in bags if bag_split[b] == split]) for split in ("train", "val")}
    source_train = entry([b for b in bags if bag_tier[b] in set(splits["source_train"])])
    return {"splits": new, "source_train_recomputed": source_train}


def write(dataset: Path, variant: str, out_root: Path, the_plan: dict, *, baselines: bool) -> Path:
    stem = dataset.stem
    k = the_plan["val_robots"]
    out_dir = out_root / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"{stem}_val{k}.parquet"
    moved = {m["tier"] for m in the_plan["moved"]}

    source = pq.read_table(dataset)
    table, changed = relabel_table(source, moved)
    pq.write_table(table, output, compression="snappy")

    split_block = {**the_plan["contract"]["split"], **the_plan["split"],
                   "relabelled": {"from": the_plan["source_split"]["val"], "moved_to_val": sorted(moved),
                                  "sidecar": f"{output.stem}.relabel.json"}}
    contract = dict(the_plan["contract"])
    contract["split"] = split_block
    manifest = dict(the_plan["manifest"])
    manifest["split"] = {**manifest["split"], **the_plan["split"], "relabelled": split_block["relabelled"]}
    manifest["records"] = [{**r, "split": "val"} if r.get("tier") in moved else r for r in manifest["records"]]
    manifest["relabelled_from"] = str(dataset)

    check = None
    if baselines:
        result = recompute_baselines(output, the_plan["manifest"],
                                     {"source_train": the_plan["source_split"]["train"]})
        old = (contract.get("baselines") or {}).get("splits") or {}
        # Test and all are the same bags as in the source: kept verbatim.
        contract["baselines"] = {**contract["baselines"], "splits": {**old, **result["splits"]},
                                 "relabelled": "train/val recomputed from per-bag diagnostics; test and all verbatim"}
        source_train = old.get("train") or {}
        check = {key: {"source": source_train.get(key), "recomputed": value}
                 for key, value in result["source_train_recomputed"].items() if key in source_train}
    else:
        stale = "not recomputed after relabelling (relabel_val.py --no-baselines)"
        splits = dict((contract.get("baselines") or {}).get("splits") or {})
        for name in ("train", "val"):
            splits.pop(name, None)
        contract["baselines"] = {**(contract.get("baselines") or {}), "splits": splits, "relabelled": stale}

    output.with_suffix(".contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    sidecar = {
        "tool": "elastic_robot_sim/scripts/relabel_val.py (FR_03 T-3)",
        "source": str(dataset), "source_sha256": sha256(dataset),
        "output": str(output), "output_sha256": sha256(output),
        "rule": RULE, "val_robots": k,
        "moved": the_plan["moved"], "ranking": the_plan["ranking"],
        "rows_relabelled": changed,
        "split_before": the_plan["source_split"], "split_after": the_plan["split"],
        "test_unchanged": the_plan["source_split"]["test"] == the_plan["split"]["test"],
        "baselines_check_source_train": check,
    }
    output.with_suffix(".relabel.json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="production parquet (its .contract/.manifest.json beside it)")
    parser.add_argument("--variant", required=True, help="output sub-folder, e.g. iiwa_drive, fmrr")
    parser.add_argument("--val-robots", type=int, default=None,
                        help="val robots after relabelling (default: 4, or 3 with <= 8 train tiers)")
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--no-baselines", action="store_true",
                        help="skip recomputing the train/val baselines (drops them from the contract)")
    parser.add_argument("--dry-run", action="store_true", help="print the plan, write nothing")
    args = parser.parse_args(argv)

    dataset = Path(args.dataset).expanduser().resolve()
    the_plan = plan(dataset, args.val_robots)
    print(f"{dataset.name}: val {the_plan['source_split']['val']} -> {the_plan['split']['val']} "
          f"({the_plan['val_robots']} robots); train {len(the_plan['source_split']['train'])} -> "
          f"{len(the_plan['split']['train'])}; test unchanged {the_plan['split']['test']}")
    for item in the_plan["ranking"]:
        mark = " <- val" if item["tier"] in {m["tier"] for m in the_plan["moved"]} else ""
        print(f"  rank {item['rank']:2d}  {item['tier']:6s}  K/K_nom (geo. mean) {item['stiffness_ratio_geomean']:.3f}{mark}")
    if args.dry_run:
        print("dry run: nothing written")
        return 0
    out_root = Path(args.out_root)
    out_root = out_root if out_root.is_absolute() else _REPO / out_root
    output = write(dataset, args.variant, out_root, the_plan, baselines=not args.no_baselines)
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
