#!/usr/bin/env python3
"""identified_filter/01 Sec 6.2-6.3: real-file evaluation of the r8 campaign's final checkpoints (idempotent).

* the old real files (iiwa_real_all_01, iiwa_fast_all_01, ur_A_all_02, ur_B_all_02), zero-shot, with
  ``evaluate_on.py`` (torch): the RMSE per joint and, for the identified classes, K_hat and sigma per bag
  (sigma should be large on these probe-off files);
* every ``test`` entry of ``REFACTOR_SPECS/identified_filter/real_files.yaml`` the same way;
* round 2 (``02_review_round2`` R4): every real ``test`` file is scored from the platform's r8 contract
  burn-in on (``--exclude-head-s``, the scored-rows rule of ``final_results/12``), and on the UR10 the
  q0-taking classes (J4, J5: the r7 configs) also run with the pan shifted by -pi (``12`` Sec 3.2,
  ``input = pan``) beside the raw run;
* every ``modal_reference`` entry: ``tools/modal_fit.py`` (measured M, K, D, f_n, zeta per joint and pose),
  then the models' zero-shot K_hat, D_hat and diag M_phi on the same file beside it.

A result is skipped when its marker (checkpoint x file SHA-256) exists.  Output under
``REFACTOR_SPECS/identified_filter/results/real/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml

SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import r8_campaign  # noqa: E402  (platform paths, contract burn-in, job kinds)

NN = (SIM / ".." / "dynamic_model_nn").resolve()
SPEC = SIM / "REFACTOR_SPECS" / "identified_filter"
OUT = SPEC / "results" / "real"
OLD_FILES = {
    "iiwa_real_all_01": ("iiwa_drive", "data/real_robot/kuka_lbr_iiwa_14_r820/20261014/iiwa_real_all_01/dataset/"
                                       "kuka_lbr_iiwa_14_r820.parquet"),
    "iiwa_fast_all_01": ("iiwa_drive", "data/real_robot/kuka_lbr_iiwa_14_r820/20261014/iiwa_fast_all_01/dataset/"
                                       "kuka_lbr_iiwa_14_r820.parquet"),
    "ur_A_all_02": ("ur10", "data/real_robot/ur10_cb3/20261021/ur_A_all_02/dataset/ur10_cb3.parquet"),
    "ur_B_all_02": ("ur10", "data/real_robot/ur10_cb3/20261021/ur_B_all_02/dataset/ur10_cb3.parquet"),
}
ROBOT_PLATFORM = {"iiwa": "iiwa_drive", "ur10": "ur10"}
ASSET = {"iiwa_drive": "kuka_lbr_iiwa_14_r820_table", "ur10": "ur10_table"}
FINAL_STAGES = ("J", "T", "P", "R")
PAN_SHIFT = {"ur10": (0, -3.141592653589793)}   # 12 Sec 3.2: (joint, rad)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def final_checkpoints(run_id: str) -> list[dict]:
    """Per job, platform and seed: the checkpoint of the last stage that finished."""
    out = []
    for job_dir in sorted((SIM / "runs" / run_id / "jobs").glob("*")):
        for platform_dir in sorted(job_dir.glob("*")):
            seeds = {}
            for done in platform_dir.glob("s*_*.done.json"):
                seed, stage = done.name.split(".")[0].split("_")
                seeds.setdefault(seed, {})[stage] = json.loads(done.read_text())
            for seed, stages in sorted(seeds.items()):
                stage = next((s for s in FINAL_STAGES if s in stages), None)
                if stage is not None:
                    out.append({"job": job_dir.name, "platform": platform_dir.name, "seed": seed, "stage": stage,
                                "checkpoint": stages[stage]["checkpoint_path"]})
    return out


def burn_in(settings: dict, platform: str) -> float:
    """The platform's r8 contract burn-in (0 when the contract is not there yet, with a warning)."""
    try:
        return float(r8_campaign.contract_kwargs(settings, platform)["burn_in_s"])
    except FileNotFoundError:
        print(f"no r8 contract for {platform}: real files scored from t = 0", file=sys.stderr)
        return 0.0


def takes_q0(settings: dict, job: str) -> bool:
    """J4/J5-like jobs: the r7 classes, which read q0 (the identified classes and J11 drop the base angle)."""
    spec = settings["jobs"].get(job) or {}
    return spec.get("kind") in ("r7_residual", "r7_twomass") and \
        not (spec.get("model_kwargs") or {}).get("drop_base_angle", False)


def evaluate(entry: dict, file_id: str, path: Path, dry_run: bool, head_s: float = 0.0,
             shift: tuple | None = None) -> None:
    label = "pan" if shift else "raw"
    marker = OUT / "markers" / (f"{entry['job']}-{entry['platform']}-{entry['seed']}-{file_id}-{label}-"
                                f"{sha256(path)[:12]}.done")
    if marker.exists():
        return
    command = ["uv", "run", "--project", str(NN), "python", str(NN / "evaluate_on.py"), "--checkpoint",
               entry["checkpoint"], "--dataset", str(path), "--split", "all", "--out", str(OUT / f"{file_id}.csv"),
               "--per-bag-out", str(OUT / f"{file_id}.per_bag.csv"), "--exclude-head-s", str(head_s),
               "--input-label", label]
    if shift:
        command += ["--q-shift", f"{shift[0]}={shift[1]}"]
    print(" ".join(command[4:]))
    if dry_run:
        return
    marker.parent.mkdir(parents=True, exist_ok=True)
    code = subprocess.call(command, cwd=NN)
    if code == 0:
        marker.write_text(json.dumps(entry))
    else:
        print(f"FAIL {file_id} {entry}", file=sys.stderr)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", default="r8-campaign")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-old", action="store_true", help="only the files of real_files.yaml")
    parser.add_argument("--config", default=str(r8_campaign.CONFIG), help="the campaign settings")
    args = parser.parse_args(argv)
    settings = r8_campaign.load_settings(Path(args.config))
    OUT.mkdir(parents=True, exist_ok=True)
    checkpoints = final_checkpoints(args.run_id)
    if not checkpoints:
        print(f"no finished checkpoint under runs/{args.run_id}/jobs yet")
    files = [] if args.skip_old else [(file_id, platform, SIM / rel, "test") for file_id, (platform, rel) in OLD_FILES.items()]
    listed = yaml.safe_load((SPEC / "real_files.yaml").read_text(encoding="utf-8")) or {}
    for item in listed.get("files") or []:
        path = SIM / item["path"]
        if not path.exists():
            print(f"missing {path} (real_files.yaml {item['id']})", file=sys.stderr)
            continue
        if item.get("sha256") and not str(item["sha256"]).startswith("<") and sha256(path) != item["sha256"]:
            print(f"SHA-256 mismatch for {item['id']}: listed {item['sha256']}", file=sys.stderr)
            continue
        files.append((item["id"], ROBOT_PLATFORM[item["robot"]], path, item.get("split", "test")))
    for file_id, platform, path, split in files:
        if split == "calibration":
            continue
        if split == "modal_reference":
            fit = OUT / f"modal_fit_{file_id}"
            if not fit.with_suffix(".json").exists() and not args.dry_run:
                subprocess.call(["uv", "run", "--project", str(SIM), "python", str(SIM / "tools/modal_fit.py"),
                                 "--dataset", str(path), "--asset", ASSET[platform], "--out", str(fit)], cwd=SIM)
        head = burn_in(settings, platform)
        for entry in checkpoints:
            if entry["platform"] != platform:
                continue
            evaluate(entry, file_id, path, args.dry_run, head)
            if platform in PAN_SHIFT and takes_q0(settings, entry["job"]):
                evaluate(entry, file_id, path, args.dry_run, head, PAN_SHIFT[platform])
    print(f"results: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
