#!/usr/bin/env python3
"""Build one platform's round-6 datasets and hold them to the gates (`R6_00` Secs 7, 9).

From one ``schema_version: 3`` config, in one process so every variant reuses
the same optimized trajectories:

* ``production``    -> ``<stem>.parquet``: the config as written (train/val/test,
  holdout robots);
* ``gainshift``     -> ``<stem>_gainshift.parquet``: the test robots only, same
  trajectories and seeds, with omega drawn outside the training band
  (``gates.gainshift_fraction`` of omega_max, Sec 7 item 4);
* ``ablation_off``  -> ``<stem>_ablation_off.parquet``: every Sec 5 effect off
  (link friction, nonlinear spring, transmission error / ripple, motor
  friction);
* ``ablation_clean``-> ``<stem>_ablation_clean.parquet``: the noise block off and
  the clean target (Sec 6's noise ablation).

Every file gets the Sec 9 hard checks; the production file also the Sec 7
gates, and the gain-shift file gate 4.  The report goes to
``<stem>.round6_checks.json``.  A file that fails is written as
``<stem>....refused.parquet`` instead, so a consumer selecting by exact name
never picks it up, and the script exits 3.

Examples
--------
    # production + gain shift, full size
    python scripts/build_round6.py --config config/identification/ur10_table_round6.yaml

    # a preflight-sized iiwa build with both ablations, nothing written
    python scripts/build_round6.py --config config/identification/kuka_lbr_iiwa_14_r820_table_round6_drive.yaml \\
        --robots 2 --trajectories 1 --ablations --no-save
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(_REPO / "src"))
# One BLAS/OpenMP thread per worker: the build already runs one process per
# core, and threaded BLAS on top drove the load past 400 on 32 cores.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
sys.path.insert(0, os.fspath(_REPO / "scripts"))

import numpy as np

from diagnose_controller_modes import _load_asset, _shrink, add_baselines_to_contract
from elastic_sim.dataset import PlantExtrasSampling, assign_splits, default_jobs, generate, load_config, write_dataset
from elastic_sim.diagnostics import dataset_diagnostics
from elastic_sim.round6_checks import check_dataset, gate_gainshift

VARIANTS = ("production", "gainshift", "ablation_off", "ablation_clean")
SUFFIX = {"production": "", "gainshift": "_gainshift", "ablation_off": "_ablation_off",
          "ablation_clean": "_ablation_clean"}


def variant_config(config, variant: str):
    """The config of one variant, derived from the production one."""
    if variant == "production":
        return config
    if variant == "gainshift":
        labels = assign_splits(config.tiers, config.split)
        test = tuple(sorted(name for name, label in labels.items() if label == "test"))
        if not test:
            raise ValueError("the gain-shift file needs test robots (dataset.split.test_robots)")
        return replace(config, only_tiers=test,
                       control_gains=config.control_gains.with_fraction_bands(config.gates.gainshift_fraction))
    if variant == "ablation_off":
        return replace(config, plant_extras=PlantExtrasSampling(),
                       motor_friction=replace(config.motor_friction, viscous_fraction=0.0, coulomb_fraction=0.0))
    if variant == "ablation_clean":
        return replace(config, noise=replace(config.noise, enabled=False),
                       signals=replace(config.signals, target_source="clean"))
    raise ValueError(f"unknown variant {variant!r}")


def build_variant(config, asset, variant: str, *, out_dir: Path, stem: str, jobs: int, verbose: bool,
                  save: bool, trajectory_cache: dict, production_manifest: dict | None = None) -> dict:
    variant_cfg = variant_config(config, variant)
    frame, manifest, comparison = generate(variant_cfg, asset, verbose=verbose, jobs=jobs,
                                           trajectory_cache=trajectory_cache)
    diagnostics = dataset_diagnostics(frame, manifest, asset)
    report = check_dataset(frame, manifest, diagnostics=diagnostics, production=variant == "production")
    if variant == "gainshift" and production_manifest is not None:
        report["gates"].append(gate_gainshift(production_manifest, manifest))
        report["ok"] = bool(report["ok"] and report["gates"][-1]["ok"]
                            and next(g for g in report["gates"] if g["gate"].startswith("1"))["ok"])
    report["variant"] = variant
    report["bags"] = int(manifest["n_bags"])
    report["qc_v2_median"] = {
        kind: float(np.nanmedian(diagnostics.loc[mask, "tau_unexplained_fraction"]))
        for kind, mask in (("elastic", ~diagnostics["tier"].eq("rigid")), ("rigid", diagnostics["tier"].eq("rigid")))
        if mask.any()
    }
    name = f"{stem}{SUFFIX[variant]}" + ("" if report["ok"] else ".refused")
    report["file"] = None
    if save:
        output = out_dir / f"{name}.parquet"
        write_dataset(frame, manifest, output, comparison, metadata_columns=variant_cfg.metadata_columns)
        add_baselines_to_contract(output, frame, diagnostics)
        report["file"] = str(output)
    return {"report": report, "manifest": manifest}


def _print_report(report: dict) -> None:
    status = "OK" if report["ok"] else "REFUSED"
    print(f"\n--- {report['variant']}: {status} ({report['bags']} bags) {report['file'] or '(not written)'}")
    for check in report["hard_checks"]:
        print(f"  [{'x' if check['ok'] else ' '}] {check['check']}")
    for gate in report["gates"]:
        print(f"  [{'x' if gate['ok'] else ' '}] gate {gate['gate']}"
              + (f": worst {gate['worst_fraction']:.2%} ({gate['worst_bag']}, {gate['worst_joint']})"
                 if "worst_fraction" in gate else "")
              + (f": {gate['per_tier']}" if "per_tier" in gate else ""))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="a schema_version: 3 identification config")
    parser.add_argument("--variants", nargs="+", default=["production", "gainshift"], choices=VARIANTS)
    parser.add_argument("--ablations", action="store_true", help="also build ablation_off and ablation_clean (iiwa)")
    parser.add_argument("--robots", type=int, default=None, help="shrink to this many robots (preflight)")
    parser.add_argument("--trajectories", type=int, default=None, help="shrink to this many trajectories")
    parser.add_argument("--jobs", type=int, default=None, help="worker processes (default nproc - 1)")
    parser.add_argument("--out-dir", default=None, help="default: the config's output directory")
    parser.add_argument("--no-save", action="store_true", help="build and check, write nothing")
    parser.add_argument("--refusal-ok", action="store_true",
                        help="exit 0 even when a file is refused (the launcher lists refusals, R6_04 Sec 4 step 8)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if not config.is_round6:
        parser.error("build_round6.py builds schema_version: 3 configs only")
    asset = _load_asset(config.asset)
    if args.robots is not None or args.trajectories is not None:
        config = _shrink(config, trajectories=args.trajectories or config.n_trajectories,
                         robots=args.robots if args.robots is not None else config.transmission.robots,
                         backends=tuple(config.backends))
    variants = list(dict.fromkeys(args.variants + (["ablation_off", "ablation_clean"] if args.ablations else [])))
    if "gainshift" in variants and "production" not in variants:
        parser.error("the gain-shift file is checked against the production file; build both")
    jobs = default_jobs(config.backends) if args.jobs is None else args.jobs
    out_dir = Path(args.out_dir) if args.out_dir else _REPO / Path(config.output).parent
    stem = Path(config.output).stem
    verbose = not args.quiet
    save = not args.no_save
    if save:
        out_dir.mkdir(parents=True, exist_ok=True)

    trajectory_cache: dict = {}
    results: dict[str, dict] = {}
    for variant in [v for v in VARIANTS if v in variants]:
        if verbose:
            print(f"\n=== {variant} ===")
        results[variant] = build_variant(
            config, asset, variant, out_dir=out_dir, stem=stem, jobs=jobs, verbose=verbose, save=save,
            trajectory_cache=trajectory_cache,
            production_manifest=results.get("production", {}).get("manifest"),
        )
        _print_report(results[variant]["report"])

    reports = {variant: result["report"] for variant, result in results.items()}
    summary = {"config": str(args.config), "stem": stem, "ok": all(r["ok"] for r in reports.values()),
               "variants": reports}
    if save:
        path = out_dir / f"{stem}.round6_checks.json"
        path.write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
        print(f"\nwrote {path}")
    print(f"\nround-6 build {'OK' if summary['ok'] else 'REFUSED'}")
    for variant, report in reports.items():
        if not report["ok"]:
            print(f"REFUSED {variant} {report.get('file') or '(not written)'}")
    return 0 if summary["ok"] or args.refusal_ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
