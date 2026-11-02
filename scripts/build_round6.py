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

``--round8`` (``identified_filter/01`` Sec 1, a config with a ``round8:`` block)
adds the ``oop`` (out-of-prior robots, test only) and ``modal`` (single-joint
chirps of 10 test robots) variants, the per-bag oracle verification of every
elastic bag (written as privileged per-bag columns), the modal-coverage gate on
the production file's ``full`` bags with up to ``retries`` regenerations of a
failing bag at the next seed, the ``mjj*`` audit columns and the contract's
``privileged`` / ``modal_coverage`` / ``round8`` blocks.  A platform whose gate
joint is observable in fewer than ``raise_fraction`` of its ``full`` bags exits
4: rerun with ``--coverage-escalation`` one higher (raises the lines' level).

``--taufree`` (FR_04, a ``payload.mode: none``/``fixed`` config) adds to every
contract the ``audit_columns``, ``transmission_priors`` and
``baselines.two_mass`` blocks (:mod:`elastic_sim.round7`), and writes the
production file's ``<stem>.two_mass_reference.npz`` cross-test fixture.

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
import pandas as pd

from diagnose_controller_modes import _load_asset, _shrink, add_baselines_to_contract
from elastic_sim.dataset import PlantExtrasSampling, assign_splits, default_jobs, generate, load_config, write_dataset
from elastic_sim.diagnostics import dataset_diagnostics
from elastic_sim.round6_checks import check_dataset, gate_gainshift
from elastic_sim.round7 import augment_contract
from elastic_sim.round8 import (add_bag_columns, add_inertia_audit, add_modal_columns, contract_block, coverage_report, regimes_of,
                                sample_oop_robots, urdf_home_inertia, verify_bags)

VARIANTS = ("production", "gainshift", "ablation_off", "ablation_clean", "oop", "modal")
SUFFIX = {"production": "", "gainshift": "_gainshift", "ablation_off": "_ablation_off",
          "ablation_clean": "_ablation_clean", "oop": "_oop", "modal": "_modal"}
#: Exit code of a round-8 build whose modal-coverage gate asks for a higher escalation (Sec 1.1.3).
EXIT_COVERAGE = 4
#: Round 8: stop regenerating when a retry recovers fewer than this share of the failing bags.
REGEN_MIN_RECOVERY = 0.10


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
    if variant in ("oop", "modal") and config.round8 is None:
        raise ValueError(f"the {variant} file needs a round8 config block")
    if variant == "oop":
        # identified_filter/01 Sec 1: K outside the widened range, test only, no rigid tier.
        spec = config.round8
        tiers = sample_oop_robots(config.transmission, config.seed, spec.oop)
        return replace(config, tiers=tiers, rigid_reference=False, fixed_split="test",
                       transmission=replace(config.transmission, robots=len(tiers)),
                       round8=replace(spec, trajectories_by_split=(("test", spec.oop.trajectories),)),
                       n_trajectories=max(config.n_trajectories, spec.oop.trajectories))
    if variant == "modal":
        labels = assign_splits(config.tiers, config.split)
        test = tuple(sorted(name for name, label in labels.items() if label == "test"))[:config.round8.modal.robots]
        return replace(config, only_tiers=test, modal=config.round8.modal)
    raise ValueError(f"unknown variant {variant!r}")


def paired_with_production(gainshift_config, production_manifest: dict):
    """The gain-shift config on the production file's final seeds and dropped bags (`identified_filter/03` F2)."""
    return replace(gainshift_config,
                   reseed=tuple(tuple(r) for r in production_manifest.get("reseed", ())),
                   excluded_bags=tuple(tuple(b) for b in production_manifest.get("excluded_bags", ())))


def accept_sag_failures(report: dict) -> None:
    """Round 8 (`identified_filter/03` F4): a bag over the sag limit is reported, not a refusal of the file."""
    sag = next(g for g in report["gates"] if g["gate"].startswith("2"))
    if sag["ok"]:
        return
    report["sag_failing_bags"] = sag.get("failing_bags", [])
    report["ok"] = all(c["ok"] for c in report["hard_checks"]) and all(g["ok"] for g in report["gates"] if g is not sag)


def _verify(frame, manifest, contract, asset, spec, jobs: int):
    """``verify_bags`` over the elastic bags, one bag per task in a pool when ``jobs > 1``."""
    import concurrent.futures

    bags = [b for b, g in frame.groupby("bag", sort=False) if str(g["tier"].iloc[0]) != "rigid"]
    if jobs <= 1 or len(bags) < 2:
        return verify_bags(frame, manifest, contract, asset, spec)
    chunks = [frame[frame["bag"] == bag] for bag in bags]
    with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as pool:
        parts = list(pool.map(verify_bags, chunks, *(([x] * len(chunks)) for x in (manifest, contract, asset, spec))))
    return pd.concat(parts, ignore_index=True)


def _splice(frame, manifest, new_frame, new_manifest, bags):
    """Replace ``bags`` of a build by their regenerated versions, keeping the bag order."""
    order = {bag: i for i, bag in enumerate(frame["bag"].unique())}
    kept = frame[~frame["bag"].isin(bags)]
    merged = pd.concat([kept, new_frame[new_frame["bag"].isin(bags)]], ignore_index=True)
    merged = merged.iloc[np.argsort(merged["bag"].map(order).to_numpy(), kind="stable")].reset_index(drop=True)
    records = {r["bag"]: r for r in new_manifest.get("records", []) if r["bag"] in bags}
    manifest = dict(manifest)
    manifest["records"] = [records.get(r["bag"], r) for r in manifest.get("records", [])]
    return merged, manifest


def round8_verification(variant_cfg, asset, frame, manifest, *, jobs: int, verbose: bool, gate: bool,
                        trajectory_cache: dict):
    """Sec 1.1.3: verify every elastic bag; on the production file regenerate failing ``full`` bags.

    Returns ``(frame, manifest, verification, coverage, attempts)``.  An
    excitation-limited bag (a resolvable main joint not observable, its SNR below
    the threshold) is rebuilt at the next seed, at most ``retries`` times, then
    kept and flagged (``regen_attempt``).  A fit-limited bag is never rebuilt:
    its SNR is fine, so a new seed cannot help (`identified_filter/03` F6).
    """
    from elastic_sim.dataset_config import differentiation_policy
    from elastic_sim.round7 import transmission_priors

    spec = variant_cfg.round8
    # What verify_bags reads of the contract, before it is written: the SG window and the burn-in.
    contract = {"differentiation": differentiation_policy(variant_cfg),
                "transmission_priors": transmission_priors(manifest, asset,
                                                           burn_in_override_s=variant_cfg.burn_in_override_s)}
    verification = _verify(frame, manifest, contract, asset, spec, jobs)
    regimes = regimes_of(variant_cfg, frame)
    coverage = coverage_report(verification, regimes, spec)
    attempts: dict[str, int] = {}
    reseed = {(r[0], int(r[1])): int(r[2]) for r in variant_cfg.reseed}
    previous_failing = 0
    for attempt in range(1, spec.verification.retries + 1 if gate else 1):
        failing = [b for b in coverage["excitation_limited_bags"] if attempts.get(b, 0) == attempt - 1]
        if not failing:
            break
        if attempt > 1 and len(failing) > (1.0 - REGEN_MIN_RECOVERY) * previous_failing:
            # The last retry fixed almost nothing: the failure is not the draw (a new seed cannot fix a plant
            # whose spring is not the linear filter's, e.g. the nonlinear spring and transmission error).
            coverage["regeneration_stopped"] = (f"attempt {attempt - 1} recovered {previous_failing - len(failing)} "
                                                f"of {previous_failing} bags")
            break
        previous_failing = len(failing)
        pairs, proposed = [], dict(reseed)
        for bag in failing:
            tier = str(frame.loc[frame["bag"] == bag, "tier"].iloc[0])
            index = int(str(bag).split("_")[0][1:])
            pairs.append((tier, index))
            proposed[(tier, index)] = proposed.get((tier, index), 0) + 1
        if verbose:
            print(f"[round8] regenerating {len(failing)} full bag(s), attempt {attempt}: {failing[:6]}...")
        sub = replace(variant_cfg, reseed=tuple((t, i, a) for (t, i), a in proposed.items()), only_bags=tuple(pairs))
        new_frame, new_manifest, _ = generate(sub, asset, verbose=False, jobs=jobs, trajectory_cache=trajectory_cache)
        done = sorted(set(new_frame["bag"]) & set(failing))
        frame, manifest = _splice(frame, manifest, new_frame, new_manifest, done)
        # A bag the retry could not build keeps its previous version, so it keeps that version's seed.
        final = {(r[0], int(r[1])): int(r[2]) for r in new_manifest.get("reseed", ())}
        for bag in done:
            pair = (str(frame.loc[frame["bag"] == bag, "tier"].iloc[0]), int(str(bag).split("_")[0][1:]))
            reseed[pair] = final[pair]
        for bag in done:
            attempts[bag] = attempt
        fresh = _verify(frame[frame["bag"].isin(done)], manifest, contract, asset, spec, jobs)
        verification = pd.concat([verification[~verification["bag"].isin(done)], fresh], ignore_index=True)
        variant_cfg = replace(variant_cfg, reseed=tuple((t, i, a) for (t, i), a in reseed.items()))
        regimes = regimes_of(variant_cfg, frame)
        coverage = coverage_report(verification, regimes, spec)
    coverage["regenerated"] = attempts
    coverage["still_failing"] = coverage["failing_full_bags"]
    manifest["reseed"] = [[t, i, a] for (t, i), a in reseed.items()]
    # Escalation (stronger lines) can only help a joint whose full-regime in-band SNR is below the threshold.
    full = coverage["per_regime"].get("full", {})
    coverage["excitation_limited"] = sorted(
        joint for joint in coverage["raise_joints"]
        if not (full.get(joint, {}).get("median_snr", float("nan")) >= spec.verification.snr_min))
    frame = add_bag_columns(frame, regimes, verification, attempts)
    return frame, manifest, verification, coverage, attempts


def build_variant(config, asset, variant: str, *, out_dir: Path, stem: str, jobs: int, verbose: bool,
                  save: bool, trajectory_cache: dict, production_manifest: dict | None = None,
                  taufree: bool = False, priors: dict | None = None, round8: bool = False) -> dict:
    variant_cfg = variant_config(config, variant)
    if variant == "gainshift" and production_manifest is not None:
        variant_cfg = paired_with_production(variant_cfg, production_manifest)
    frame, manifest, comparison = generate(variant_cfg, asset, verbose=verbose, jobs=jobs,
                                           trajectory_cache=trajectory_cache)
    if round8:
        # What generate settled: the seeds and the dropped bags every later step of this file must read.
        variant_cfg = replace(variant_cfg, reseed=tuple(tuple(r) for r in manifest.get("reseed", ())),
                              excluded_bags=tuple(tuple(b) for b in manifest.get("excluded_bags", ())))
    coverage = None
    if round8 and variant != "modal":
        frame, manifest, _, coverage, _ = round8_verification(
            variant_cfg, asset, frame, manifest, jobs=jobs, verbose=verbose, gate=variant == "production",
            trajectory_cache=trajectory_cache)
    elif round8:
        frame = add_bag_columns(frame, {b: "modal" for b in frame["bag"].unique()}, None)
        frame = add_modal_columns(frame, variant_cfg.modal)
    if round8:
        frame = add_inertia_audit(frame, asset, int(manifest["n_dof"]))
    diagnostics = dataset_diagnostics(frame, manifest, asset)
    report = check_dataset(frame, manifest, diagnostics=diagnostics, production=variant == "production")
    if variant == "gainshift" and production_manifest is not None:
        report["gates"].append(gate_gainshift(production_manifest, manifest))
        report["ok"] = bool(report["ok"] and report["gates"][-1]["ok"]
                            and next(g for g in report["gates"] if g["gate"].startswith("1"))["ok"])
    if round8:
        accept_sag_failures(report)
        report["dropped_bags"] = manifest.get("dropped_bags", [])
    report["variant"] = variant
    report["bags"] = int(manifest["n_bags"])
    if coverage is not None:
        report["modal_coverage"] = coverage
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
        if taufree:
            # FR_04 S-2..S-4; a gain-shift twin takes the production file's priors (one burn-in per variant).
            priors = augment_contract(output, frame, manifest, asset, priors=priors, fixture=variant == "production",
                                      burn_in_override_s=config.burn_in_override_s)
            report["transmission_priors_burn_in_s"] = priors["burn_in_s"]
        if round8:
            write_round8_contract(output, variant_cfg, asset, coverage, variant)
    return {"report": report, "manifest": manifest, "priors": priors}


def write_round8_contract(output: Path, config, asset, coverage, variant: str) -> None:
    """The contract's r8 blocks, ``M_ref`` per joint in ``transmission_priors``, and the file's SHA-256."""
    import hashlib

    spec = config.round8
    home = spec.modal.poses[0][1] if spec.modal.poses else [0.0] * len(asset.joint_names)
    reference = urdf_home_inertia(asset, home)
    path = output.with_suffix(".contract.json")
    contract = json.loads(path.read_text(encoding="utf-8"))
    contract.update(contract_block(spec, coverage, variant=variant, reference_inertia=reference, home=home))
    priors = contract.get("transmission_priors")
    if priors:
        manifest = json.loads(output.with_suffix(".manifest.json").read_text(encoding="utf-8"))
        robots = manifest.get("robots") or []
        for j, joint in enumerate(priors["joints"]):
            joint["link_inertia_home"] = reference[j]
            if robots:
                damping = [float(r["damping"][j]) for r in robots]
                joint["damping_range"] = [min(damping), max(damping)]
    digest = hashlib.sha256()
    with open(output, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    contract["dataset_sha256"] = digest.hexdigest()
    path.write_text(json.dumps(contract, indent=2, default=float), encoding="utf-8")


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
    parser.add_argument("--taufree", action="store_true",
                        help="FR_04: also write audit_columns, transmission_priors and the two-mass baselines into "
                             "each contract, and the production file's cross-test fixture (needs payload.mode "
                             "fixed or none)")
    parser.add_argument("--round8", action="store_true",
                        help="identified_filter/01 Sec 1: also build the oop and modal files, verify every bag, "
                             "gate and regenerate full bags, write the privileged metadata (needs a round8 block)")
    parser.add_argument("--coverage-escalation", type=int, default=0,
                        help="round 8: 1 = all full bags acceleration-weighted at fraction [0.8, 1]; 2 = also the "
                             "fewest lines (exit 4 asks for one level more)")
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
    if args.round8:
        if config.round8 is None:
            parser.error("--round8 needs a config with a round8: block")
        config = replace(config, round8=replace(config.round8, escalation=int(args.coverage_escalation)))
        if args.robots is not None:
            # A preflight-sized r8 build: one out-of-prior robot, one modal robot, one chirp per joint and pose.
            spec = config.round8
            config = replace(config, round8=replace(spec, oop=replace(spec.oop, robots=1),
                                                    modal=replace(spec.modal, robots=1, repeats=1)))
    variants = list(dict.fromkeys(args.variants + (["ablation_off", "ablation_clean"] if args.ablations else [])
                                  + (["oop", "modal"] if args.round8 else [])))
    if "gainshift" in variants and "production" not in variants:
        parser.error("the gain-shift file is checked against the production file; build both")
    if args.taufree and config.payload.mode == "random":
        parser.error("--taufree needs a payload.mode: fixed or none config (FR_04 S-1)")
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
            taufree=args.taufree, priors=results.get("production", {}).get("priors"), round8=args.round8,
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
    coverage = reports.get("production", {}).get("modal_coverage")
    if coverage is not None:
        print(f"round-8 modal coverage gate {'OK' if coverage['gate_ok'] else 'FAILED'}; "
              f"raise: {coverage['raise_joints'] or 'none'}; still failing: {len(coverage['still_failing'])} bag(s)")
        if coverage["raise_joints"] and coverage.get("excitation_limited"):
            print(f"EXIT {EXIT_COVERAGE}: rerun with --coverage-escalation {args.coverage_escalation + 1}")
            return EXIT_COVERAGE
        if coverage["raise_joints"]:
            print("round-8 coverage: the failing joints' in-band SNR is above the threshold, so stronger lines "
                  "cannot fix them (the oracle fit, not the excitation, fails); the gate is recorded, not escalated")
    for variant, report in reports.items():
        if not report["ok"]:
            print(f"REFUSED {variant} {report.get('file') or '(not written)'}")
    return 0 if summary["ok"] or args.refusal_ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
