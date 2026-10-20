#!/usr/bin/env python3
"""S2R_06 F-9: what of the R0 residual is above the floor and coherent with a physical signal.

On every dynamic bag (excitation + identify_excitation) of the runs, every
1 kHz row, at one theta (default: the P-3 S1 result, else theta0):

* **real**: the R0 residual (W = 31) per band and joint;
* **floor**: the same at theta on the twin of each run (R0 at theta + FR_05
  noise on q and ft): noise plus differentiation;
* **explained**: the share of the real residual's band power that ``q''``
  and ``cmd - q`` explain linearly (coherence), and the phase-free share of
  the > 2 Hz power at the ripple order 320 / rad (real, and on the twin as
  the chance level);
* **R1** (unless ``--no-r1``): the R1 residual at the same theta (nominal K,
  measured-q reference): does the elastic plant explain it?

Writes ``f9.{json,md}`` to ``--output`` (default
``data/sim2real/kuka_lbr_iiwa_14_r820/p4b/f9``); ``--no-save`` prints only.

Example::

    python scripts/sim2real/f9_diagnostic.py --jobs 31
    python scripts/sim2real/f9_diagnostic.py --runs B --no-r1 --no-save
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, os.fspath(_REPO / "src"))

from elastic_sim.sim2real.config import load_config

DEFAULT_CONFIG = _REPO / "config" / "sim2real" / "kuka_lbr_iiwa_14_r820_base.yaml"
DATA = _REPO / "data" / "sim2real" / "kuka_lbr_iiwa_14_r820"
DEFAULT_THETA = DATA / "p3" / "iiwa_s1_rigid" / "best.yaml"
DEFAULT_OUTPUT = DATA / "p4b" / "f9"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--runs", nargs="+", default=["A", "B"])
    parser.add_argument("--theta", type=Path, default=None,
                        help=f"overlay; default {DEFAULT_THETA.relative_to(_REPO)} if present, else theta0")
    parser.add_argument("--theta0", action="store_true", help="use theta0 (nominal)")
    parser.add_argument("--seed", type=int, default=20261019, help="twin noise seed")
    parser.add_argument("--no-r1", action="store_true")
    parser.add_argument("--jobs", type=int, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args(argv)

    from elastic_sim.sim2real import spectra
    from elastic_sim.sim2real.parameters import Registry, check_write_target
    from elastic_sim.sim2real.plant import IntermediateBodies, PlantBase, PlantParameters
    from elastic_sim.sim2real.real_runs import load_real_run
    from elastic_sim.sim2real.replay_r0 import R0Engine, prepare_bags
    from elastic_sim.sim2real.replay_r1 import R1Engine, default_jobs, run_sigma_q
    from elastic_sim.sim2real.replay_r1 import prepare_bags as prepare_r1
    from elastic_sim.sim2real.report import write_json
    from elastic_sim.sim2real.twin import make_twin

    config = load_config(args.config)
    base = PlantBase.from_config(config)
    theta_path = None if args.theta0 else (args.theta or (DEFAULT_THETA if DEFAULT_THETA.is_file() else None))
    theta = {} if theta_path is None else {n: s.value for n, s in Registry.from_overlay(theta_path).specs.items()}
    intermediate = IntermediateBodies.from_elastic_build(base)
    r0 = dataclasses.replace(config.replay.r0, decimate=1, rows="all")
    joints = list(base.joint_names)
    real_res, twin_res, ddq, error, angles, r1_res, r1_diag = [], [], [], [], [], [], []
    rng = np.random.default_rng(args.seed)
    for run_id in args.runs:
        spec = config.runs[run_id]
        run = load_real_run(spec.path, expected_sha256=spec.sha256, segments=config.segments, run_id=run_id)
        plant = PlantParameters.from_theta(base, theta, runs=(run_id,))
        bags = [b for b in run.bags if b.kind != "hold"]
        engine = R0Engine(base, prepare_bags(bags, r0, time_step=1.0 / run.rate_hz), intermediate=intermediate)
        results = engine.replay(plant)
        real_res += [r.residual for r in results]
        ddq += [r.ddq for r in results]
        error += [np.asarray(b.cmd_echo, dtype=float) - np.asarray(b.q, dtype=float) for b in bags]
        angles += [np.asarray(b.q, dtype=float) for b in bags]
        twin, _ = make_twin(dataclasses.replace(run, bags=tuple(bags)), base, theta_star_full(plant, base, run_id),
                            config.replay.r0, rng, intermediate=intermediate)
        twin_engine = R0Engine(base, prepare_bags(twin.bags, r0, time_step=1.0 / run.rate_hz),
                               intermediate=intermediate)
        twin_res += [r.residual for r in twin_engine.replay(plant)]
        if not args.no_r1:
            r1_engine = R1Engine(base, prepare_r1(bags, r0, sigma_q={run_id: run_sigma_q(run)}), config.replay.r1,
                                 jobs=args.jobs or default_jobs())
            r1_results = r1_engine.replay(plant)
            r1_res += [r.residual for r in r1_results]
            r1_diag += [{"bag": f"{r.run_id}:{r.bag}", "wall_s": r.extra["wall_s"],
                         "tracking_rms_reference_quanta": (np.asarray(r.extra["tracking_rms_reference"])
                                                           / 5.989e-8).tolist()} for r in r1_results]
        print(f"run {run_id}: {len(bags)} dynamic bags", flush=True)

    labels = [spectra.band_label(b) for b in spectra.BANDS]
    f, p_real = spectra.psd_sum(real_res)
    _, p_twin = spectra.psd_sum(twin_res)
    rows_total = sum(len(r) for r in real_res)
    rms = {"real": np.sqrt(spectra.band_powers(f, p_real) / rows_total),
           "floor": np.sqrt(spectra.band_powers(f, p_twin) / rows_total)}
    rms["excess"] = np.sqrt(np.maximum(rms["real"] ** 2 - rms["floor"] ** 2, 0.0))
    if r1_res:
        _, p_r1 = spectra.psd_sum(r1_res)
        rms["r1"] = np.sqrt(spectra.band_powers(f, p_r1) / rows_total)
    explained = {"ddq": spectra.explained_fraction(real_res, ddq), "cmd_minus_q": spectra.explained_fraction(real_res, error),
                 "ddq_twin": spectra.explained_fraction(twin_res, ddq)}
    ripple = {f"order_{order:g}": {"real": spectra.order_fraction(real_res, angles, order=order).tolist(),
                                   "twin": spectra.order_fraction(twin_res, angles, order=order).tolist()}
              for order in (160.0, spectra.RIPPLE_ORDER, 640.0)}
    payload = {"diagnostic": "S2R_06 F-9", "runs": args.runs, "theta": str(theta_path) if theta_path else "theta0",
               "bands": labels, "joint_names": joints, "rows": rows_total, "twin_seed": args.seed,
               "rms_nm": {k: v.tolist() for k, v in rms.items()},
               "explained_fraction": {k: v.tolist() for k, v in explained.items()}, "ripple": ripple,
               "r1": r1_diag, "r1_settings": dataclasses.asdict(config.replay.r1)}
    text = to_markdown(payload)
    print(text)
    if not args.no_save:
        out = args.output.resolve()
        check_write_target(out / "f9.json")
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "f9.json", payload)
        (out / "f9.md").write_text(text, encoding="utf-8")
        print(f"written: {out}")
    return 0


def theta_star_full(plant, base, run_id: str) -> dict:
    """The twin's theta*: the analysis theta itself, every parameter named
    (``make_twin`` builds the plant from it)."""
    theta = dict(plant.theta)
    for j in base.joint_names:
        theta.setdefault(f"sensor.ft.{j}.offset@{run_id}", 0.0)
    return theta


def to_markdown(payload) -> str:
    from elastic_sim.sim2real import spectra

    joints = payload["joint_names"]
    labels = payload["bands"]
    rms = {k: np.asarray(v) for k, v in payload["rms_nm"].items()}
    lines = [f"# F-9 diagnostic (S2R_06): runs {', '.join(payload['runs'])}, theta `{payload['theta']}`", "",
             f"{payload['rows']} rows at 1 kHz (every dynamic bag), R0 W = 31. RMS per band, N m.", "",
             "## Real R0 residual", ""] + spectra.table(rms["real"], joints, rows=labels)
    lines += ["", "## Floor: the twin at the same theta (noise + differentiation)", ""]
    lines += spectra.table(rms["floor"], joints, rows=labels)
    lines += ["", "## Excess over the floor, sqrt(real^2 - floor^2)", ""]
    lines += spectra.table(rms["excess"], joints, rows=labels)
    for key, title in (("ddq", "q'' (SG)"), ("cmd_minus_q", "cmd - q (FRI command echo)"),
                       ("ddq_twin", "q'' on the twin (chance level)")):
        lines += ["", f"## Share of the residual's band power explained by {title}", ""]
        lines += spectra.table(np.asarray(payload["explained_fraction"][key]), joints, rows=labels)
    lines += ["", "## Ripple order: phase-free share of the > 2 Hz power (real / twin)", "",
              "| order / rad | " + " | ".join(joints) + " |", "|---|" + "---|" * len(joints)]
    for order, block in payload["ripple"].items():
        lines.append(f"| {order.split('_')[1]} | " + " | ".join(f"{a:.4f} / {b:.4f}" for a, b in
                                                                 zip(block["real"], block["twin"])) + " |")
    if "r1" in rms:
        lines += ["", "## R1 residual at the same theta (nominal K, measured-q reference)", ""]
        lines += spectra.table(rms["r1"], joints, rows=labels)
        lines += ["", "## R1 / R0 per band (< 1: the elastic plant explains part of it)", ""]
        lines += spectra.table(rms["r1"] / np.maximum(rms["real"], 1e-12), joints, rows=labels)
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
