#!/usr/bin/env python3
"""R2 at nominal theta on every bag: does it run, and real vs simulated q - cmd (S2R_01 T-3.4; S2R_08 §4).

The hand-set KUKA position-loop emulation (``--bandwidth-hz``, default 10;
zeta 1; ``--delay-cycles`` 1; ``--drive-rate`` 4 kHz; model scale 1) on the
elastic plant at theta0, every bag of the runs (no fit). Reported per joint
(A-2 (e)): the tracking error ``q - cmd`` RMS and max, real and simulated,
their difference and correlation (median over bags); the bags that diverged;
the ft residual; wall time per bag.

Writes ``r2.{json,md}`` to ``--output`` (default
``data/sim2real/kuka_lbr_iiwa_14_r820/p4b/r2_nominal``); ``--no-save``
prints only.

Example::

    python scripts/sim2real/r2_nominal.py --jobs 31
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, os.fspath(_REPO / "src"))

from elastic_sim.sim2real.config import load_config

DEFAULT_CONFIG = _REPO / "config" / "sim2real" / "kuka_lbr_iiwa_14_r820_base.yaml"
DEFAULT_OUTPUT = _REPO / "data" / "sim2real" / "kuka_lbr_iiwa_14_r820" / "p4b" / "r2_nominal"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--runs", nargs="+", default=["A", "B"])
    parser.add_argument("--kinds", nargs="+", default=["excitation", "identify_excitation", "hold"])
    parser.add_argument("--bandwidth-hz", type=float, default=10.0)
    parser.add_argument("--delay-cycles", type=int, default=1)
    parser.add_argument("--drive-rate", type=float, default=4000.0)
    parser.add_argument("--model-scale", type=float, default=1.0)
    parser.add_argument("--jobs", type=int, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args(argv)

    from elastic_sim.sim2real.parameters import check_write_target
    from elastic_sim.sim2real.plant import PlantBase, PlantParameters
    from elastic_sim.sim2real.real_runs import load_real_run
    from elastic_sim.sim2real.replay_r1 import default_jobs
    from elastic_sim.sim2real.replay_r2 import R2Engine, nominal_controller_theta, prepare_bags
    from elastic_sim.sim2real.report import write_json

    config = load_config(args.config)
    base = PlantBase.from_config(config)
    controller = nominal_controller_theta(base, bandwidth_hz=args.bandwidth_hz, delay_cycles=args.delay_cycles,
                                          drive_rate=args.drive_rate, model_scale=args.model_scale)
    plant = PlantParameters.from_theta(base, controller, runs=tuple(args.runs))
    bags = []
    for run_id in args.runs:
        spec = config.runs[run_id]
        run = load_real_run(spec.path, expected_sha256=spec.sha256, segments=config.segments, run_id=run_id)
        bags += prepare_bags([b for b in run.bags if b.kind in args.kinds], config.replay.r0,
                             split_of=config.split.split_of)
    engine = R2Engine(base, bags, config.replay.r2, jobs=args.jobs or default_jobs())
    started = time.perf_counter()
    results = engine.replay(plant)
    wall = time.perf_counter() - started
    per_bag, failed = {}, []
    for bag, output in zip(bags, engine.last_outputs):
        key = f"{bag.run_id}:{bag.bag}"
        if "error" in output:
            failed.append({"bag": key, "error": output["error"], "diverged": output["diverged"]})
            continue
        per_bag[key] = {"kind": bag.kind, **output["tracking"], "wall_s": output["diagnostics"]["wall_s"],
                        "time_step": output["diagnostics"]["time_step"],
                        "equilibrium_residual_nm": output["diagnostics"]["equilibrium"].get("residual_max_nm")}
    for result in results:
        per_bag[f"{result.run_id}:{result.bag}"]["ft_rmse"] = np.sqrt(np.mean(result.residual ** 2, axis=0)).tolist()
    dynamic = [v for v in per_bag.values() if v["kind"] != "hold"]
    summary = {key: np.median([v[key] for v in dynamic], axis=0).tolist()
               for key in ("real_rms", "sim_rms", "difference_rms", "real_max", "sim_max", "ft_rmse")}
    summary["correlation"] = np.median([[np.nan if c is None else c for c in v["correlation"]] for v in dynamic],
                                       axis=0).tolist()
    payload = {"engine": "R2", "theta": "theta0", "controller": controller, "emulation": config.replay.r2.emulation,
               "runs": args.runs, "n_bags": len(bags), "n_ran": len(per_bag), "failed": failed,
               "evaluation_wall_s": wall, "jobs": engine.jobs, "joint_names": list(base.joint_names),
               "summary_dynamic_median": summary, "per_bag": per_bag}
    text = to_markdown(payload)
    print(text)
    if not args.no_save:
        out = args.output.resolve()
        check_write_target(out / "r2.json")
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "r2.json", payload)
        (out / "r2.md").write_text(text, encoding="utf-8")
        print(f"written: {out}")
    return 0 if not failed else 1


def to_markdown(payload) -> str:
    joints = payload["joint_names"]
    c = payload["controller"]
    s = payload["summary_dynamic_median"]
    lines = [f"# R2 at nominal theta ({payload['emulation']}), runs {', '.join(payload['runs'])}", "",
             f"Hand-set controller: bandwidth {c[f'controller.{joints[0]}.bandwidth'] / (2 * np.pi):g} Hz, zeta "
             f"{c[f'controller.{joints[0]}.damping_ratio']:g}, delay {c['controller.delay_cycles']:g} cycle(s), drive "
             f"{c['controller.drive_rate']:g} Hz, model scale {c['controller.model_scale']:g}.",
             f"**{payload['n_ran']} of {payload['n_bags']} bags ran; diverged / failed: {len(payload['failed'])}**"
             + (": " + ", ".join(f"{f['bag']} ({f['error']})" for f in payload["failed"]) if payload["failed"] else "")
             + f". One evaluation (every bag, {payload['jobs']} workers): {payload['evaluation_wall_s']:.0f} s.", "",
             "Median over the dynamic bags (excitation + identify_excitation):", "",
             "| | " + " | ".join(joints) + " |", "|---|" + "---|" * len(joints)]
    for key, label, scale, fmt in (("real_rms", "real q - cmd RMS, mrad", 1e3, ".3f"),
                                   ("sim_rms", "sim q - cmd RMS, mrad", 1e3, ".3f"),
                                   ("difference_rms", "RMS(sim - real) of q - cmd, mrad", 1e3, ".3f"),
                                   ("correlation", "correlation real / sim", 1.0, ".3f"),
                                   ("real_max", "real q - cmd max, mrad", 1e3, ".3f"),
                                   ("sim_max", "sim q - cmd max, mrad", 1e3, ".3f"),
                                   ("ft_rmse", "ft RMSE (scored rows), N m", 1.0, ".3f")):
        lines.append(f"| {label} | " + " | ".join(format(scale * float(v), fmt) for v in s[key]) + " |")
    lines += ["", "Per bag (q - cmd RMS, mrad, real / sim; wall s):", "",
              "| bag | " + " | ".join(joints) + " | wall |", "|---|" + "---|" * (len(joints) + 1)]
    for bag, v in payload["per_bag"].items():
        lines.append(f"| {bag} | " + " | ".join(f"{1e3 * a:.3f} / {1e3 * b:.3f}" for a, b in
                                                 zip(v["real_rms"], v["sim_rms"])) + f" | {v['wall_s']:.0f} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
