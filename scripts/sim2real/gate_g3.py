#!/usr/bin/env python3
"""Gate G-3: R1 with K x 100 against R0 on three bags of B (S2R_01 T-3.3; S2R_08 §4).

R1's reference is pre-filtered by R0's SG (W = 31, poly 3) so both engines
see the same motion. Per bag and joint, the RMS of tau_s(R1) - tau_s(R0):
``spec`` (R0 as configured, S2R_01's <= 0.05 N m), ``same_kinematics`` (R0
on R1's reference), ``lowpass`` (that, below 10 Hz). Also the motor
tracking (T-3.3: <= 10 encoder quanta) and the wall time per bag.

Writes ``g3.{json,md}`` to ``--output`` (default
``data/sim2real/kuka_lbr_iiwa_14_r820/p4b/g3``); ``--no-save`` prints only.

Example::

    python scripts/sim2real/gate_g3.py --jobs 3
    python scripts/sim2real/gate_g3.py --bags excitation_5 --no-save
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, os.fspath(_REPO / "src"))

from elastic_sim.sim2real.config import load_config

DEFAULT_CONFIG = _REPO / "config" / "sim2real" / "kuka_lbr_iiwa_14_r820_base.yaml"
DEFAULT_OUTPUT = _REPO / "data" / "sim2real" / "kuka_lbr_iiwa_14_r820" / "p4b" / "g3"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run", default="B")
    parser.add_argument("--bags", nargs="+", default=["excitation_0", "excitation_5", "excitation_9"])
    parser.add_argument("--theta", type=Path, help="an overlay (best.yaml / working.yaml); default theta0")
    parser.add_argument("--stiffness-scale", type=float, default=None, help="default 100 (S2R_01)")
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-save", action="store_true", help="print only")
    args = parser.parse_args(argv)

    from elastic_sim.sim2real.parameters import Registry, check_write_target
    from elastic_sim.sim2real.plant import PlantBase, PlantParameters
    from elastic_sim.sim2real.real_runs import load_real_run
    from elastic_sim.sim2real.replay_r1 import G3_STIFFNESS_SCALE, gate_g3
    from elastic_sim.sim2real.report import write_json

    config = load_config(args.config)
    base = PlantBase.from_config(config)
    spec = config.runs[args.run]
    run = load_real_run(spec.path, expected_sha256=spec.sha256, segments=config.segments, run_id=args.run)
    theta = {} if args.theta is None else {n: s.value for n, s in Registry.from_overlay(args.theta).specs.items()}
    plant = PlantParameters.from_theta(base, theta, runs=(args.run,))
    gate = gate_g3(run, base, plant, bags=args.bags, r0=config.replay.r0, r1=config.replay.r1, jobs=args.jobs,
                   stiffness_scale=args.stiffness_scale or G3_STIFFNESS_SCALE)
    gate["theta_source"] = "theta0" if args.theta is None else str(args.theta)
    text = to_markdown(gate)
    print(text)
    if not args.no_save:
        out = args.output.resolve()
        check_write_target(out / "g3.json")
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "g3.json", gate)
        (out / "g3.md").write_text(text, encoding="utf-8")
        print(f"written: {out}")
    return 0


def to_markdown(gate) -> str:
    joints = gate["joint_names"]
    ok = {True: "pass", False: "**FAIL**"}
    lines = [f"# Gate G-3 (R1 K x {gate['stiffness_scale']:g} vs R0), run {gate['run']}, theta {gate['theta_source']}",
             "", f"* spec (R0 as configured, W = 31): worst {gate['worst']['spec']:.4f} N m, tolerance "
             f"{gate['tolerance_nm']} -> {ok[gate['pass_spec']]}",
             f"* same kinematics (R0 on R1's reference): worst {gate['worst']['same_kinematics']:.4f} N m",
             f"* same kinematics below {gate['lowpass_hz']:g} Hz: worst {gate['worst']['lowpass']:.5f} N m -> "
             f"{ok[gate['pass_lowpass']]}",
             f"* motor tracking (vs reference): worst {gate['tracking_worst_quanta']:.2f} quanta (T-3.3: <= 10)", "",
             "| bag | comparison | " + " | ".join(joints) + " |", "|---|---|" + "---|" * len(joints)]
    for bag, block in gate["per_bag"].items():
        for key in ("spec", "same_kinematics", "lowpass"):
            lines.append(f"| {bag} | {key} | " + " | ".join(f"{v:.4f}" for v in block[key]) + " |")
        lines.append(f"| {bag} | tracking, quanta | " + " | ".join(f"{v:.2f}" for v in
                                                                block["tracking_rms_reference_quanta"]) + " |")
    lines += ["", "| bag | time step, s | wall, s | equilibrium residual, N m |", "|---|---|---|---|"]
    lines += [f"| {bag} | {b['time_step']:.3g} | {b['wall_s']:.1f} | {b['equilibrium_residual_nm']:.2e} |"
              for bag, b in gate["per_bag"].items()]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
