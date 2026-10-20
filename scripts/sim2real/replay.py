#!/usr/bin/env python3
"""Watch one bag: replay it at theta0 or an overlay, print the metrics, plot ft (S2R_01 T-3.7).

Modes R0, R1 (motor-forced elastic replay) and R2 (closed loop on the FRI
command echo, the hand-set KUKA position-loop emulation: ``--bandwidth-hz``,
``--delay-cycles``, ``--drive-rate``, ``--model-scale``). ``--theta`` takes a
working or calibrated overlay (``working.yaml`` / ``best.yaml``, S2R_02 A-3).
``--show`` opens the plots without saving anything; ``--visualize`` (R1/R2)
watches the rollout in the MuJoCo viewer, nothing saved; ``--out DIR`` saves
the figure and the replay parquet; with none of them, only the numbers are
printed.

Examples
--------
::

    python scripts/sim2real/replay.py --run B --bag excitation_5 --show
    python scripts/sim2real/replay.py --run B --bag excitation_5 --mode R1 --show
    python scripts/sim2real/replay.py --run A --bag excitation_2 --mode R2 --bandwidth-hz 10 --visualize
    python scripts/sim2real/replay.py --run A --bag excitation_0 \
        --theta data/sim2real/kuka_lbr_iiwa_14_r820/iiwa_s1_rigid/working.yaml --show
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, os.fspath(_REPO / "src"))

from elastic_sim.sim2real.config import load_config

DEFAULT_CONFIG = _REPO / "config" / "sim2real" / "kuka_lbr_iiwa_14_r820_base.yaml"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="replay settings, runs, registry")
    parser.add_argument("--run", required=True, help="run id of the config (A, B)")
    parser.add_argument("--bag", required=True, help="bag name (excitation_5, identify_hold_0, ...)")
    parser.add_argument("--mode", default=None, choices=("R0", "R1", "R2"), help="default: the config's")
    parser.add_argument("--theta", type=Path, help="overlay (working.yaml / best.yaml); default theta0")
    parser.add_argument("--show", action="store_true", help="plot without saving")
    parser.add_argument("--out", type=Path, help="save figure + replay parquet here")
    parser.add_argument("--visualize", action="store_true", help="R1/R2: MuJoCo viewer, nothing saved")
    parser.add_argument("--bandwidth-hz", type=float, default=10.0, help="R2: hand-set loop bandwidth")
    parser.add_argument("--delay-cycles", type=int, default=1, help="R2: command delay, 1 ms cycles")
    parser.add_argument("--drive-rate", type=float, default=4000.0, help="R2: loop rate, Hz")
    parser.add_argument("--model-scale", type=float, default=1.0, help="R2: controller model inertia scale")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    mode = args.mode or config.replay.mode
    from elastic_sim.sim2real.metrics import metrics
    from elastic_sim.sim2real.parameters import Registry
    from elastic_sim.sim2real.plant import PlantBase, PlantParameters
    from elastic_sim.sim2real.real_runs import load_real_run
    from elastic_sim.sim2real.replay_r0 import R0Engine, prepare_bags, r0_settings_label

    spec = config.runs[args.run]
    run = load_real_run(spec.path, expected_sha256=spec.sha256, segments=config.segments, run_id=args.run)
    bag = run.bag(args.bag)
    base = PlantBase.from_config(config)
    registry = Registry.from_config(config, base)
    if args.theta is not None:
        registry = registry.with_overlay(args.theta)
    theta = registry.values()
    if mode == "R2":
        from elastic_sim.sim2real.replay_r2 import nominal_controller_theta

        theta.update(nominal_controller_theta(base, bandwidth_hz=args.bandwidth_hz, delay_cycles=args.delay_cycles,
                                              drive_rate=args.drive_rate, model_scale=args.model_scale))
    plant = PlantParameters.from_theta(base, theta, runs=tuple(config.runs))
    if mode == "R0":
        prepared = prepare_bags([bag], config.replay.r0, time_step=1.0 / run.rate_hz, split_of=config.split.split_of)
        result = R0Engine(base, prepared).replay(plant)[0]
        label = f"R0 [{r0_settings_label(config.replay.r0)}]"
    elif mode == "R1":
        from elastic_sim.sim2real import replay_r1

        engine = replay_r1.R1Engine(base, replay_r1.prepare_bags([bag], config.replay.r0,
                                                                split_of=config.split.split_of,
                                                                sigma_q={args.run: replay_r1.run_sigma_q(run)}),
                                    config.replay.r1, visualize=args.visualize)
        result = engine.replay(plant)[0]
        label = (f"R1 [track {config.replay.r1.track_bandwidth:g} rad/s, reference {config.replay.r1.reference_filter}, "
                 f"extras {config.replay.r1.extras}]; motor tracking (quanta): "
                 + " ".join(f"{v:.2f}" for v in replay_r1.tracking_check([result])["per_bag_quanta"].popitem()[1]))
    else:
        from elastic_sim.sim2real import replay_r2

        engine = replay_r2.R2Engine(base, replay_r2.prepare_bags([bag], config.replay.r0,
                                                                split_of=config.split.split_of),
                                    config.replay.r2, visualize=args.visualize)
        results = engine.replay(plant)
        if not results:
            raise SystemExit(f"R2 failed: {engine.last_outputs[0]['error']}")
        result = results[0]
        tracking = result.extra["tracking"]
        label = (f"R2 [{config.replay.r2.emulation}, {args.bandwidth_hz:g} Hz, delay {args.delay_cycles}, "
                 f"{args.drive_rate:g} Hz]; q - cmd RMS, mrad, real / sim: "
                 + " ".join(f"{1e3 * a:.3f}/{1e3 * b:.3f}" for a, b in zip(tracking["real_rms"], tracking["sim_rms"])))
    block = metrics([result], None, contracts={args.run: run.contract})
    print(f"{args.run}:{args.bag} ({bag.kind}, split {result.split or '-'}), {label}, "
          f"theta {'theta0' if args.theta is None else args.theta}")
    torque = block.get("torque")
    if torque:
        print(f"  RMSE {torque['rmse_median_over_bags']:.3f} N m; per joint RMSE (NRMSE): "
              + ", ".join(f"{r:.3f} ({n:.2f})" for r, n in zip(torque["per_joint_rmse"], torque["per_joint_nrmse"])))
        base_line = block["baselines"].get(args.run, {})
        print(f"  contract ({args.run}, median over its bags): rigid {base_line.get('baseline_rms_rigid'):.3f}, "
              f"state-fit {base_line.get('baseline_rms_state'):.3f}, tau-fit {base_line.get('baseline_rms_tau'):.3f}")
    residual = result.residual
    print("  residual mean per joint: " + " ".join(f"{v:+.3f}" for v in residual.mean(axis=0)))
    if args.show or args.out:
        import matplotlib

        if not args.show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = base.joint_names
        columns = 3 if mode == "R2" else 2
        fig, axes = plt.subplots(len(names), columns, figsize=(6.5 * columns, 2.0 * len(names)), sharex=True)
        for j, name in enumerate(names):
            axes[j, 0].plot(result.t, result.ft[:, j], lw=0.8, label="real ft")
            axes[j, 0].plot(result.t, result.ft_sim[:, j], lw=0.8, label="sim ft")
            axes[j, 0].set_ylabel(f"{name}\nN m")
            axes[j, 1].plot(result.t, residual[:, j], lw=0.6, color="C3")
            axes[j, 1].axhline(0.0, color="k", lw=0.4)
            axes[j, 1].set_ylabel(f"sim - real\nRMS {np.sqrt(np.mean(residual[:, j] ** 2)):.3f}")
            if mode == "R2":
                command = result.extra["command"]
                axes[j, 2].plot(result.t, 1e3 * (result.q[:, j] - command[:, j]), lw=0.7, label="real q - cmd")
                axes[j, 2].plot(result.t, 1e3 * (result.extra["q_motor_sim"][:, j] - command[:, j]), lw=0.7,
                                label="sim q - cmd")
                axes[j, 2].set_ylabel("mrad")
        axes[0, 0].legend(loc="upper right", fontsize=7)
        if mode == "R2":
            axes[0, 2].legend(loc="upper right", fontsize=7)
        axes[0, 0].set_title(f"{args.run}:{args.bag}, {mode}")
        fig.tight_layout()
        if args.out:
            from elastic_sim.sim2real.parameters import check_write_target
            from elastic_sim.sim2real.result import write_replay

            out = args.out.resolve()
            check_write_target(out / "x")
            out.mkdir(parents=True, exist_ok=True)
            fig.savefig(out / f"{args.run}_{args.bag}_{mode}.png", dpi=90)
            write_replay([result], out / f"{args.run}_{args.bag}_{mode}.parquet")
            print(f"  saved to {out}")
        if args.show:
            plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
