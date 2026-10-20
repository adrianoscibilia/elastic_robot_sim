#!/usr/bin/env python3
"""Gate G-2a': the R0 twin experiment (S2R_01 T-4.6; S2R_08 §2.2).

Makes a synthetic copy of run B from R0 at a known theta* (seeded), runs
the config's calibration on it (a normal run folder,
``<output>/<run id>/``: fit, then the A-12' label), and scores the
recovery: ``twin_gate.{json,md}`` in that folder. With several seeds (default:
the gate's five, 20261019, 7, 11, 3, 5) it also writes the verdict over seeds
to ``<output>/<prefix>_gate.{md,json}``. Per seed (S2R_08 §2.2):

(i) identifiable combinations within 2 %; (ii) the run's offsets with
d_post >= 0.9 within 0.05 N m; (iii) held-out ft RMSE <= 1.25 x the residual
at theta* (noise + differentiation floor) on every joint (1.2 printed);
(iv)-b on the full-set posterior, >= 90 % of the free parameters' errors
within 2 posterior sd and none beyond 3 sd. Pass = every seed passes.

Example::

    python scripts/sim2real/twin.py --config config/sim2real/iiwa_twin_r0.yaml --run-id-prefix iiwa_twin_r0_label
    python scripts/sim2real/twin.py --seeds 7          # one seed
    python scripts/sim2real/twin.py --mode freeze_refit --run-id-prefix twin_a12   # S2R_06's A-12
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, os.fspath(_REPO / "src"))

from elastic_sim.sim2real.config import load_config

DEFAULT_CONFIG = _REPO / "config" / "sim2real" / "iiwa_twin_r0.yaml"


def main(argv: list[str] | None = None) -> int:
    from elastic_sim.sim2real.twin import GATE_SEEDS

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run", default="B", help="the real run whose states the twin uses")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(GATE_SEEDS), help="theta* / noise seeds")
    parser.add_argument("--budget", type=int)
    parser.add_argument("--run-id-prefix", help="run folders <prefix>_seed<seed> (default: the config stem)")
    parser.add_argument("--output", type=Path, help="override the config's output folder (under data/sim2real)")
    parser.add_argument("--mode", choices=("label", "freeze_refit"), help="override identifiability.mode")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    from elastic_sim.sim2real.parameters import check_write_target
    from elastic_sim.sim2real.plant import PlantBase
    from elastic_sim.sim2real.real_runs import load_real_run
    from elastic_sim.sim2real.report import write_json
    from elastic_sim.sim2real.twin import gate_summary, run_twin_gate

    config = load_config(args.config)
    if args.output is not None or args.mode is not None:
        import dataclasses

        resolved = dict(config.resolved)
        if args.output is not None:
            config = dataclasses.replace(config, output=args.output.resolve())
            resolved["output"] = str(config.output)
        if args.mode is not None:
            config = dataclasses.replace(config, identifiability=dataclasses.replace(config.identifiability,
                                                                                    mode=args.mode))
            resolved["identifiability"] = {**dict(resolved.get("identifiability") or {}), "mode": args.mode}
        config = dataclasses.replace(config, resolved=resolved)
    base = PlantBase.from_config(config)
    spec = config.runs[args.run]
    real = load_real_run(spec.path, expected_sha256=spec.sha256, segments=config.segments, run_id=args.run)
    prefix = args.run_id_prefix or config.source.stem
    gates = []
    for seed in args.seeds:
        folder, gate = run_twin_gate(config, run=args.run, seed=seed, budget=args.budget,
                                     run_id=f"{prefix}_seed{seed}", resume=args.resume, real=real, base=base,
                                     figures=not args.no_figures)
        print((folder / "twin_gate.md").read_text(encoding="utf-8").split("\n\nPer joint")[0])
        print(f"run folder: {folder}\n")
        gates.append(gate)
    table, verdict = gate_summary(gates)
    print(table)
    if len(gates) > 1:
        out = (config.output / f"{prefix}_gate.md").resolve()
        check_write_target(out)
        out.write_text(table, encoding="utf-8")
        write_json(out.with_suffix(".json"), {**verdict, "gates": {str(g["seed"]): g["checks"] for g in gates}})
        print(f"verdict: {out}")
    return 0 if verdict["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
