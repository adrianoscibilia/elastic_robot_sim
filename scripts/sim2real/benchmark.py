#!/usr/bin/env python3
"""T-3.6: wall time per bag of R0, R1, R2 at nominal theta; single worker and jobs = nproc - 1 (S2R_01 §10).

* single worker: R0 over every bag (one evaluation, per bag = total / bags),
  R1 and R2 on one excitation bag and one hold bag each;
* ``--jobs`` workers (default nproc - 1): one evaluation of R1 and of R2
  over every bag of the runs (the cost an optimizer pays per theta).

S2R_08 §4: **stop and report if one evaluation exceeds 10 min** -- the
script flags it (exit code 2).

Writes ``benchmark.{json,md}`` to ``--output`` (default
``data/sim2real/kuka_lbr_iiwa_14_r820/p4b/benchmark``); ``--no-save``
prints only.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, os.fspath(_REPO / "src"))

from elastic_sim.sim2real.config import load_config

DEFAULT_CONFIG = _REPO / "config" / "sim2real" / "kuka_lbr_iiwa_14_r820_base.yaml"
DEFAULT_OUTPUT = _REPO / "data" / "sim2real" / "kuka_lbr_iiwa_14_r820" / "p4b" / "benchmark"
LIMIT_S = 600.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--runs", nargs="+", default=["A", "B"])
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args(argv)

    from elastic_sim.sim2real import replay_r1, replay_r2
    from elastic_sim.sim2real.parameters import check_write_target
    from elastic_sim.sim2real.plant import PlantBase, PlantParameters
    from elastic_sim.sim2real.real_runs import load_real_run
    from elastic_sim.sim2real.replay_r0 import R0Engine, prepare_bags
    from elastic_sim.sim2real.report import write_json

    config = load_config(args.config)
    base = PlantBase.from_config(config)
    runs = {}
    for run_id in args.runs:
        spec = config.runs[run_id]
        runs[run_id] = load_real_run(spec.path, expected_sha256=spec.sha256, segments=config.segments, run_id=run_id)
    plant = PlantParameters.from_theta(base, replay_r2.nominal_controller_theta(base), runs=tuple(args.runs))
    all_bags = [b for run in runs.values() for b in run.bags]
    sigma = {r: replay_r1.run_sigma_q(run) for r, run in runs.items()}
    rows = {}

    def timed(label, function, bags, jobs):
        started = time.perf_counter()
        function()
        wall = time.perf_counter() - started
        rows[label] = {"bags": bags, "jobs": jobs, "wall_s": wall, "per_bag_s": wall / bags}
        print(f"{label}: {wall:.1f} s ({bags} bags, {jobs} worker(s))", flush=True)

    prepared = [x for run in runs.values() for x in prepare_bags(run.bags, config.replay.r0,
                                                                 time_step=1.0 / run.rate_hz)]
    r0 = R0Engine(base, prepared)
    timed("R0, every bag", lambda: r0.replay(plant), len(prepared), 1)
    first = next(iter(runs.values()))
    excitation = first.bag("excitation_5")
    hold = first.bag("identify_hold_0")
    for bag in (excitation, hold):
        engine1 = replay_r1.R1Engine(base, replay_r1.prepare_bags([bag], config.replay.r0, sigma_q=sigma),
                                     config.replay.r1)
        timed(f"R1, {bag.run_id}:{bag.bag} ({bag.t[-1]:.1f} s)", lambda e=engine1: e.replay(plant), 1, 1)
        engine2 = replay_r2.R2Engine(base, replay_r2.prepare_bags([bag], config.replay.r0), config.replay.r2)
        timed(f"R2, {bag.run_id}:{bag.bag} ({bag.t[-1]:.1f} s)", lambda e=engine2: e.replay(plant), 1, 1)
    engine1 = replay_r1.R1Engine(base, replay_r1.prepare_bags(all_bags, config.replay.r0, sigma_q=sigma),
                                 config.replay.r1, jobs=args.jobs)
    timed("R1, every bag", lambda: engine1.replay(plant), len(all_bags), args.jobs)
    engine2 = replay_r2.R2Engine(base, replay_r2.prepare_bags(all_bags, config.replay.r0), config.replay.r2,
                                 jobs=args.jobs)
    timed("R2, every bag", lambda: engine2.replay(plant), len(all_bags), args.jobs)
    over = [k for k, v in rows.items() if v["wall_s"] > LIMIT_S]
    payload = {"benchmark": "T-3.6", "cpu_count": os.cpu_count(), "jobs": args.jobs, "rows": rows,
               "limit_s": LIMIT_S, "over_limit": over,
               "physics_steps": {"R1": engine1.last_diagnostics[0]["time_step"] if engine1.last_diagnostics else None}}
    lines = ["# T-3.6 benchmark (nominal theta)", "",
             f"{os.cpu_count()} CPUs; parallel rows use {args.jobs} workers. S2R_08: stop if one evaluation > "
             f"{LIMIT_S / 60:g} min: **{'EXCEEDED: ' + ', '.join(over) if over else 'not exceeded'}**.", "",
             "| what | bags | workers | wall, s | per bag, s |", "|---|---|---|---|---|"]
    lines += [f"| {k} | {v['bags']} | {v['jobs']} | {v['wall_s']:.1f} | {v['per_bag_s']:.2f} |" for k, v in rows.items()]
    text = "\n".join(lines) + "\n"
    print(text)
    if not args.no_save:
        out = args.output.resolve()
        check_write_target(out / "benchmark.json")
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "benchmark.json", payload)
        (out / "benchmark.md").write_text(text, encoding="utf-8")
        print(f"written: {out}")
    return 2 if over else 0


if __name__ == "__main__":
    raise SystemExit(main())
