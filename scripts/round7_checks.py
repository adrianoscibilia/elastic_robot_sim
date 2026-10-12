#!/usr/bin/env python3
"""FR_04 S-1: a round-7 file has round 6's robots and trajectories, bit for bit.

Compares one r7 manifest (``payload.mode: none`` or ``fixed``) with the round-6
manifest it was copied from:

* ``trajectory_digests``: equal, key for key;
* ``robots[]``: stiffness, damping, damping ratio, rotor inertia, mode
  frequency and time step equal (exact float equality, as JSON stores them);
* ``records[]``, per bag: the same transmission fields plus the motor friction
  (``viscous``/``coulomb``: the per-robot friction factors), the trajectory
  and signal digests, the controller draw, the plant extras and the split;
* the attached payload: none (or the declared fixed body) on every elastic bag.

Exit 0 when every comparison holds, 1 otherwise; ``--out`` writes the report
as JSON.

    python scripts/round7_checks.py --r6 data/identification/r6-full-20260929-0049/fmrr/fmrr_tecnobody_round6.manifest.json \\
        --r7 data/identification/r7-nopayload/fmrr/fmrr_tecnobody_round7_nopayload.manifest.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROBOT_FIELDS = ("stiffness", "damping", "damping_ratio", "rotor_inertia", "mode_frequency_hz", "time_step")
RECORD_FIELDS = ROBOT_FIELDS + (
    "trajectory", "tier", "split", "viscous", "coulomb", "trajectory_digest", "trajectory_signal_digest",
    "control_natural_frequency", "control_damping_ratio", "control_position_gain", "control_velocity_bandwidth",
    "control_integral_time", "plant_extras", "exc_base_frequency", "exc_max_acceleration", "exc_velocity_fraction",
    "exc_probe_top_hz", "kp", "kd", "joint_frequency", "drive_load_inertia", "control_decimation",
)


def compare(r6: dict[str, Any], r7: dict[str, Any]) -> dict[str, Any]:
    """Every S-1 identity check between two manifests; ``ok`` when all hold."""
    checks: list[dict[str, Any]] = []

    d6, d7 = r6.get("trajectory_digests") or {}, r7.get("trajectory_digests") or {}
    differing = sorted(k for k in set(d6) | set(d7) if d6.get(k) != d7.get(k))
    checks.append({"check": "trajectory digests equal to round 6", "ok": bool(d6) and not differing,
                   "n_trajectories": len(d6), "differing": differing})

    robots6 = {r["name"]: r for r in r6.get("robots") or []}
    robots7 = {r["name"]: r for r in r7.get("robots") or []}
    robot_diffs = sorted(f"{name}.{field}" for name in set(robots6) | set(robots7) for field in ROBOT_FIELDS
                         if (robots6.get(name) or {}).get(field, "missing") != (robots7.get(name) or {}).get(field, "missing"))
    checks.append({"check": "per-robot transmission parameters equal to round 6 (bit for bit)",
                   "ok": bool(robots6) and set(robots6) == set(robots7) and not robot_diffs,
                   "n_robots": len(robots6), "fields": list(ROBOT_FIELDS), "differing": robot_diffs[:50]})

    records6 = {r["bag"]: r for r in r6.get("records") or []}
    records7 = {r["bag"]: r for r in r7.get("records") or []}
    record_diffs = sorted(f"{bag}.{field}" for bag in set(records6) & set(records7) for field in RECORD_FIELDS
                          if records6[bag].get(field, "missing") != records7[bag].get(field, "missing"))
    checks.append({"check": "per-bag robot, friction, trajectory and controller draws equal to round 6",
                   "ok": bool(records6) and set(records6) == set(records7) and not record_diffs,
                   "n_bags": len(records6), "missing_in_r7": sorted(set(records6) - set(records7)),
                   "extra_in_r7": sorted(set(records7) - set(records6)), "fields": list(RECORD_FIELDS),
                   "differing": record_diffs[:50]})

    mode = (r7.get("payload_mode") or {}).get("mode", "random")
    expected = (r7.get("payload_mode") or {}).get("attached")
    wrong = sorted(r["bag"] for r in r7.get("records") or [] if r.get("tier") != "rigid" and r.get("payload") != expected)
    checks.append({"check": f"attached payload is the payload.mode: {mode} body on every elastic bag",
                   "ok": mode != "random" and not wrong, "mode": mode, "attached": expected, "differing": wrong[:50]})

    masses6 = [(r.get("payload") or {}).get("mass", 0.0) for r in r6.get("robots") or []]
    return {"ok": all(c["ok"] for c in checks), "checks": checks,
            "round6_payload_mass_range": [min(masses6), max(masses6)] if masses6 else None}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--r6", required=True, help="the round-6 manifest the r7 config was copied from")
    parser.add_argument("--r7", required=True, help="the r7 manifest")
    parser.add_argument("--out", default=None, help="write the report here (JSON)")
    args = parser.parse_args(argv)
    report = compare(json.loads(Path(args.r6).read_text(encoding="utf-8")),
                     json.loads(Path(args.r7).read_text(encoding="utf-8")))
    report.update({"r6": str(args.r6), "r7": str(args.r7)})
    for check in report["checks"]:
        detail = "" if check["ok"] else f"  differing: {check.get('differing')}"
        print(f"  [{'x' if check['ok'] else ' '}] {check['check']}{detail}")
    print(f"S-1 identity {'OK' if report['ok'] else 'FAILED'}: {Path(args.r7).name} vs {Path(args.r6).name}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
