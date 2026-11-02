"""Standalone ``quicklook`` entry point (no ``rclpy``; RR_23 C6a).

Run as a subprocess from :mod:`erd_recording.pipeline` with
:func:`erd_recording.env_guard.clean_ros_subprocess_env`, like ``convert_cli``,
or directly on a run folder copied to peepo. Exit 0 = pass, 3 = stop.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .env_guard import assert_environment

assert_environment(require_ros=False, require_pinocchio=True)

from .config import load_lab_config  # noqa: E402
from .quicklook import print_summary, quicklook, write  # noqa: E402

EXIT_STOP = 3


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="erd_recording.quicklook_cli")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--previous", default=None, help="the previous rung's run folder (with quicklook.json)")
    parser.add_argument("--reference", default=None,
                        help="the mount-check run folder (with quicklook.json); with --mount-check, ur_B_all_02")
    parser.add_argument("--mount-check", action="store_true")
    args = parser.parse_args(argv or sys.argv[1:])

    config = load_lab_config(args.config)
    run_dir = Path(args.run_dir).expanduser().resolve()
    report = quicklook(config, run_dir, previous=Path(args.previous) if args.previous else None,
                       reference=Path(args.reference) if args.reference else None, mount_check=args.mount_check)
    write(report, run_dir)
    print_summary(report)
    print(f"wrote {run_dir / 'quicklook.json'} and quicklook.md")
    return EXIT_STOP if report["verdict"] == "stop" else 0


if __name__ == "__main__":
    raise SystemExit(main())
