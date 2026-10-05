"""RR_08 S2d: a third RTDE connection, carrying only the speed-slider
*input* recipe, that raises/lowers ``speed_slider_fraction`` mid-run --
scripting the live monitor's speed-scaling abort trigger (P4 item 9) instead
of operating the pendant by hand.

RTDE allows several simultaneous connections to the one controller, each with
its own recipe; this never touches the driver's own connection or the
sidecar's (:mod:`erd_ur10.rtde_logger`, a separate, read-only *output*
connection). The monitor reads the result back as ``speed_scaling`` over
*its* own connection (``/speed_scaling_state_broadcaster/speed_scaling``,
normalized 0-100 -> 0-1 in ``pipeline.RecordingNode._on_speed_scaling``) and
aborts once it drops below ``safety.SPEED_SCALING_FLOOR`` (0.999) --
``--fraction 0.5`` (the architect's own worked example) is comfortably below
that.
"""

from __future__ import annotations

import argparse
import time
from typing import Any

#: UR RTDE input recipe for the speed slider (RTDE Guide, control package
#: inputs): mask 1 means "use speed_slider_fraction"; mask 0 releases it back
#: to the pendant/program default.
SPEED_SLIDER_RECIPE: tuple[tuple[str, str], ...] = (
    ("speed_slider_mask", "UINT32"),
    ("speed_slider_fraction", "DOUBLE"),
)


def build_speed_slider_connection(host: str, port: int = 30004) -> tuple[Any, Any]:
    """Return a connected, started ``rtde.RTDE`` client plus its input
    recipe object, ready for :func:`trigger_speed_slider`.

    Split out from :func:`main` so a test can monkeypatch ``rtde.RTDE`` with
    a fake connection instead of touching a real network socket (mirrors
    ``erd_ur10.rtde_logger.build_rtde_connection``).
    """
    from rtde import rtde as rtde_module

    names = [name for name, _ in SPEED_SLIDER_RECIPE]
    types = [kind for _, kind in SPEED_SLIDER_RECIPE]
    connection = rtde_module.RTDE(host, port)
    connection.connect()
    connection.get_controller_version()
    setup = connection.send_input_setup(names, types)
    if setup is None:
        raise RuntimeError(f"RTDE input setup refused for recipe {names}")
    if not connection.send_start():
        raise RuntimeError("RTDE send_start() failed")
    return connection, setup


def trigger_speed_slider(connection: Any, setup: Any, *, fraction: float) -> None:
    """RR_08 S2d: ``speed_slider_mask = 1``, ``speed_slider_fraction =
    <fraction>``."""
    setup.speed_slider_mask = 1
    setup.speed_slider_fraction = fraction
    connection.send(setup)


def release_speed_slider(connection: Any, setup: Any) -> None:
    """Hand the slider back to the pendant/program default (mask 0)."""
    setup.speed_slider_mask = 0
    setup.speed_slider_fraction = 1.0
    connection.send(setup)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-ip", required=True)
    parser.add_argument("--fraction", type=float, default=0.5,
                        help="speed_slider_fraction to force (RR_08 S2d default: 0.5)")
    parser.add_argument("--delay-s", type=float, default=0.0,
                        help="wait this long after connecting before triggering -- "
                             "time it to land mid-excitation (P4 item 9)")
    parser.add_argument("--hold-s", type=float, default=5.0,
                        help="how long to hold the lowered fraction before releasing it "
                             "(expect the monitor to have already aborted by then)")
    args = parser.parse_args(argv)

    connection, setup = build_speed_slider_connection(args.robot_ip)
    try:
        if args.delay_s > 0:
            time.sleep(args.delay_s)
        trigger_speed_slider(connection, setup, fraction=args.fraction)
        time.sleep(args.hold_s)
        release_speed_slider(connection, setup)
    finally:
        connection.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
