"""RR_08 S2d / RR_10 item 2: lower the UR speed slider mid-run through the
driver's own ``/io_and_status_controller/set_speed_slider`` service
(``ur_msgs/srv/SetSpeedSliderFraction``) -- scripting the live monitor's
speed-scaling abort trigger (P4 item 9) instead of operating the pendant by
hand.

Why the driver's service and not a separate RTDE connection (S-24): the
driver's own RTDE connection already owns the ``speed_slider_mask`` /
``speed_slider_fraction`` inputs, so a second RTDE client's input setup comes
back ``IN_USE`` and is refused. The driver forwards the service value on its
connection instead.

The monitor reads the result back as ``speed_scaling``
(``/speed_scaling_state_broadcaster/speed_scaling``, normalized 0-100 -> 0-1
in ``pipeline.RecordingNode._on_speed_scaling``) and aborts once it drops
below ``safety.SPEED_SCALING_FLOOR`` (0.999) -- ``--fraction 0.5`` (the
architect's own worked example) is comfortably below that. The slider is
always reset to 1.0 at the end, also on Ctrl-C, so the next run starts at
full speed.
"""

from __future__ import annotations

import argparse
import time
from typing import Any, Callable

DEFAULT_SERVICE = "/io_and_status_controller/set_speed_slider"
RESET_FRACTION = 1.0


def set_speed_slider(call: Callable[[float], bool], fraction: float) -> None:
    """Send ``fraction`` through ``call`` (one service round trip) and raise
    if the driver reports failure. ``call`` is injected so the sequence is
    testable without a running driver."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"speed slider fraction must be in [0, 1], got {fraction}")
    if not call(fraction):
        raise RuntimeError(f"driver refused speed_slider_fraction={fraction}")


def run_sequence(call: Callable[[float], bool], *, fraction: float, delay_s: float,
                 hold_s: float, sleep: Callable[[float], None] = time.sleep) -> None:
    """Wait ``delay_s``, lower the slider to ``fraction``, hold ``hold_s``,
    then reset it to 1.0 -- the reset runs even if the hold is interrupted."""
    if delay_s > 0:
        sleep(delay_s)
    try:
        set_speed_slider(call, fraction)
        sleep(hold_s)
    finally:
        set_speed_slider(call, RESET_FRACTION)


def make_service_call(node: Any, service: str, timeout_s: float) -> Callable[[float], bool]:
    """Return a blocking ``fraction -> success`` callable over ``service``."""
    import rclpy
    from ur_msgs.srv import SetSpeedSliderFraction

    client = node.create_client(SetSpeedSliderFraction, service)
    if not client.wait_for_service(timeout_sec=timeout_s):
        raise RuntimeError(f"{service} not available after {timeout_s} s "
                           "(is ur10.launch.py running with io_and_status_controller active?)")

    def call(fraction: float) -> bool:
        request = SetSpeedSliderFraction.Request()
        request.speed_slider_fraction = float(fraction)
        future = client.call_async(request)
        rclpy.spin_until_future_complete(node, future, timeout_sec=timeout_s)
        if not future.done() or future.result() is None:
            raise RuntimeError(f"{service} did not answer within {timeout_s} s")
        return bool(future.result().success)

    return call


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fraction", type=float, default=0.5,
                        help="speed_slider_fraction to force (RR_08 S2d default: 0.5)")
    parser.add_argument("--delay-s", type=float, default=0.0,
                        help="wait this long before lowering the slider -- time it to "
                             "land mid-excitation (P4 item 9)")
    parser.add_argument("--hold-s", type=float, default=5.0,
                        help="how long to hold the lowered fraction before resetting to "
                             "1.0 (expect the monitor to have already aborted by then)")
    parser.add_argument("--service", default=DEFAULT_SERVICE)
    parser.add_argument("--timeout-s", type=float, default=5.0,
                        help="service discovery and per-call timeout")
    args = parser.parse_args(argv)

    import rclpy

    rclpy.init()
    node = rclpy.create_node("erd_speed_slider_abort")
    try:
        call = make_service_call(node, args.service, args.timeout_s)
        run_sequence(call, fraction=args.fraction, delay_s=args.delay_s, hold_s=args.hold_s)
        node.get_logger().info(
            f"speed slider held at {args.fraction} for {args.hold_s} s, reset to {RESET_FRACTION}")
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
