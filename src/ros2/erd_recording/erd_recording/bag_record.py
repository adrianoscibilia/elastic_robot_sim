"""RR_16 Q-1(a): ``ros2 bag record`` with only the recorder's own logger at DEBUG.

rosbag2 0.26.11 logs every QoS message-lost event per topic, with a time and
a running total, at DEBUG on the ``rosbag2_recorder`` logger (S-31). ``ros2
bag record --log-level`` only takes a global level (its argparse ``choices``),
and a global DEBUG writes ~0.5 MB/s of rcl per-take lines (RR_15: 33 MB in
69 s). ``rosbag2_py.Recorder`` passes its level string on unchecked as
``--ros-args --log-level <string>``, and ``<logger>:=<level>`` sets one
logger's level (S-32).

So this module *is* ``ros2 bag record``: it builds the verb's own parser
(``ros2bag.verb.record.add_recorder_arguments``) and runs the verb's own
``RecordVerb.main``, which builds ``StorageOptions``/``RecordOptions`` exactly
as the CLI does. The only change is the parsed ``log_level``, replaced after
argparse by :data:`PRODUCTION_LOG_LEVEL`. ``--erd-global-log-level`` restores
the CLI's global level (diagnostic only, RR_13's ``--recorder-log-level``).

Run as ``python -m erd_recording.bag_record <ros2 bag record arguments>``.
"""

from __future__ import annotations

import re
import sys
import time
from typing import Any, Callable, Sequence

#: rcl, rmw and every other logger stay at the default INFO.
PRODUCTION_LOG_LEVEL = "rosbag2_recorder:=debug"

#: RR_13 B-2: segments start this long after rosbag2's "Recording..." line,
#: past its start-up receive stall (seen at 1.10-1.44 s).
RECORDER_STARTUP_S = 2.0
#: RR_18 R-5: every listed topic but the sidecar's must have its "Subscribed
#: to topic" line within this long, else the bag is stopped and the stage
#: fails (RR_17 F-14: the old wait warned and continued).
SUBSCRIBE_TIMEOUT_S = 15.0
#: RR_18 R-5: the sidecar's status topic appears only once the sidecar runs;
#: its "Subscribed" line is confirmed after D-6's wait for RTDE rows.
SIDECAR_STATUS_TOPIC = "/erd/rtde_status"
SIDECAR_SUBSCRIBE_TIMEOUT_S = 5.0

_RECORDING = re.compile(r"\[(\d+\.\d+)\] \[rosbag2_recorder\]: Recording\.\.\.")
_SUBSCRIBED = re.compile(r"\[rosbag2_recorder\]: Subscribed to topic '([^']+)'")


def bag_start_state(log_text: str, topics: Sequence[str], now_s: float, *,
                    startup_s: float = RECORDER_STARTUP_S) -> dict[str, Any]:
    """RR_18 R-5: has the recorder logged "Subscribed to topic" for every one
    of ``topics``, and is ``now_s`` at least ``startup_s`` past its
    "Recording..." line?"""
    started = _RECORDING.search(log_text)
    subscribed = set(_SUBSCRIBED.findall(log_text))
    missing = [topic for topic in topics if topic not in subscribed]
    recording_s = float(started.group(1)) if started else None
    return {"recording_s": recording_s, "missing": missing,
            "ready": recording_s is not None and not missing and now_s >= recording_s + startup_s}


def wait_for_bag_start(read_log: Callable[[], str], topics: Sequence[str], *, timeout_s: float,
                       alive: Callable[[], bool], startup_s: float = RECORDER_STARTUP_S,
                       clock: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep,
                       poll_s: float = 0.05) -> dict[str, Any]:
    """Poll the recorder's log until :func:`bag_start_state` is ready. ``ok``
    false (with ``reason`` and the ``missing`` topics) on the timeout or when
    the recorder exits; the caller stops the bag and fails the stage."""
    deadline = clock() + timeout_s
    while True:
        state = bag_start_state(read_log(), topics, clock(), startup_s=startup_s)
        if state["ready"]:
            return {**state, "ok": True}
        if not alive():
            return {**state, "ok": False, "reason": "the recorder exited"}
        if clock() >= deadline:
            return {**state, "ok": False, "reason": f"not ready within {timeout_s:g} s"}
        sleep(poll_s)


def recorder_log_level(global_level: str | None) -> str:
    """The level string handed to ``rosbag2_py.Recorder``."""
    return PRODUCTION_LOG_LEVEL if global_level is None else global_level


def main(argv: list[str] | None = None) -> int:
    from argparse import ArgumentParser

    from ros2bag.verb.record import RecordVerb, add_recorder_arguments

    parser = ArgumentParser(prog="erd_bag_record")
    add_recorder_arguments(parser)
    parser.add_argument("--erd-global-log-level", default=None, choices=["debug"],
                        help="diagnostic only: one global level for every logger, as `ros2 bag record --log-level`")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    args.log_level = recorder_log_level(args.erd_global_log_level)
    error = RecordVerb().main(args=args)
    if error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
