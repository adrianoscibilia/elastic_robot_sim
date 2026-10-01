"""RTDE sidecar logger: a second RTDE connection recording the full CB3
recipe at 125 Hz (RR_01 S3.4). RTDE allows several read-only clients, so this
runs alongside the driver's own connection without disturbing it.

RR_04 A-12 rewrite of the pass-1 version, which polled a *blocking*
``receive()`` from a 125 Hz ROS timer -- falling behind under any scheduling
jitter, since a blocking call inside a timer callback stalls every other
callback for as long as it takes -- and re-read/rewrote the whole parquet
file on every flush (O(n^2) over a run, and the file is lost if the process
dies mid-write). Now a dedicated reader thread loops on the blocking
``receive()`` on its own (nothing else contends for its time slice), and a
``pyarrow.parquet.ParquetWriter`` appends one row group per flush instead of
rewriting.

The recorder logic (:class:`RtdeRecorder`) is a plain, non-ROS class so the
gap-detection and flush behaviour are unit-tested without a live RTDE server
or an ``rclpy`` context (``test/test_rtde_logger.py``); :class:`RtdeLoggerNode`
is the thin ROS wrapper the orchestrator actually runs, one subprocess per
run (RR_04 A-12: "the orchestrator starts and stops it ... like the bag";
the launch file no longer starts it).
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

#: RR_01 S1.2 table + S2 "extra" columns, at 125 Hz.
RECIPE: tuple[tuple[str, str], ...] = (
    ("timestamp", "DOUBLE"),
    ("actual_q", "VECTOR6D"),
    ("actual_qd", "VECTOR6D"),
    ("target_q", "VECTOR6D"),
    ("target_qd", "VECTOR6D"),
    ("target_qdd", "VECTOR6D"),
    ("target_current", "VECTOR6D"),
    ("target_moment", "VECTOR6D"),
    ("joint_control_output", "VECTOR6D"),
    ("actual_current", "VECTOR6D"),
    ("actual_current_window", "VECTOR6D"),
    ("actual_TCP_force", "VECTOR6D"),
    ("actual_joint_voltage", "VECTOR6D"),
    ("joint_temperatures", "VECTOR6D"),
    ("joint_mode", "VECTOR6INT32"),
    ("speed_scaling", "DOUBLE"),
    ("target_speed_fraction", "DOUBLE"),
    ("robot_mode", "INT32"),
    ("safety_status", "INT32"),
    ("runtime_state", "UINT32"),
    ("actual_execution_time", "DOUBLE"),
)

#: RTDE's nominal sample period on CB3 (125 Hz) and RR_04 A-12's gap
#: tolerance around it ("Delta != 8 ms +/- 0.5 ms").
RTDE_PERIOD_S = 1.0 / 125.0
RTDE_PERIOD_TOLERANCE_S = 0.0005
#: Flush roughly once a second (RR_01 S3.4: "1 s row groups").
FLUSH_EVERY_N_ROWS = 125


def build_rtde_connection(host: str, port: int = 30004):
    """Return a connected, started ``rtde.RTDE`` client for :data:`RECIPE`.

    Split out from :func:`run` so a test can monkeypatch ``rtde.RTDE`` with a
    fake connection instead of touching a real network socket.
    """
    from rtde import rtde as rtde_module

    names = [name for name, _ in RECIPE]
    types = [kind for _, kind in RECIPE]
    connection = rtde_module.RTDE(host, port)
    connection.connect()
    connection.get_controller_version()
    if not connection.send_output_setup(names, types, frequency=125):
        raise RuntimeError(f"RTDE output setup refused for recipe {names}")
    if not connection.send_start():
        raise RuntimeError("RTDE send_start() failed")
    return connection


def state_to_row(state: Any) -> dict[str, Any]:
    """Flatten one RTDE ``state`` sample into a parquet-friendly row.

    ``VECTORnD``/``VECTORnINT32`` fields become ``<name>0..<name>{n-1}``, so
    the sidecar file matches the flat-column convention every other
    ``erd_recording`` dataframe uses.
    """
    row: dict[str, Any] = {}
    for name, kind in RECIPE:
        value = getattr(state, name)
        if kind.startswith("VECTOR"):
            for index, item in enumerate(value):
                row[f"{name}{index}"] = item
        else:
            row[name] = value
    return row


class RtdeRecorder:
    """Non-ROS: gap detection + append-only parquet writing for one RTDE
    session (RR_04 A-12). Thread-safe for the single-producer
    (:meth:`on_sample`, from the reader thread) / single-consumer
    (:meth:`flush`, from the timer/main thread) pattern
    :class:`RtdeLoggerNode` uses.

    ``output_path`` is a **directory** (e.g. ``<run>/rtde.parquet/``), not a
    single file: each :meth:`flush` writes whatever accumulated since the
    last one as its own complete, independently-readable parquet part file
    (``part-00000.parquet``, ...), so ``pandas.read_parquet(output_path)``
    (which reads a directory of parquet files as one dataset) always returns
    every *completed* row group -- immediately after each flush, and
    unchanged by a crash before the next one -- without ever re-reading or
    rewriting earlier data (the pass-1 version's O(n^2) full-file rewrite,
    RR_04 A-12). A single ever-open ``pyarrow.parquet.ParquetWriter`` cannot
    give this property: its footer, which is what makes the file readable at
    all, is only written on an explicit, clean :meth:`close`.
    """

    def __init__(self, output_path: str | Path):
        self._output_path = Path(output_path)
        self._lock = threading.Lock()
        self._pending_rows: list[dict[str, Any]] = []
        self._total_rows = 0
        self._last_timestamp: float | None = None
        self._gap_count = 0
        self._next_part = 0

    def on_sample(self, row: dict[str, Any]) -> None:
        """Called from the reader thread for every received RTDE sample."""
        timestamp = row.get("timestamp")
        with self._lock:
            if self._last_timestamp is not None and timestamp is not None:
                delta = timestamp - self._last_timestamp
                if abs(delta - RTDE_PERIOD_S) > RTDE_PERIOD_TOLERANCE_S:
                    self._gap_count += 1
            self._last_timestamp = timestamp
            self._pending_rows.append(row)
            self._total_rows += 1

    def flush(self) -> None:
        """Write whatever has accumulated since the last flush as one new,
        already-complete parquet part file. Safe to call on an empty buffer
        (a no-op)."""
        with self._lock:
            rows, self._pending_rows = self._pending_rows, []
            part_index = self._next_part
            self._next_part += 1
        if not rows:
            return
        self._output_path.mkdir(parents=True, exist_ok=True)
        part_path = self._output_path / f"part-{part_index:05d}.parquet"
        table = pa.Table.from_pandas(pd.DataFrame(rows), preserve_index=False)
        pq.write_table(table, part_path)

    def close(self) -> None:
        self.flush()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {"rows": self._total_rows, "last_timestamp": self._last_timestamp, "gap_count": self._gap_count}


class RtdeLoggerNode(Node):
    def __init__(self) -> None:
        super().__init__("erd_rtde_logger")
        self.declare_parameter("robot_ip", "192.168.56.10")
        self.declare_parameter("frequency", 125.0)
        self.declare_parameter("output_path", "/tmp/erd_rtde.parquet")
        self._status_pub = self.create_publisher(String, "/erd/rtde_status", 10)
        self._recorder = RtdeRecorder(self.get_parameter("output_path").value)
        self._connection = build_rtde_connection(self.get_parameter("robot_ip").value, port=30004)
        self._keep_running = True
        self._rows_since_flush = 0
        # RR_04 A-12: a dedicated thread blocking on receive() -- nothing else
        # contends for its time slice, unlike the pass-1 125 Hz timer that
        # polled a blocking receive() and fell behind under jitter.
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()
        self.create_timer(1.0, self._publish_status)
        self.create_timer(1.0, self._recorder.flush)

    def _reader_loop(self) -> None:
        while self._keep_running:
            try:
                state = self._connection.receive(binary=False)
            except Exception:
                if not self._keep_running:
                    return
                time.sleep(0.01)
                continue
            if state is None:
                continue
            self._recorder.on_sample(state_to_row(state))

    def _publish_status(self) -> None:
        msg = String()
        msg.data = json.dumps(self._recorder.status())
        self._status_pub.publish(msg)

    def destroy_node(self) -> bool:
        self._keep_running = False
        self._recorder.close()
        try:
            self._connection.send_pause()
            self._connection.disconnect()
        except Exception:
            pass
        return super().destroy_node()


def main(argv: list[str] | None = None) -> int:
    rclpy.init(args=argv)
    node = RtdeLoggerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
