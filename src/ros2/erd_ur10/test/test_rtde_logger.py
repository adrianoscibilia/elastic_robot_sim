"""RTDE recipe flattening, tested against a fake state object (no live RTDE
server available in this environment, T1.8), and the RR_04 A-12 recorder
(gap detection, append-only parquet writing, and readability after an
unclean shutdown) -- all pure/non-ROS, so no rclpy context is needed."""

from types import SimpleNamespace

import pandas as pd
import pytest

from erd_ur10.rtde_logger import RECIPE, RTDE_PERIOD_S, RtdeRecorder, state_to_row


def _fake_state():
    values = {}
    for name, kind in RECIPE:
        if kind.startswith("VECTOR6"):
            values[name] = [float(i) for i in range(6)]
        elif kind == "DOUBLE":
            values[name] = 1.5
        else:
            values[name] = 3
    return SimpleNamespace(**values)


def test_state_to_row_flattens_vectors():
    row = state_to_row(_fake_state())
    for i in range(6):
        assert row[f"actual_q{i}"] == float(i)
        assert row[f"actual_current{i}"] == float(i)
    assert row["timestamp"] == 1.5
    assert row["robot_mode"] == 3


def test_state_to_row_has_one_column_per_recipe_entry():
    row = state_to_row(_fake_state())
    expected_columns = 0
    for _, kind in RECIPE:
        expected_columns += 6 if kind.startswith("VECTOR6") else 1
    assert len(row) == expected_columns


# ---------------------------------------------------------------------------
# RtdeRecorder (RR_04 A-12)
# ---------------------------------------------------------------------------


def _row(timestamp: float) -> dict:
    row = state_to_row(_fake_state())
    row["timestamp"] = timestamp
    return row


def test_recorder_detects_no_gap_on_a_clean_125hz_stream(tmp_path):
    recorder = RtdeRecorder(tmp_path / "rtde.parquet")
    for i in range(50):
        recorder.on_sample(_row(i * RTDE_PERIOD_S))
    assert recorder.status()["gap_count"] == 0
    assert recorder.status()["rows"] == 50


def test_recorder_detects_a_dropped_sample_gap(tmp_path):
    recorder = RtdeRecorder(tmp_path / "rtde.parquet")
    for i in range(20):
        recorder.on_sample(_row(i * RTDE_PERIOD_S))
    recorder.on_sample(_row(20 * RTDE_PERIOD_S + 5 * RTDE_PERIOD_S))  # one dropped-sample-sized jump
    for i in range(21, 40):
        recorder.on_sample(_row(i * RTDE_PERIOD_S + 5 * RTDE_PERIOD_S))
    assert recorder.status()["gap_count"] == 1


def test_recorder_flush_appends_row_groups_without_rewriting(tmp_path):
    output = tmp_path / "rtde.parquet"
    recorder = RtdeRecorder(output)
    for i in range(10):
        recorder.on_sample(_row(i * RTDE_PERIOD_S))
    recorder.flush()
    for i in range(10, 20):
        recorder.on_sample(_row(i * RTDE_PERIOD_S))
    recorder.flush()
    recorder.close()

    frame = pd.read_parquet(output)
    assert len(frame) == 20
    assert list(frame["timestamp"]) == pytest.approx([i * RTDE_PERIOD_S for i in range(20)])


def test_recorder_file_is_readable_after_an_unclean_shutdown(tmp_path):
    # "file readable after SIGKILL (all but the last row group)" (RR_04 A-12
    # acceptance): simulate a crash between two flushes by never calling
    # close() -- only the flushed row groups must survive, and the file must
    # still open cleanly (a ParquetWriter that never got its footer written
    # would leave a corrupt file; flush() closes each row group as it goes).
    output = tmp_path / "rtde.parquet"
    recorder = RtdeRecorder(output)
    for i in range(5):
        recorder.on_sample(_row(i * RTDE_PERIOD_S))
    recorder.flush()
    for i in range(5, 8):
        recorder.on_sample(_row(i * RTDE_PERIOD_S))  # never flushed or closed
    del recorder  # no close(): the "crash"

    frame = pd.read_parquet(output)
    assert len(frame) == 5
