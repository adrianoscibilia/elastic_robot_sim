"""RR_12 A-2a: the link-test statistics, on synthetic streams."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from erd_recording.link_test import iiwa_link_stats, stamp_step_histogram, ur_link_stats


def _iiwa_frame(n: int, *, lost_fri=(), lost_publication=()) -> pd.DataFrame:
    cycles = np.arange(n, dtype=float)
    keep = np.ones(n, dtype=bool)
    keep[list(lost_fri)] = False  # never received: both counters skip
    cycles = cycles[keep]
    received = np.arange(len(cycles), dtype=float)
    stamps = (np.arange(n, dtype=np.int64) * 1_000_000)[keep]
    drop = np.ones(len(cycles), dtype=bool)
    drop[list(lost_publication)] = False  # read, but never published: only `cycle` and `received` jump together
    return pd.DataFrame({"stamp_ns": stamps[drop], "cycle": cycles[drop], "received_cycle": received[drop],
                         "connection_quality": 3.0, "session_state": 4.0})


def test_lost_fri_and_publication_cycles_are_separated():
    stats = iiwa_link_stats(_iiwa_frame(300_000, lost_fri=(1000, 1001, 50_000), lost_publication=(2000,)))
    assert stats["lost_fri_cycles"] == 3
    assert stats["lost_publication_cycles"] == 1
    assert stats["pass_rung1"] is True
    assert stats["expected_valid_12s_window_fraction"] == math.exp(-12 * 3 / stats["duration_s"])
    assert stats["connection_quality"] == {"3": len(_iiwa_frame(300_000, lost_fri=(1000, 1001, 50_000),
                                                                lost_publication=(2000,)))}


def test_rung1_fails_above_five_lost_cycles_per_300_s():
    stats = iiwa_link_stats(_iiwa_frame(300_000, lost_fri=tuple(range(10_000, 300_000, 40_000))))
    assert stats["lost_fri_cycles"] == 8 and stats["pass_rung1"] is False


def test_stamp_step_histogram_counts_every_step():
    stamps = np.cumsum([0] + [1_000_000] * 98 + [4_000_000]).astype(np.int64)
    histogram = stamp_step_histogram(stamps)
    assert histogram["n"] == 99 and sum(histogram["histogram_ms"].values()) == 99
    assert histogram["histogram_ms"]["3-5"] == 1 and histogram["max_ms"] == 4.0


def test_ur_link_stats_gaps_and_exact_match():
    times = np.arange(0, 10, 0.008)
    times = np.delete(times, [100, 101])  # two missing sidecar cycles
    q = np.column_stack([np.round(np.sin(times + j), 6) for j in range(6)])
    sidecar = pd.DataFrame({"timestamp": times, **{f"actual_q{j}": q[:, j] for j in range(6)}})
    driver_q = np.vstack([q[:-10], q[-10:] + 1e-6])  # the last 10 rows don't match bit-exactly
    stats = ur_link_stats(sidecar, driver_q, (times * 1e9).astype(np.int64))
    assert stats["sidecar_gaps"] == 1 and stats["sidecar_missing_cycles"] == 2
    assert stats["exact_actual_q_match_fraction"] == 1 - 10 / len(times)
    assert abs(stats["sidecar_rate_hz"] - 125.0) < 0.5  # 2 of 1250 rows missing
