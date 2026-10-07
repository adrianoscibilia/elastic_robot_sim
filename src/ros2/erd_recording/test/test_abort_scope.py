"""RR_14 P-3: the rung-4a tracking override acts inside excitation segments
only, and `report` suggests its value from a `--ladder 0.1` run."""

from __future__ import annotations

import numpy as np
import pytest

from erd_recording.profile import IIWA_PROFILE, UR10_PROFILE
from erd_recording.reporting import excitation_windows, max_tracking_error, suggest_abort_tracking_rad
from erd_recording.safety import EXCITATION_KINDS, segment_tracking_rad

FILE_RAD = 0.05


@pytest.mark.parametrize("kind", sorted(EXCITATION_KINDS))
def test_override_applies_inside_excitations(kind):
    assert segment_tracking_rad(kind, file_rad=FILE_RAD, excitation_override=0.0006) == 0.0006


@pytest.mark.parametrize("kind", ["approach_commissioning", "return_commissioning", "approach", "return",
                                  "standstill_approach", "identify_approach", None])
def test_approaches_and_returns_keep_the_file_value(kind):
    # rr13_a4 cancelled in `excitation_0_ladder_approach` (approach_commissioning)
    assert segment_tracking_rad(kind, file_rad=FILE_RAD, excitation_override=0.0006) == FILE_RAD


def test_no_override_means_the_file_value_everywhere():
    for kind in sorted(EXCITATION_KINDS) + ["approach"]:
        assert segment_tracking_rad(kind, file_rad=FILE_RAD, excitation_override=None) == FILE_RAD


def _events():
    return [{"segment_id": "excitation_0_ladder_approach", "kind": "approach_commissioning_start", "stamp_ns": 0},
            {"segment_id": "excitation_0_ladder_approach", "kind": "approach_commissioning_end", "stamp_ns": 100},
            {"segment_id": "excitation_0_ladder", "kind": "excitation_commissioning_start", "stamp_ns": 200},
            {"segment_id": "excitation_0_ladder", "kind": "excitation_commissioning_end", "stamp_ns": 300}]


def test_suggestion_uses_excitation_windows_only():
    windows = excitation_windows(_events())
    assert windows == [("excitation_0_ladder", 200, 300)]
    stamps = np.arange(0, 400, 10)
    error = np.where(stamps < 150, 0.02, 0.00125)  # large in the approach, 1.25 mrad in the excitation
    worst = max_tracking_error(stamps, error, windows)
    assert worst == {"max_abs_error_rad": 0.00125, "segment": "excitation_0_ladder"}
    suggestion = suggest_abort_tracking_rad(worst["max_abs_error_rad"], IIWA_PROFILE.position_count_rad)
    assert suggestion["suggested_rad"] == pytest.approx(0.000625)
    assert not suggestion["floored"]


def test_suggestion_is_floored_at_four_counts():
    suggestion = suggest_abort_tracking_rad(1e-7, UR10_PROFILE.position_count_rad)
    assert suggestion["floored"]
    assert suggestion["suggested_rad"] == pytest.approx(4 * 4.79e-7)
    assert IIWA_PROFILE.position_count_rad == 5.989e-8
