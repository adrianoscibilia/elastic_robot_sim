"""The speed-slider sequence (RR_08 S2d, RR_10 item 2), tested against a fake
service call (no running driver in this environment)."""

import pytest

from erd_ur10.speed_slider_abort import RESET_FRACTION, run_sequence, set_speed_slider


class _FakeService:
    def __init__(self, answers=None):
        self.calls = []
        self._answers = list(answers or [])

    def __call__(self, fraction):
        self.calls.append(fraction)
        return self._answers.pop(0) if self._answers else True


def test_sequence_lowers_holds_then_resets():
    service = _FakeService()
    sleeps = []
    run_sequence(service, fraction=0.5, delay_s=2.0, hold_s=3.0, sleep=sleeps.append)
    assert service.calls == [0.5, RESET_FRACTION]
    assert sleeps == [2.0, 3.0]


def test_reset_runs_when_the_hold_is_interrupted():
    service = _FakeService()

    def interrupted(seconds):
        if seconds == 3.0:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_sequence(service, fraction=0.5, delay_s=0.0, hold_s=3.0, sleep=interrupted)
    assert service.calls == [0.5, RESET_FRACTION]


def test_refused_fraction_raises_and_still_resets():
    service = _FakeService(answers=[False, True])
    with pytest.raises(RuntimeError):
        run_sequence(service, fraction=0.5, delay_s=0.0, hold_s=1.0, sleep=lambda s: None)
    assert service.calls == [0.5, RESET_FRACTION]


def test_out_of_range_fraction_is_rejected_before_calling():
    service = _FakeService()
    with pytest.raises(ValueError):
        set_speed_slider(service, 1.5)
    assert service.calls == []
