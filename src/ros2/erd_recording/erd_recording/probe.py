"""The real probe comb: the training comb, capped and moved onto the lab's
base-frequency grid (RR_20 T1/T2).

One rule for the loader and the planner. The training config gives harmonic
*indices* of its own base (0.1 Hz for every r6/r7 config); the lab plays
them as indices of ``excitation.base_frequency`` (0.2 Hz on the B files).
So each line is turned back into a frequency, capped, and snapped to the
nearest lab harmonic: ``h' = round(h * base_train / base_lab)``. Lines that
collide after snapping are kept once, and lines at or below the main
trajectory's harmonics are dropped. On the A base (lab base = training
base) the result equals the old capped index list.

Nothing here imports pinocchio at module level: the loader runs inside the
ROS pipeline process (env_guard), so it can only use :func:`resolve_probe_comb`
with ``resolve_design=False``. A training config with a ``probe_design``
needs ``elastic_sim.link_modes`` (and so pinocchio) to resolve; the loader
skips it and ``plan`` checks it the same way.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

#: SG resolvability bound (RR_01 S4 "Probe rule"), the same fraction dataset_config
#: (elastic_sim) documents as `0.09 * rate` (the order-3 SG differentiator's floor).
PROBE_SG_BOUND_FRACTION = 0.09


@dataclass(frozen=True)
class ProbeComb:
    """A resolved probe comb on the lab's base grid."""

    harmonics: tuple[int, ...]
    base_frequency: float
    bound_hz: float
    training_base_frequency: float
    #: the training config's comb, before the cap and the remap
    training_harmonics: tuple[int, ...]

    @property
    def lines_hz(self) -> tuple[float, ...]:
        return tuple(h * self.base_frequency for h in self.harmonics)

    @property
    def top_hz(self) -> float:
        return self.harmonics[-1] * self.base_frequency if self.harmonics else 0.0


def probe_bound_hz(max_top_hz: float | None, rate_hz: float) -> float:
    """``min(0.09 * rate, max_top_hz)``: the highest line the real probe may carry."""
    bound = PROBE_SG_BOUND_FRACTION * rate_hz
    return bound if max_top_hz is None else min(bound, float(max_top_hz))


def remap_probe_harmonics(training_harmonics: Sequence[int], training_base: float, lab_base: float,
                          n_harmonics: int, bound_hz: float) -> tuple[int, ...]:
    """Training harmonic indices -> lab harmonic indices, capped at ``bound_hz``."""
    remapped: set[int] = set()
    for h in training_harmonics:
        frequency = h * training_base
        if frequency > bound_hz:
            continue
        index = int(round(frequency / lab_base))
        if index > n_harmonics and index * lab_base <= bound_hz * (1.0 + 1e-9):
            remapped.add(index)
    return tuple(sorted(remapped))


def resolve_probe_comb(probe: Any, *, base_frequency: float, n_harmonics: int, rate_hz: float,
                       resolve_design: bool = True) -> ProbeComb | None:
    """The comb ``probe`` (a :class:`~erd_recording.config.ProbeSpec`) plays at the lab base.

    ``None`` when the probe is off, or when the training config has a
    ``probe_design`` and ``resolve_design`` is false (see the module doc).
    """
    if not probe.enabled or not probe.from_training_config:
        return None
    from elastic_sim.dataset_config import load_config

    training = load_config(Path(probe.from_training_config).expanduser())
    if training.probe_design is not None:
        if not resolve_design:
            return None
        from elastic_sim.link_modes import resolve_probe_design

        training = resolve_probe_design(training)
    training_harmonics = tuple(int(h) for h in training.excitation.probe_harmonics)
    training_base = float(training.excitation.base_frequency)
    bound = probe_bound_hz(probe.max_top_hz, rate_hz)
    return ProbeComb(
        harmonics=remap_probe_harmonics(training_harmonics, training_base, float(base_frequency),
                                        int(n_harmonics), bound),
        base_frequency=float(base_frequency), bound_hz=bound,
        training_base_frequency=training_base, training_harmonics=training_harmonics,
    )
