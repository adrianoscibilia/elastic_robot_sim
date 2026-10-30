"""Zollo's preprocessing of one bag: filter, differentiate, parallel filter, decimate.

Zollo et al. low-pass the measured positions (5th-order Butterworth, 20 Hz
at 1 kHz), differentiate them by first-order differences, and discard the
samples where the joint moves slower than 0.5 rad/s.  Here:

* the filter is zero-phase (``sosfiltfilt``), so differentiated signals and
  the target stay in phase -- the deflection ``theta - q`` is milliradians and
  a phase lag of either side would swamp it;
* the cutoff is a parameter, defaulting to the dataset's own probe band
  (`00_recon.md` Sec 2: 20 Hz would cut the arms' elastic modes);
* ``parallel_filter`` applies the same filter to every regressor column and to
  the target before decimation (Gautier's parallel filtering), so the rows
  stay consistent after the nonlinear regressor;
* the low-speed threshold is per joint, a fraction of the joint's largest
  speed in the bag, and drops only the rows that carry a friction column.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Mapping

import numpy as np
from scipy.signal import butter, sosfiltfilt

DIFFERENCES = ("central", "forward")


@dataclass(frozen=True)
class Preprocessing:
    cutoff_hz: float | None
    order: int = 5
    difference: str = "central"
    parallel_filter: bool = True
    decimate: int = 1
    edge_s: float = 0.05
    low_speed_fraction: float = 0.0

    def __post_init__(self) -> None:
        if self.difference not in DIFFERENCES:
            raise ValueError(f"difference must be one of {DIFFERENCES}")
        if self.decimate < 1:
            raise ValueError("decimate must be >= 1")
        if not 0.0 <= self.low_speed_fraction < 1.0:
            raise ValueError("low_speed_fraction must be in [0, 1)")

    def describe(self) -> dict[str, Any]:
        return asdict(self)

    def filter_for(self, rate_hz: float) -> np.ndarray | None:
        """Second-order sections of the low-pass at ``rate_hz``, or ``None`` (no filter)."""
        if self.cutoff_hz is None:
            return None
        if not 0.0 < self.cutoff_hz < 0.5 * rate_hz:
            raise ValueError(f"cutoff {self.cutoff_hz} Hz is outside (0, Nyquist = {0.5 * rate_hz} Hz)")
        return butter(self.order, self.cutoff_hz, btype="low", fs=rate_hz, output="sos")


def lowpass(x: np.ndarray, sos: np.ndarray | None) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x.copy() if sos is None else sosfiltfilt(sos, x, axis=0)


def derivative(x: np.ndarray, step: float, kind: str = "central") -> np.ndarray:
    """``dx/dt`` along axis 0: central (second order, zero phase) or forward (Zollo's) differences."""
    x = np.asarray(x, dtype=float)
    if kind == "central":
        return np.gradient(x, step, axis=0)
    out = np.empty_like(x)
    out[:-1] = np.diff(x, axis=0) / step
    out[-1] = out[-2]
    return out


def prepare(raw: Mapping[str, np.ndarray], step: float, pre: Preprocessing) -> dict[str, np.ndarray]:
    """Filtered positions and their derivatives; torques passed through.

    ``raw`` holds ``theta`` and ``q`` (positions, ``[T, n]``) and the torques
    ``tau_J`` and optionally ``u``.  Velocities and accelerations are always
    derived from the filtered positions (Zollo), on both sides alike.
    """
    sos = pre.filter_for(1.0 / step)
    out: dict[str, np.ndarray] = {}
    for side, (p, v, a) in {"theta": ("theta", "dtheta", "ddtheta"), "q": ("q", "dq", "ddq")}.items():
        position = lowpass(raw[side], sos)
        velocity = derivative(position, step, pre.difference)
        out[p], out[v], out[a] = position, velocity, derivative(velocity, step, pre.difference)
    for name in ("tau_J", "u"):
        if name in raw and raw[name] is not None:
            value = np.asarray(raw[name], dtype=float)
            out[name] = value if pre.parallel_filter else lowpass(value, sos)
    return out


def parallel(W: np.ndarray, target: np.ndarray, step: float, pre: Preprocessing) -> tuple[np.ndarray, np.ndarray]:
    """The same low-pass on every regressor column and on the target (time on axis 0)."""
    if not pre.parallel_filter:
        return W, target
    sos = pre.filter_for(1.0 / step)
    return lowpass(W, sos), lowpass(target, sos)


def kept_samples(T: int, step: float, pre: Preprocessing) -> np.ndarray:
    """Indices left after trimming ``edge_s`` at both ends (filter transients) and decimating."""
    edge = int(round(pre.edge_s / step))
    if T <= 2 * edge:
        return np.zeros(0, dtype=int)
    return np.arange(edge, T - edge, pre.decimate)


def low_speed_mask(dtheta: np.ndarray, pre: Preprocessing) -> np.ndarray:
    """``[T, n]``: True where the joint moves fast enough to keep its friction rows."""
    speed = np.abs(np.asarray(dtheta, dtype=float))
    if pre.low_speed_fraction <= 0.0:
        return np.ones_like(speed, dtype=bool)
    return speed >= pre.low_speed_fraction * speed.max(axis=0, keepdims=True)
