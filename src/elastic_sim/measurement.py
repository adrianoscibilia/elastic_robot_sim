"""Sensor model applied to a recorded rollout (round 5).

Round-4 datasets record the simulator's own state and the exact applied
torque: no quantization, no noise, no gain error, no delay (``R5_00`` Sec 1).
The legacy 3-DoF platform's generator did model all four (``R5_00`` Sec 3), and
that is one of the two ways it was more realistic than the modern stack.

This module puts that back, generically and per bag:

* **encoder quantization** -- positions are read as integer counts;
* **additive noise** on position, velocity and torque, the torque noise
  proportional to the reading plus a floor, as a current-derived torque
  estimate behaves;
* **gain error** -- one constant per joint per bag, drawn once: a torque read
  from motor current carries a calibration error that does not average out,
  and a model trained on a single gain would never see it vary;
* **delay** -- whole-sample transport delay of the measured channels relative
  to the commanded torque, the cheapest faithful stand-in for a fieldbus
  round trip plus the drive's own filtering.

Two rules the whole design follows:

1. **The label stays exact internally.** Noise is applied to what is
   *recorded*; the clean signal is kept beside it (``*_clean`` columns, off by
   default) so that any diagnostic can still decompose the error into physics
   and measurement (``R5_00`` Sec 6.2, Q-C.3).
2. **Motor and link channels are separate instruments.** A gain error on the
   drive's current-derived torque is not the same number as the one on a
   link-side torque sensor, so each channel draws its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MeasurementModel:
    """What the recorder sees instead of the simulator's state.

    Every field defaults to the perfect instrument, so
    :meth:`MeasurementModel.is_ideal` is true for any config that does not
    configure one and the recorded columns are then bit-identical to round 4's.

    Units follow the asset: rad / rad s^-1 / Nm on a revolute robot, m /
    m s^-1 / N on the prismatic FMRR gantry.  ``encoder_resolution`` is the
    size of one count, not a bit depth.
    """

    encoder_resolution: float = 0.0
    q_noise: float = 0.0
    dq_noise: float = 0.0
    tau_noise_rel: float = 0.0
    tau_noise_abs: float = 0.0
    tau_gain_error: float = 0.0
    delay_samples: int = 0

    def __post_init__(self) -> None:
        for name in ("encoder_resolution", "q_noise", "dq_noise", "tau_noise_rel",
                     "tau_noise_abs", "tau_gain_error"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"simulation.measurement.{name} must be non-negative")
        if int(self.delay_samples) < 0:
            raise ValueError("simulation.measurement.delay_samples must be non-negative")
        if float(self.tau_gain_error) >= 1.0:
            raise ValueError(
                "simulation.measurement.tau_gain_error is a relative half-width and must be < 1 "
                "(0.05 means every channel's gain is drawn from [0.95, 1.05])"
            )
        object.__setattr__(self, "delay_samples", int(self.delay_samples))

    @property
    def is_ideal(self) -> bool:
        return (
            self.encoder_resolution == 0.0 and self.q_noise == 0.0 and self.dq_noise == 0.0
            and self.tau_noise_rel == 0.0 and self.tau_noise_abs == 0.0
            and self.tau_gain_error == 0.0 and self.delay_samples == 0
        )

    # -- individual channels -------------------------------------------------

    def quantize(self, position: np.ndarray) -> np.ndarray:
        """Round a position to the encoder grid."""
        if self.encoder_resolution == 0.0:
            return np.asarray(position, dtype=float)
        step = float(self.encoder_resolution)
        return np.round(np.asarray(position, dtype=float) / step) * step

    def measure_position(self, position: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Quantize *after* adding noise, the order a real encoder imposes.

        The quantizer is the last thing in the chain on real hardware: the
        count it reports is already the noisy angle rounded, so a position
        read at rest is a constant count, not a count dithered by noise.  The
        reverse order would leak sub-count noise into a stationary reading and
        make the quantization invisible.
        """
        position = np.asarray(position, dtype=float)
        if self.q_noise > 0.0:
            position = position + rng.normal(0.0, self.q_noise, size=position.shape)
        return self.quantize(position)

    def measure_velocity(self, velocity: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        velocity = np.asarray(velocity, dtype=float)
        if self.dq_noise > 0.0:
            velocity = velocity + rng.normal(0.0, self.dq_noise, size=velocity.shape)
        return velocity

    def measure_torque(
        self, torque: np.ndarray, rng: np.random.Generator, *, gain: np.ndarray | None = None,
    ) -> np.ndarray:
        """Scale by a per-joint gain, then add proportional plus floor noise."""
        torque = np.asarray(torque, dtype=float)
        if gain is not None:
            torque = torque * np.asarray(gain, dtype=float)[None, :]
        scale = self.tau_noise_rel * np.abs(torque) + self.tau_noise_abs
        if np.any(scale > 0.0):
            torque = torque + rng.normal(0.0, 1.0, size=torque.shape) * scale
        return torque

    def sample_gain(self, rng: np.random.Generator, n_dof: int) -> np.ndarray:
        """One constant gain per joint, drawn once per bag and per channel."""
        if self.tau_gain_error == 0.0:
            return np.ones(n_dof)
        return 1.0 + rng.uniform(-self.tau_gain_error, self.tau_gain_error, size=n_dof)

    def delay(self, values: np.ndarray) -> np.ndarray:
        """Shift a channel later in time by ``delay_samples``, holding row 0.

        The first rows are filled with the first sample rather than with NaN or
        with a truncation: every bag must keep its declared length and its
        uniform time step (``generate()`` asserts both), and a transient at the
        start of a bag is what a real recorder's first samples are anyway.
        """
        values = np.asarray(values, dtype=float)
        if self.delay_samples == 0:
            return values
        shift = min(self.delay_samples, len(values))
        delayed = np.empty_like(values)
        delayed[:shift] = values[0]
        delayed[shift:] = values[: len(values) - shift]
        return delayed

    def describe(self) -> dict[str, Any]:
        return {
            "encoder_resolution": float(self.encoder_resolution),
            "q_noise": float(self.q_noise),
            "dq_noise": float(self.dq_noise),
            "tau_noise_rel": float(self.tau_noise_rel),
            "tau_noise_abs": float(self.tau_noise_abs),
            "tau_gain_error": float(self.tau_gain_error),
            "delay_samples": int(self.delay_samples),
        }


#: A perfect instrument: what every round-4 config gets.
IDEAL_MEASUREMENT = MeasurementModel()


@dataclass(frozen=True)
class MeasuredSignals:
    """One bag's recorded channels, after the sensor model.

    ``*_clean`` fields are the simulator's own values on the same grid, kept so
    that a diagnostic can separate physics from instrument (``R5_00`` Q-C.3)
    and so that the exact label survives in the file when a config asks for it.
    """

    q_motor: np.ndarray
    dq_motor: np.ndarray
    q_link: np.ndarray
    dq_link: np.ndarray
    tau_motor: np.ndarray
    tau_link: np.ndarray
    gain_motor: np.ndarray
    gain_link: np.ndarray


def measure_bag(
    model: MeasurementModel,
    *,
    q_motor: np.ndarray,
    dq_motor: np.ndarray,
    q_link: np.ndarray,
    dq_link: np.ndarray,
    tau_motor: np.ndarray,
    tau_link: np.ndarray,
    seed: int,
) -> MeasuredSignals:
    """Apply ``model`` to one bag's resampled channels.

    ``seed`` keys the bag's own RNG stream, so the same bag measured twice
    gives the same numbers and a bag's noise does not depend on how many bags
    ran before it -- the same property every other draw in this pipeline has,
    and what makes a parallel build reproducible.

    The delay is applied to the *measured* channels only: ``tau_motor`` here is
    the controller's own command, which the recorder timestamps when it is
    sent, while positions, velocities and the link-side torque come back
    through the bus.  Delaying the command as well would just shift the whole
    bag and change nothing.
    """
    n_dof = int(np.asarray(q_motor).shape[1])
    if model.is_ideal:
        ones = np.ones(n_dof)
        return MeasuredSignals(
            q_motor=np.asarray(q_motor, dtype=float), dq_motor=np.asarray(dq_motor, dtype=float),
            q_link=np.asarray(q_link, dtype=float), dq_link=np.asarray(dq_link, dtype=float),
            tau_motor=np.asarray(tau_motor, dtype=float), tau_link=np.asarray(tau_link, dtype=float),
            gain_motor=ones, gain_link=ones,
        )
    rng = np.random.default_rng((int(seed), 8))
    # Gains first, and one draw per channel: the motor's current-derived
    # torque and a link-side torque sensor are different instruments.
    gain_motor = model.sample_gain(rng, n_dof)
    gain_link = model.sample_gain(rng, n_dof)
    return MeasuredSignals(
        q_motor=model.delay(model.measure_position(q_motor, rng)),
        dq_motor=model.delay(model.measure_velocity(dq_motor, rng)),
        q_link=model.delay(model.measure_position(q_link, rng)),
        dq_link=model.delay(model.measure_velocity(dq_link, rng)),
        # The command is recorded as it is issued, so it is not delayed.
        tau_motor=model.measure_torque(tau_motor, rng, gain=gain_motor),
        tau_link=model.delay(model.measure_torque(tau_link, rng, gain=gain_link)),
        gain_motor=gain_motor, gain_link=gain_link,
    )
