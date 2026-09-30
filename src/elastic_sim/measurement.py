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
    #: Noise of the *force/torque cell*, when the target comes from one
    #: (``dataset.signals.target: ee_wrench_joint``).  ``None`` means "use the
    #: ``tau_*`` values", which is what a config that does not distinguish them
    #: gets.  They need distinguishing because the two instruments are not
    #: alike: a drive's current-derived torque really is noisy in proportion to
    #: the reading, while a strain-gauge cell's noise is a resolution floor
    #: that does not grow with the load.  Applying a 3 % relative figure to a
    #: wrench channel carrying a 49 N static bias buries the signal in a noise
    #: term that bias alone creates (`R5_05` Sec 6.4 measures it).
    ft_noise_rel: float | None = None
    ft_noise_abs: float | None = None

    def __post_init__(self) -> None:
        for name in ("encoder_resolution", "q_noise", "dq_noise", "tau_noise_rel",
                     "tau_noise_abs", "tau_gain_error"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"simulation.measurement.{name} must be non-negative")
        for name in ("ft_noise_rel", "ft_noise_abs"):
            value = getattr(self, name)
            if value is not None and float(value) < 0.0:
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
            and not self.ft_noise_rel and not self.ft_noise_abs
        )

    def for_force_cell(self) -> "MeasurementModel":
        """This model with the force/torque cell's own noise figures applied.

        Returns ``self`` unchanged when the config does not distinguish them, so
        every existing config keeps exactly the behaviour it had.
        """
        if self.ft_noise_rel is None and self.ft_noise_abs is None:
            return self
        from dataclasses import replace

        return replace(
            self,
            tau_noise_rel=self.tau_noise_rel if self.ft_noise_rel is None else float(self.ft_noise_rel),
            tau_noise_abs=self.tau_noise_abs if self.ft_noise_abs is None else float(self.ft_noise_abs),
            ft_noise_rel=None, ft_noise_abs=None,
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
            "ft_noise_rel": None if self.ft_noise_rel is None else float(self.ft_noise_rel),
            "ft_noise_abs": None if self.ft_noise_abs is None else float(self.ft_noise_abs),
        }


#: The instruments a recorded channel can go through.  ``force_cell`` is the
#: 6-axis wrench of an end-effector cell, which is the same *instrument slot*
#: as a link-side torque sensor but a different transducer, so it carries the
#: cell's own noise figures (:meth:`MeasurementModel.for_force_cell`).
CHANNEL_KINDS = ("position", "velocity", "motor_torque", "link_torque", "force_cell")


@dataclass(frozen=True)
class MeasuredChannel:
    """One channel after its instrument, with the gain that produced it."""

    values: np.ndarray
    gain: np.ndarray


def measure_channel(
    model: MeasurementModel, values: np.ndarray, kind: str, seed: int,
) -> MeasuredChannel:
    """Apply ``model`` to one channel of one bag (``R5_06`` Sec 3).

    :func:`measure_bag` measures the six state channels of a rollout together,
    on one stream, so that their draws stay in a fixed order.  A channel that
    is not one of those six -- the force/torque cell's 6-axis wrench is the
    only one so far -- needs the same instrument model without pretending to
    be a joint channel, which is what this is for.  Passing a wrench into all
    six ``measure_bag`` slots happened to work because ``n_dof`` is inferred
    from the array width; it was not a contract.

    ``kind`` selects the instrument.  Which channels are delayed follows
    :func:`measure_bag`: everything that comes back over the bus is, and the
    controller's own command (``motor_torque``) is not.
    """
    if kind not in CHANNEL_KINDS:
        raise ValueError(f"measure_channel: kind must be one of {CHANNEL_KINDS}, got {kind!r}")
    values = np.atleast_2d(np.asarray(values, dtype=float))
    width = int(values.shape[1])
    if kind == "force_cell":
        model = model.for_force_cell()
    if model.is_ideal:
        return MeasuredChannel(values=values, gain=np.ones(width))
    rng = np.random.default_rng((int(seed), 9, CHANNEL_KINDS.index(kind)))
    if kind == "position":
        return MeasuredChannel(model.delay(model.measure_position(values, rng)), np.ones(width))
    if kind == "velocity":
        return MeasuredChannel(model.delay(model.measure_velocity(values, rng)), np.ones(width))
    gain = model.sample_gain(rng, width)
    measured = model.measure_torque(values, rng, gain=gain)
    if kind != "motor_torque":
        measured = model.delay(measured)
    return MeasuredChannel(values=measured, gain=gain)


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


# ---------------------------------------------------------------------------
# Round 6: percentage noise on every recorded signal (`R6_00` Sec 6)
# ---------------------------------------------------------------------------

#: The rng stream index of the round-6 instruments, one sub-stream per channel.
#: New index, so it perturbs no round-5 draw.
_NOISE_STREAM = 13
_NOISE_CHANNELS = ("q", "dq", "ft", "wrench")


def _per_joint_values(values: Any, n_dof: int, name: str) -> np.ndarray:
    array = np.atleast_1d(np.asarray(values, dtype=float)).reshape(-1)
    if len(array) not in (1, n_dof):
        raise ValueError(f"simulation.noise.{name} has {len(array)} values; expected 1 or {n_dof}")
    return np.broadcast_to(array, (n_dof,)).copy()


@dataclass(frozen=True)
class ChannelNoise:
    """One recorded channel's instrument (`R6_00` Sec 6, "Definition").

    For joint ``j`` of a bag the chain is

    1. white Gaussian noise of
       ``sigma_j = p * RMS(x_j - mean(x_j)) (+) sigma (+) sigma_quanta * quantization_j``,
       fixed for the bag (``(+)`` is a root-sum-square; the absolute floor
       ``sigma`` is the end-effector cell's, ``sigma_quanta`` the encoders':
       a percentage of RMS is the wrong convention for a position, `R6_02` Sec 3);
    2. a per-bag gain ``1 + g``, ``g ~ U(-gain, gain)``;
    3. a per-bag offset ``o ~ U(-offset, offset) * effort_j`` (torque channels);
    4. quantization to ``quantization`` (one value, or one per joint);
    5. the bag's ``delay_samples``.

    The levels are a modelling-error-class choice, not a claim about the real
    sensor (`R6_00` Sec 6, "Principle"); where an interface's resolution is
    known, the quantization is exact.
    """

    p: float = 0.0
    sigma: float = 0.0
    gain: float = 0.0
    offset: float = 0.0
    quantization: tuple[float, ...] = (0.0,)
    sigma_quanta: float = 0.0

    def __post_init__(self) -> None:
        for name in ("p", "sigma", "gain", "offset", "sigma_quanta"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"simulation.noise.*.{name} must be non-negative")
        if float(self.gain) >= 1.0:
            raise ValueError("simulation.noise.*.gain is a relative half-width and must be < 1")
        quantization = tuple(float(v) for v in np.atleast_1d(self.quantization))
        if any(v < 0.0 for v in quantization):
            raise ValueError("simulation.noise.*.quantization must be non-negative")
        object.__setattr__(self, "quantization", quantization)

    @property
    def is_ideal(self) -> bool:
        return (self.p == 0.0 and self.sigma == 0.0 and self.gain == 0.0 and self.offset == 0.0
                and self.sigma_quanta == 0.0 and all(v == 0.0 for v in self.quantization))

    def sigma_for(self, basis: np.ndarray) -> np.ndarray:
        """Per-joint noise std for a bag whose signal (or reference) is ``basis``."""
        basis = np.atleast_2d(np.asarray(basis, dtype=float))
        rms = np.sqrt(np.mean((basis - basis.mean(axis=0)) ** 2, axis=0))
        quanta = float(self.sigma_quanta) * _per_joint_values(self.quantization, basis.shape[-1], "quantization")
        return np.sqrt((float(self.p) * rms) ** 2 + float(self.sigma) ** 2 + quanta ** 2)

    def quantize(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        step = _per_joint_values(self.quantization, values.shape[-1], "quantization")
        return np.where(step > 0.0, np.round(values / np.where(step > 0.0, step, 1.0)) * step, values)

    def draw_calibration(self, rng: np.random.Generator, effort: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The bag's ``(gain, offset)`` per joint: ``1 + g`` and ``o * effort``."""
        effort = np.asarray(effort, dtype=float)
        n = len(effort)
        gain = 1.0 + (rng.uniform(-self.gain, self.gain, size=n) if self.gain else np.zeros(n))
        finite = np.where(np.isfinite(effort), effort, 0.0)
        offset = (rng.uniform(-self.offset, self.offset, size=n) * finite) if self.offset else np.zeros(n)
        return gain, offset

    def measure_sample(self, value: np.ndarray, rng: np.random.Generator, sigma: np.ndarray,
                       gain: np.ndarray, offset: np.ndarray) -> np.ndarray:
        """Steps 1-4 on one sample (the delay is the caller's, it spans samples)."""
        value = np.asarray(value, dtype=float)
        if np.any(sigma > 0.0):
            value = value + rng.normal(0.0, 1.0, size=value.shape) * sigma
        return self.quantize(gain * value + offset)

    def describe(self) -> dict[str, Any]:
        return {"p": float(self.p), "sigma": float(self.sigma), "gain": float(self.gain),
                "offset": float(self.offset), "quantization": list(self.quantization),
                "sigma_quanta": float(self.sigma_quanta)}


#: How the recorded motor velocity is produced.  ``sensor``: the drive's own
#: estimate (UR10 RTDE, FMRR ``0x606C``), i.e. the true velocity through the
#: channel's instrument.  ``position_derivative``: the interface has no
#: velocity at all (the iiwa's FRI), so it is the consumer's own
#: Savitzky-Golay derivative of the recorded position.
DQ_SOURCES = ("sensor", "position_derivative")


@dataclass(frozen=True)
class NoiseModel:
    """The ``simulation.noise`` block: every recorded signal's instrument.

    ``enabled: false`` is the ideal instrument *including* the delay: the
    recorder and the loop then see the true state at the sample instants.
    That is the noise ablation (`R6_00` Sec 6), which asks how much of a
    model's error is instrument rather than physics.
    """

    enabled: bool = True
    q: ChannelNoise = None  # type: ignore[assignment]
    dq: ChannelNoise = None  # type: ignore[assignment]
    ft: ChannelNoise = None  # type: ignore[assignment]
    wrench: ChannelNoise = None  # type: ignore[assignment]
    dq_source: str = "sensor"
    delay_samples: int = 1
    tag: str = "modelling_error"
    provenance: str = "E"

    def __post_init__(self) -> None:
        for name in _NOISE_CHANNELS:
            if getattr(self, name) is None:
                object.__setattr__(self, name, ChannelNoise())
        if self.dq_source not in DQ_SOURCES:
            raise ValueError(f"simulation.noise.dq.source must be one of {DQ_SOURCES}")
        if int(self.delay_samples) < 0:
            raise ValueError("simulation.noise.delay_samples must be non-negative")
        object.__setattr__(self, "delay_samples", int(self.delay_samples))
        if self.dq_source == "position_derivative" and not self.dq.is_ideal:
            raise ValueError(
                "simulation.noise.dq: a velocity derived from the recorded position carries that "
                "position's noise and nothing else; give it no p/gain/offset/quantization of its own"
            )

    @property
    def effective_delay(self) -> int:
        return self.delay_samples if self.enabled else 0

    def channel(self, name: str) -> ChannelNoise:
        return getattr(self, name) if self.enabled else ChannelNoise()

    def seed_for(self, seed: int, channel: str) -> tuple[int, int, int]:
        return (int(seed), _NOISE_STREAM, _NOISE_CHANNELS.index(channel))

    def delay(self, values: np.ndarray) -> np.ndarray:
        return MeasurementModel(delay_samples=self.effective_delay).delay(values)

    def measure(self, name: str, clean: np.ndarray, seed: int, *, effort: np.ndarray,
                basis: np.ndarray | None = None) -> "NoisyChannel":
        """A whole recorded channel, after the fact (``ft``, the wrench)."""
        clean = np.atleast_2d(np.asarray(clean, dtype=float))
        channel = self.channel(name)
        n = clean.shape[1]
        rng = np.random.default_rng(self.seed_for(seed, name))
        sigma = channel.sigma_for(clean if basis is None else basis)
        gain, offset = channel.draw_calibration(rng, np.broadcast_to(np.asarray(effort, dtype=float), (n,)))
        if channel.is_ideal:
            values = clean
        else:
            values = channel.measure_sample(clean, rng, sigma[None, :], gain[None, :], offset[None, :])
        return NoisyChannel(values=self.delay(values), sigma=sigma, gain=gain, offset=offset)

    def describe(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "definition": (
                "per bag and joint: x + N(0, sigma^2) with sigma = p * RMS(x - mean(x)) (root-sum-square "
                "with the absolute sigma and with sigma_quanta quantization steps where given; positions "
                "use quantization steps only, R6_02 Sec 3), then gain (1 + g), g ~ U(-gain, gain), then offset "
                "o ~ U(-offset, offset) * effort limit, then quantization, then delay_samples. For the "
                "channels the controller reads in the loop (q, and dq when it is a sensor) x is the "
                "bag's reference trajectory, the only signal known before the bag runs; for ft it is "
                "the clean target itself (R6_00 Sec 6)"
            ),
            "q": self.q.describe(), "dq": {**self.dq.describe(), "source": self.dq_source},
            "ft": self.ft.describe(), "wrench": self.wrench.describe(),
            "delay_samples": int(self.delay_samples),
            "tag": self.tag, "provenance": self.provenance,
        }


@dataclass(frozen=True)
class NoisyChannel:
    values: np.ndarray
    sigma: np.ndarray
    gain: np.ndarray
    offset: np.ndarray


class LoopInstrument:
    """The motor-side encoder and velocity the controller reads (`R6_00` Sec 2.1).

    Called once per sample period with the true motor state, it records the
    measured, delayed channels exactly as they are written to ``q*``/``dq*``,
    and returns what the controller may use at that instant: the latest
    recorded position, and either the latest recorded velocity (a drive's
    estimate) or, where the interface has no velocity, a *causal*
    Savitzky-Golay derivative of the last ``window`` recorded positions.

    ``sigma_q``/``sigma_dq`` are fixed for the bag by the caller from the
    reference trajectory (:meth:`ChannelNoise.sigma_for`), since the clean
    signal they would otherwise be scaled by does not exist yet.
    """

    def __init__(
        self, noise: NoiseModel, *, n_dof: int, seed: int, sigma_q: np.ndarray, sigma_dq: np.ndarray,
        sample_time: float, sg_window: int = 5, sg_poly: int = 3,
    ) -> None:
        self.noise = noise
        self.n_dof = int(n_dof)
        self.delay = noise.effective_delay
        self._q_channel, self._dq_channel = noise.channel("q"), noise.channel("dq")
        self._rng_q = np.random.default_rng(noise.seed_for(seed, "q"))
        self._rng_dq = np.random.default_rng(noise.seed_for(seed, "dq"))
        self.sigma_q = np.broadcast_to(np.asarray(sigma_q, dtype=float), (self.n_dof,)).copy()
        self.sigma_dq = np.broadcast_to(np.asarray(sigma_dq, dtype=float), (self.n_dof,)).copy()
        if not noise.enabled:
            self.sigma_q[:] = 0.0
            self.sigma_dq[:] = 0.0
        ones, zeros = np.ones(self.n_dof), np.zeros(self.n_dof)
        self._unit = (ones, zeros)
        self.derivative = noise.dq_source == "position_derivative"
        self.sample_time = float(sample_time)
        self.sg_window, self.sg_poly = int(sg_window), int(sg_poly)
        if self.derivative:
            from scipy.signal import savgol_coeffs

            if self.sg_window <= self.sg_poly or self.sg_window % 2 == 0:
                raise ValueError("the causal velocity needs an odd sg_window > sg_poly")
            self._causal = savgol_coeffs(self.sg_window, self.sg_poly, deriv=1, delta=self.sample_time,
                                         pos=self.sg_window - 1, use="dot")
        self._q_raw: list[np.ndarray] = []
        self._dq_raw: list[np.ndarray] = []
        self.q_recorded: list[np.ndarray] = []
        self.dq_recorded: list[np.ndarray] = []
        self.dq_loop: list[np.ndarray] = []

    def read_q(self, q: np.ndarray) -> np.ndarray:
        """One undelayed encoder reading."""
        gain, offset = self._unit
        return self._q_channel.measure_sample(q, self._rng_q, self.sigma_q, gain, offset)

    def measure_dq(self, dq: np.ndarray) -> np.ndarray:
        """A velocity through the recorded ``dq`` channel's instrument."""
        gain, offset = self._unit
        return self._dq_channel.measure_sample(dq, self._rng_dq, self.sigma_dq, gain, offset)

    def read(self, q: np.ndarray, dq: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """One undelayed reading of the encoder (and the velocity sensor, when there is one)."""
        q_now = self.read_q(q)
        if self.derivative:
            return q_now, None
        return q_now, self.measure_dq(dq)

    def sample(self, q: np.ndarray, dq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.record(*self.read(q, dq))

    def record(self, q_read: np.ndarray, dq_read: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        """Record one reading on the bus, delayed; return what a bus-side controller would use."""
        self._q_raw.append(np.asarray(q_read, dtype=float))
        index = len(self._q_raw) - 1
        source = max(index - self.delay, 0)
        q_now = self._q_raw[source]
        self.q_recorded.append(q_now)
        if self.derivative:
            window = [self.q_recorded[max(index - offset_, 0)] for offset_ in range(self.sg_window - 1, -1, -1)]
            dq_now = self._causal @ np.asarray(window)
        else:
            self._dq_raw.append(np.asarray(dq_read, dtype=float))
            dq_now = self._dq_raw[source]
            self.dq_recorded.append(dq_now)
        self.dq_loop.append(dq_now)
        return q_now, dq_now

    def history(self) -> dict[str, np.ndarray]:
        record = {
            "q_motor_measured": np.asarray(self.q_recorded, dtype=float),
            "dq_motor_loop": np.asarray(self.dq_loop, dtype=float),
            "noise_sigma_q": self.sigma_q, "noise_sigma_dq": self.sigma_dq,
        }
        if not self.derivative:
            record["dq_motor_measured"] = np.asarray(self.dq_recorded, dtype=float)
        return record


class DriveInstrument:
    """A PD that lives inside the drive (`R6_02` Sec 7 amended, `R6_04` A-4).

    Every drive tick the drive reads its own encoder -- the joint encoder of
    Sec 6: quantization plus ``sigma_quanta`` counts, no bus delay -- and
    computes its velocity the way a real drive does: the difference of two
    consecutive readings over the drive period, through a first-order
    low-pass at ``cutoff_hz`` (``drive_rate / 10``).  Those two signals close
    the loop.  The Sec 6 percentage noise is a *recorded-channel* modelling
    error, so it is applied only to the copy the bus records, never fed back:
    every ``ticks_per_sample``-th tick the recorder (a :class:`LoopInstrument`
    at the sample rate, which applies the recording delay) stores the
    drive's encoder reading and its velocity estimate through the ``dq``
    channel (or, on an interface with no velocity, the recorder's own
    Savitzky-Golay derivative of the recorded position).
    """

    def __init__(self, recorder: LoopInstrument, ticks_per_sample: int, *, drive_period: float,
                 cutoff_hz: float) -> None:
        if int(ticks_per_sample) < 1:
            raise ValueError("ticks_per_sample must be >= 1")
        if drive_period <= 0.0 or cutoff_hz <= 0.0:
            raise ValueError("drive_period and cutoff_hz must be positive")
        self.recorder = recorder
        self.ticks_per_sample = int(ticks_per_sample)
        self.drive_period = float(drive_period)
        self.cutoff_hz = float(cutoff_hz)
        self.alpha = 1.0 - float(np.exp(-2.0 * np.pi * self.cutoff_hz * self.drive_period))
        self._tick = 0
        self._q_last: np.ndarray | None = None
        self._velocity: np.ndarray | None = None

    def sample(self, q: np.ndarray, dq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q_read = self.recorder.read_q(q)
        if self._q_last is None:
            # A drive enabled on a moving axis starts from its true speed.
            self._velocity = np.asarray(dq, dtype=float).copy()
        else:
            raw = (q_read - self._q_last) / self.drive_period
            self._velocity = self._velocity + self.alpha * (raw - self._velocity)
        self._q_last = q_read
        if self._tick % self.ticks_per_sample == 0:
            dq_record = None if self.recorder.derivative else self.recorder.measure_dq(self._velocity)
            self.recorder.record(q_read, dq_record)
        self._tick += 1
        return q_read, self._velocity.copy()

    def history(self) -> dict[str, np.ndarray]:
        return self.recorder.history()
