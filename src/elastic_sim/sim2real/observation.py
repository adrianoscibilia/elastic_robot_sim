"""Per-platform observation models: what the robot's torque channel reports
for a simulated transmission torque (S2R_01 T-2.4, P-8).

* **iiwa** (:class:`IiwaJointTorqueSensor`): the joint torque sensor sits
  after the gear, so ``ft = gain * tau_s + offset[run]`` per joint, with the
  run-scoped offsets of S2R_01 §5 (A and B were recorded 1.5 h apart).
* **UR10** (later, Phase 7): the current proxy of
  ``erd_recording.convert.ur10_dataset_frame``; **FMRR**: its EtherCAT
  signals. Both are declared here and refuse until implemented.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .plant import SensorCalibration


class ObservationModel(Protocol):
    """Maps the simulated transmission torque to the recorded channel and back."""

    channel: str

    def observe(self, tau_s: np.ndarray, run_id: str) -> np.ndarray: ...

    def invert(self, measured: np.ndarray, run_id: str) -> np.ndarray: ...


class IiwaJointTorqueSensor:
    """KUKA LBR joint torque sensor: ``ft = gain * tau_s + offset[run]``."""

    channel = "ft"

    def __init__(self, sensor: SensorCalibration):
        self.sensor = sensor

    def observe(self, tau_s: np.ndarray, run_id: str) -> np.ndarray:
        """``(N, n)`` (or ``(n,)``) spring torque -> what the sensor reports."""
        return np.asarray(tau_s, dtype=float) * self.sensor.gain + self.sensor.offset(run_id)

    def invert(self, measured: np.ndarray, run_id: str) -> np.ndarray:
        """The spring torque a reading implies: ``(ft - offset) / gain``."""
        return (np.asarray(measured, dtype=float) - self.sensor.offset(run_id)) / self.sensor.gain

    def residual(self, tau_s: np.ndarray, measured: np.ndarray, run_id: str) -> np.ndarray:
        """sim - real on the recorded channel (S2R_01 §6)."""
        return self.observe(tau_s, run_id) - np.asarray(measured, dtype=float)


class _NotYetPlatform:
    channel = ""
    platform = ""
    reference = ""

    def __init__(self, *_args, **_kwargs):
        raise NotImplementedError(f"{self.platform} observation model: S2R_01 Phase 7 ({self.reference})")


class Ur10CurrentProxy(_NotYetPlatform):
    channel, platform, reference = "ft", "UR10", "erd_recording.convert.ur10_dataset_frame"


class FmrrEthercat(_NotYetPlatform):
    channel, platform, reference = "ft", "FMRR", "legacy FMRR calibration [I-5]"


def observation_model(hardware: str, sensor: SensorCalibration) -> ObservationModel:
    """The observation model of a platform (``config.yaml`` ``robot`` family)."""
    if hardware.startswith("kuka_lbr_iiwa"):
        return IiwaJointTorqueSensor(sensor)
    if hardware.startswith("ur10"):
        return Ur10CurrentProxy(sensor)
    if hardware.startswith("fmrr"):
        return FmrrEthercat(sensor)
    raise ValueError(f"no observation model for {hardware!r}")


__all__ = ["FmrrEthercat", "IiwaJointTorqueSensor", "ObservationModel", "Ur10CurrentProxy", "observation_model"]
