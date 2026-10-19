"""Sim-to-real calibration on the real recordings (S2R_01).

Replays the real erd runs in the live simulation core, compares simulation
and robot on the observed channels, and calibrates the plant parameters.
Heavy dependencies (Pinocchio, MuJoCo, the optimizer backends) are imported
lazily by the modules that need them.
"""

from .config import SCHEMA, ConfigError, Sim2RealConfig, load_config

__all__ = ["SCHEMA", "ConfigError", "Sim2RealConfig", "load_config"]
