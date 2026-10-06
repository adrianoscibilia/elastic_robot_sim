"""Link-side transmission modes and elastic observability (`R6_02` Sec 2).

Seen from motor-side inputs, the elastic part of the transmission torque
scales like ``(f_exc / f_link)^2`` below the link-side mode

    f_link = sqrt(K / M_link) / (2 pi)

(the motor held by the PD; [S-14]).  Excitation far below every ``f_link``
leaves the elasticity a rounding error of the target, which is what the
round-6 pass-1 budget measured (0.1 % on both arms).  This module tabulates
each joint's ``f_link`` band over the prior and says which joints the data
can resolve at all:

* ``K`` spans the stiffness prior (``stiffness_nominal x stiffness_factor``,
  or the explicit ``stiffness`` intervals);
* ``M_link`` is the joint's link-side inertia ``M_jj(q)`` (outboard joints
  rigid), from the bare arm's smallest over the excitation window to the
  largest with the prior's heaviest, farthest payload;
* the resolvable band is the consumer's Savitzky-Golay passband,
  ``0.45 rate / 5 = 0.09 x rate`` at the filter's five-sample floor.

A joint whose band intersects the resolvable band gets probe lines on it
(:func:`link_probe_harmonics`); a joint whose band lies wholly inside it is
*elastically observable* and enters the budget gate (`R6_02` P2-2).  The
others (iiwa A5-A7, UR10 wrists) are elastically unobservable at the rate by
design: they stay in the data and out of the gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .dataset_config import DatasetConfig

#: The consumer's order-3 Savitzky-Golay derivative passes ``0.45 rate / w``;
#: its floor is ``w = 5`` samples (R5_09 D-2), so nothing above 0.09 x rate
#: survives differentiation at any window.
RESOLVABLE_FRACTION = 0.09

#: The share of the prior a joint needs in the resolvable band to count as
#: ``mostly_observable`` (a reporting aid; the gate uses the strict band).
MOSTLY_OBSERVABLE_SHARE = 0.95


@dataclass(frozen=True)
class LinkMode:
    joint: str
    k_min: float
    k_max: float
    m_link_min: float
    m_link_max: float
    resolvable_hz: float
    #: Share of the prior (log-uniform ``K`` x postures, bare and heaviest
    #: payload) whose ``f_link`` is resolvable; the strict band rule ignores
    #: how rare its corners are.
    resolvable_share: float = float("nan")

    @property
    def f_low(self) -> float:
        return float(np.sqrt(self.k_min / self.m_link_max) / (2.0 * np.pi))

    @property
    def f_high(self) -> float:
        return float(np.sqrt(self.k_max / self.m_link_min) / (2.0 * np.pi))

    @property
    def probed(self) -> bool:
        """The band reaches into the resolvable band: the comb covers it."""
        return self.f_low <= self.resolvable_hz

    @property
    def observable(self) -> bool:
        """The whole band is resolvable: the joint enters the elastic budget gate."""
        return self.f_high <= self.resolvable_hz

    @property
    def mostly_observable(self) -> bool:
        """At least 95 % of the prior is resolvable (reported beside the strict rule)."""
        return self.resolvable_share >= MOSTLY_OBSERVABLE_SHARE

    def as_dict(self) -> dict[str, Any]:
        return {"joint": self.joint, "k_min": self.k_min, "k_max": self.k_max,
                "m_link_min": self.m_link_min, "m_link_max": self.m_link_max,
                "f_link_low_hz": self.f_low, "f_link_high_hz": self.f_high,
                "resolvable_hz": self.resolvable_hz, "probed": self.probed, "observable": self.observable,
                "resolvable_share": self.resolvable_share, "mostly_observable": self.mostly_observable}


def _stiffness_range(config: DatasetConfig, n: int) -> tuple[np.ndarray, np.ndarray]:
    sampling = config.transmission
    if sampling.stiffness_nominal:
        nominal = np.asarray(sampling.stiffness_nominal, dtype=float)
        return nominal * sampling.stiffness_factor[0], nominal * sampling.stiffness_factor[1]
    intervals = np.asarray(sampling.stiffness, dtype=float).reshape(-1, 2)
    if len(intervals) != n:
        raise ValueError(f"transmission.stiffness has {len(intervals)} intervals, expected {n}")
    return intervals[:, 0], intervals[:, 1]


def _heaviest_payload(config: DatasetConfig):
    from .payload import Payload

    sampling = config.payload
    if not sampling.enabled or sampling.mass[1] <= 0.0:
        return None

    def farthest(bounds: tuple[float, float]) -> float:
        return bounds[0] if abs(bounds[0]) > abs(bounds[1]) else bounds[1]

    z = farthest(sampling.offset_z)
    if sampling.offset_z_reference == "near_face":
        z += 0.5 * float(sampling.size[1])
    return Payload(mass=float(sampling.mass[1]),
                   offset=(farthest(sampling.offset_x), farthest(sampling.offset_y), z),
                   size=float(sampling.size[1]))


def link_mode_table(config: DatasetConfig, asset: Any, *, n_samples: int | None = None,
                    seed: int = 0) -> list[LinkMode]:
    """One :class:`LinkMode` per joint over the config's prior."""
    from .dataset_bag import _inertia_envelope_bounds
    from .payload import payload_asset
    from .torque_runners import _sample_inertia_diagonals

    names = tuple(asset.joint_names)
    n = len(names)
    samples = int(n_samples or config.transmission.inertia_samples)
    bounds = _inertia_envelope_bounds(asset, config)
    bare = _sample_inertia_diagonals(asset, samples, seed, bounds)
    with payload_asset(asset, _heaviest_payload(config)) as loaded:
        heavy = _sample_inertia_diagonals(loaded, samples, seed, bounds)
    m_min = np.min(bare, axis=0)
    m_max = np.maximum(np.max(bare, axis=0), np.max(heavy, axis=0))
    k_min, k_max = _stiffness_range(config, n)
    resolvable = RESOLVABLE_FRACTION / float(config.sample_time_step)
    postures = np.vstack([bare, heavy])
    rng = np.random.default_rng(seed)
    stiffness = np.exp(rng.uniform(np.log(k_min), np.log(k_max), size=postures.shape))
    share = np.mean(np.sqrt(stiffness / postures) / (2.0 * np.pi) <= resolvable, axis=0)
    return [LinkMode(names[j], float(k_min[j]), float(k_max[j]), float(m_min[j]), float(m_max[j]), resolvable,
                     float(share[j])) for j in range(n)]


def link_probe_harmonics(modes: list[LinkMode], base_frequency: float, *, lowest_hz: float,
                         lines: int, span: tuple[float, float] = (0.5, 1.5),
                         first_harmonic: int = 1) -> tuple[int, ...]:
    """Log-spaced probe harmonics over ``[lowest_hz, top]`` that cover every probed joint's modes.

    ``top`` is the largest ``span[1] x f_link_high`` among probed joints,
    capped at the resolvable band, so the comb runs continuously from the
    sub-resonant floor through every intersecting ``[0.5, 1.5] x f_link``
    band (`R6_02` P2-1).  The lines are integer harmonics of the base
    frequency, deduplicated, from ``first_harmonic`` (above the main
    trajectory's harmonics) up to the last one inside the resolvable band.
    """
    from .excitation import log_spaced_probe_harmonics

    probed = [mode for mode in modes if mode.probed]
    if not probed:
        return ()
    resolvable = min(mode.resolvable_hz for mode in modes)
    top = min(max(span[1] * mode.f_high for mode in probed), resolvable)
    bottom = min(float(lowest_hz), min(span[0] * mode.f_low for mode in probed))
    # Round the top line up to a harmonic (so it covers the band), never past the resolvable one.
    last = min(int(np.ceil(top / base_frequency - 1e-9)), int(np.floor(resolvable / base_frequency + 1e-9)))
    top = last * base_frequency
    bottom = max(bottom, first_harmonic * base_frequency)
    return tuple(sorted({min(max(int(h), first_harmonic), last)
                         for h in log_spaced_probe_harmonics(bottom, top, lines, base_frequency)}))


def resolve_probe_design(config: DatasetConfig, *, asset: Any = None) -> DatasetConfig:
    """Replace ``excitation.probe_design`` by the comb it describes."""
    from dataclasses import replace

    design = config.probe_design
    if design is None:
        return config
    if asset is None:
        from pathlib import Path

        from .assets import AssetRegistry

        asset = AssetRegistry.for_repository(Path(__file__).resolve().parents[2]).load(config.asset)
    modes = link_mode_table(config, asset)
    excitation = config.excitation
    harmonics = link_probe_harmonics(modes, excitation.base_frequency, lowest_hz=design.lowest_hz,
                                     lines=design.lines, span=design.span,
                                     first_harmonic=excitation.n_harmonics + 1)
    return replace(config, excitation=replace(excitation, probe_harmonics=harmonics))


def format_modes(modes: list[LinkMode]) -> str:
    lines = [f"{'joint':22s} {'K [min,max]':>22s} {'M_link [min,max]':>24s} {'f_link [Hz]':>18s} "
             f"{'resolvable':>10s}  probed observable share"]
    for m in modes:
        lines.append(f"{m.joint:22s} {m.k_min:10.4g} {m.k_max:10.4g}  {m.m_link_min:11.4g} {m.m_link_max:11.4g} "
                     f"{m.f_low:8.3g} {m.f_high:8.3g} {m.resolvable_hz:10.4g}  {str(m.probed):6s} {str(m.observable):10s} "
                     f"{m.resolvable_share:.1%}")
    return "\n".join(lines)
