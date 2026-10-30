"""Round 8 (identified filter, `identified_filter/01` Sec 1): what an r8 build adds to round 7.

* **Modal band** (Sec 1.1.1): per joint, the filter-terms mode
  ``f_n = sqrt(K_j / M_jj(q)) / 2 pi`` over the widened stiffness prior and
  ``M_jj`` sampled over the excitation window on the bare URDF; the band is
  ``[0.5 min f_n, min(1.5 max f_n, f_SG)]``, ``f_SG = 0.09 x rate``.  A joint
  whose whole ``f_n`` range lies above ``f_SG`` is not resolvable.  The band is
  a property of the *platform's prior*, never of a sampled robot.
* **Probe regimes** (Sec 1.1.2): every bag is ``full`` (32-48 log lines over
  the modal band, acceleration- or position-weighted, additive budget),
  ``real_like`` (the real P1 comb on the B base, split budget) or ``off``.
  Regime, comb, weighting and budget are drawn from the bag's trajectory seed
  only, so they are statistically independent of the robot's K and D.
* **Bags per split**: ``trajectories_by_split`` (train/val 2, test 3); bag 0
  is ``full``, a train/val bag 1 is ``real_like`` or ``off`` at 50/50, a test
  robot runs one of each.
* **Out-of-prior tier** (``*_oop``): robots whose K lies outside the widened
  range, ``x [0.5, 0.67] U [1.5, 2]`` of its bounds.
* **Modal file** (``*_modal``): single-joint log chirps at fixed poses with the
  law of the draft lab configs (``erd_recording/config/lab/*_probe.yaml``,
  ``modal:``).
* **Per-bag verification** (Sec 1.1.3): an oracle least-squares fit of
  ``(K, D)`` per joint with the true ``M(theta)`` and ``tau_r`` = URDF RNEA at
  the recorded motor state plus the bag's true link friction, and the elastic
  in-band SNR.  Privileged: written to the bag's metadata, never an input.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .materialized import MaterializedTrajectory

#: ``f_SG = 0.09 x rate``: the order-3 SG derivative's passband at its five-sample floor (link_modes).
SG_BAND_FRACTION = 0.09
REGIMES = ("full", "real_like", "off")
#: Stream index of the regime draw, ``(seed, 21, trajectory_seed)``: new, so no existing draw moves.
REGIME_STREAM = 21
#: Stream index of the out-of-prior robots' draw.
OOP_STREAM = 22
#: The oracle fit's band per joint, ``x f_n`` of that robot (capped at 0.95 f_SG).
FIT_BAND = (0.3, 3.0)
#: Trajectory-seed offset per regeneration attempt (Sec 1.1.3: "the next seed").
RESEED_STRIDE = 1_000_000_007


@dataclass(frozen=True)
class FullRegime:
    """``full``: every resolvable mode excited (Sec 1.1.2)."""

    lines: tuple[int, int] = (32, 48)
    acceleration_share: float = 0.7
    fraction: tuple[float, float] = (0.5, 1.0)


@dataclass(frozen=True)
class RealLikeRegime:
    """``real_like``: the real P1 comb on the B base (RR_20 Sec 3.3)."""

    base_frequency: float = 0.2
    n_periods: int = 2
    n_harmonics: int = 5
    harmonics: tuple[int, ...] = ()
    fraction: float = 0.25
    weighting: str = "acceleration"


@dataclass(frozen=True)
class ModalSpec:
    """Single-joint chirps (the lab ``modal:`` block's law)."""

    joints: tuple[tuple[str, float, float, float], ...] = ()   # (joint, f0, f1, T)
    poses: tuple[tuple[str, tuple[float, ...]], ...] = ()      # (name, q)
    a_max: float = 2.0
    position_amplitude_max: float = 0.02
    velocity_max: float = 1.0
    jerk_max: float = 100.0
    repeats: int = 3
    hold_s: float = 1.0
    fade_s: float = 1.0
    robots: int = 10

    @property
    def segments(self) -> list[tuple[str, int, int]]:
        """``(joint, pose index, repeat)`` per trajectory index, joint slowest."""
        return [(joint[0], pose, repeat) for joint in self.joints for pose in range(len(self.poses))
                for repeat in range(self.repeats)]


@dataclass(frozen=True)
class OopSpec:
    robots: int = 10
    bands: tuple[tuple[float, float], ...] = ((0.5, 0.67), (1.5, 2.0))
    trajectories: int = 2


@dataclass(frozen=True)
class VerificationSpec:
    k_relerr_max: float = 0.10
    snr_min: float = 3.0
    main_joints: tuple[str, ...] = ()
    retries: int = 3
    gate_fraction: float = 0.95
    raise_fraction: float = 0.80


@dataclass(frozen=True)
class Round8Spec:
    full: FullRegime = field(default_factory=FullRegime)
    real_like: RealLikeRegime = field(default_factory=RealLikeRegime)
    second_bag_real_like: float = 0.5
    trajectories_by_split: tuple[tuple[str, int], ...] = (("train", 2), ("val", 2), ("test", 3))
    oop: OopSpec = field(default_factory=OopSpec)
    modal: ModalSpec = field(default_factory=ModalSpec)
    verification: VerificationSpec = field(default_factory=VerificationSpec)
    #: Resolved at load (:func:`resolve_round8`): the per-joint band table and the comb's span.
    band: tuple[Mapping[str, Any], ...] = ()
    comb_hz: tuple[float, float] = (0.0, 0.0)
    #: ``--coverage-escalation`` (Sec 1.1.3): 0 as written; 1 all-acceleration weighting and
    #: fraction [0.8, 1]; 2 also the line count at its low end (more amplitude per line).
    escalation: int = 0

    def trajectories_for(self, split: str) -> int:
        return dict(self.trajectories_by_split).get(split, max(n for _, n in self.trajectories_by_split))

    def as_dict(self) -> dict[str, Any]:
        return {
            "full": {"lines": list(self.full.lines), "acceleration_share": self.full.acceleration_share,
                     "fraction": list(self.full.fraction), "budget": "additive"},
            "real_like": {"base_frequency": self.real_like.base_frequency, "n_periods": self.real_like.n_periods,
                          "n_harmonics": self.real_like.n_harmonics, "harmonics": list(self.real_like.harmonics),
                          "fraction": self.real_like.fraction, "weighting": self.real_like.weighting,
                          "budget": "split"},
            "second_bag_real_like": self.second_bag_real_like,
            "trajectories_by_split": dict(self.trajectories_by_split),
            "escalation": self.escalation,
        }


_ROUND8_KEYS = {"probe_regimes", "trajectories_by_split", "oop", "modal", "verification"}


def parse_round8(raw: Mapping[str, Any] | None, source: Any) -> Round8Spec | None:
    """The config's ``round8:`` block, or ``None``."""
    if raw is None:
        return None
    unknown = sorted(set(raw) - _ROUND8_KEYS)
    if unknown:
        raise ValueError(f"{source}: unknown round8 keys {unknown}")
    regimes = raw.get("probe_regimes") or {}
    full_raw, real_raw = regimes.get("full") or {}, regimes.get("real_like") or {}
    full = FullRegime(lines=tuple(int(v) for v in full_raw.get("lines", (32, 48))),
                      acceleration_share=float(full_raw.get("acceleration_share", 0.7)),
                      fraction=tuple(float(v) for v in full_raw.get("fraction", (0.5, 1.0))))
    real = RealLikeRegime(base_frequency=float(real_raw.get("base_frequency", 0.2)),
                          n_periods=int(real_raw.get("n_periods", 2)),
                          n_harmonics=int(real_raw.get("n_harmonics", 5)),
                          harmonics=tuple(int(h) for h in real_raw.get("harmonics", ())),
                          fraction=float(real_raw.get("fraction", 0.25)),
                          weighting=str(real_raw.get("weighting", "acceleration")))
    if not real.harmonics:
        raise ValueError(f"{source}: round8.probe_regimes.real_like.harmonics is required (the real P1 comb)")
    if not (0.0 < full.fraction[0] <= full.fraction[1] <= 1.0):
        raise ValueError(f"{source}: round8 full fraction must lie in (0, 1] (additive budget)")
    oop_raw = raw.get("oop") or {}
    oop = OopSpec(robots=int(oop_raw.get("robots", 10)),
                  bands=tuple(tuple(float(v) for v in b) for b in oop_raw.get("bands", ((0.5, 0.67), (1.5, 2.0)))),
                  trajectories=int(oop_raw.get("trajectories", 2)))
    modal_raw = raw.get("modal") or {}
    modal = ModalSpec(
        joints=tuple((str(name), float(v["f0"]), float(v["f1"]), float(v["T"]))
                     for name, v in (modal_raw.get("joints") or {}).items()),
        poses=tuple((str(p["name"]), tuple(float(x) for x in p["q"])) for p in modal_raw.get("poses") or ()),
        a_max=float(modal_raw.get("a_max", 2.0)),
        position_amplitude_max=float(modal_raw.get("position_amplitude_max", 0.02)),
        velocity_max=float(modal_raw.get("velocity_max", 1.0)),
        jerk_max=float(modal_raw.get("jerk_max", 100.0)),
        repeats=int(modal_raw.get("repeats", 3)), hold_s=float(modal_raw.get("hold_s", 1.0)),
        fade_s=float(modal_raw.get("fade_s", 1.0)), robots=int(modal_raw.get("robots", 10)),
    )
    ver_raw = raw.get("verification") or {}
    verification = VerificationSpec(
        k_relerr_max=float(ver_raw.get("k_relerr_max", 0.10)), snr_min=float(ver_raw.get("snr_min", 3.0)),
        main_joints=tuple(str(j) for j in ver_raw.get("main_joints", ())),
        retries=int(ver_raw.get("retries", 3)), gate_fraction=float(ver_raw.get("gate_fraction", 0.95)),
        raise_fraction=float(ver_raw.get("raise_fraction", 0.80)),
    )
    by_split = raw.get("trajectories_by_split") or {"train": 2, "val": 2, "test": 3}
    second = float(regimes.get("second_bag_real_like", 0.5))
    return Round8Spec(full=full, real_like=real, second_bag_real_like=second,
                      trajectories_by_split=tuple((str(k), int(v)) for k, v in by_split.items()),
                      oop=oop, modal=modal, verification=verification)


# ------------------------------------------------------------------ modal band ---

def modal_band(config: Any, asset: Any, *, n_samples: int = 512, seed: int = 0) -> tuple[list[dict], tuple]:
    """The Sec 1.1.1 band table and the comb's ``(f_lo, f_hi)`` over the resolvable joints.

    ``M_jj(q)`` over ``n_samples`` (>= 500) configurations of the bare URDF in
    the excitation window; ``K`` over the config's (widened) stiffness prior.
    """
    from .dataset_bag import _inertia_envelope_bounds
    from .link_modes import _stiffness_range
    from .torque_runners import _sample_inertia_diagonals

    names = tuple(asset.joint_names)
    inertia = _sample_inertia_diagonals(asset, max(500, int(n_samples)), seed, _inertia_envelope_bounds(asset, config))
    k_min, k_max = _stiffness_range(config, len(names))
    f_sg = SG_BAND_FRACTION / float(config.sample_time_step)
    table = []
    for j, name in enumerate(names):
        f_min = float(np.sqrt(k_min[j] / inertia[:, j].max()) / (2.0 * math.pi))
        f_max = float(np.sqrt(k_max[j] / inertia[:, j].min()) / (2.0 * math.pi))
        resolvable = f_min <= f_sg
        table.append({"joint": name, "f_n_min_hz": f_min, "f_n_max_hz": f_max,
                      "f_lo_hz": 0.5 * f_min, "f_hi_hz": min(1.5 * f_max, f_sg), "f_sg_hz": f_sg,
                      "m_jj_min": float(inertia[:, j].min()), "m_jj_max": float(inertia[:, j].max()),
                      "k_min": float(k_min[j]), "k_max": float(k_max[j]), "resolvable": bool(resolvable)})
    resolvable = [row for row in table if row["resolvable"]]
    if not resolvable:
        raise ValueError("no joint's mode is resolvable at this rate: the round-8 full regime has no band")
    return table, (min(r["f_lo_hz"] for r in resolvable), max(r["f_hi_hz"] for r in resolvable))


def resolve_round8(config: Any, asset: Any = None) -> Any:
    """Fill ``config.round8``'s band table (called by ``load_config``)."""
    spec = config.round8
    if spec is None or spec.band:
        return config
    if asset is None:
        from pathlib import Path

        from .assets import AssetRegistry

        asset = AssetRegistry.for_repository(Path(__file__).resolve().parents[2]).load(config.asset)
    table, comb = modal_band(config, asset, n_samples=config.transmission.inertia_samples)
    return replace(config, round8=replace(spec, band=tuple(table), comb_hz=comb))


# ------------------------------------------------------------------- regimes -----

def regime_for(spec: Round8Spec, split: str, index: int, rng: np.random.Generator) -> str:
    """Bag 0 ``full``; a test robot's bags 1, 2 ``real_like``, ``off``; else a 50/50 draw (Sec 1 table)."""
    if index == 0:
        return "full"
    if split == "test":
        return REGIMES[min(index, 2)]
    return "real_like" if rng.uniform() < spec.second_bag_real_like else "off"


def regime_draw(spec: Round8Spec, base: Any, split: str, index: int, dataset_seed: int,
                trajectory_seed_value: int) -> tuple[str, Any]:
    """``(regime, excitation)`` of one bag; drawn from ``(seed, 21, trajectory_seed)`` only, never from K or D."""
    from .excitation import log_spaced_probe_harmonics

    rng = np.random.default_rng((int(dataset_seed), REGIME_STREAM, int(trajectory_seed_value)))
    regime = regime_for(spec, split, index, rng)
    # Every draw is made in a fixed order whatever the regime, so a regime's numbers never shift another's.
    lines = int(rng.integers(spec.full.lines[0], spec.full.lines[1] + 1))
    accel = rng.uniform() < spec.full.acceleration_share
    fraction = float(rng.uniform(*spec.full.fraction))
    if regime == "off":
        return regime, replace(base, probe_harmonics=())
    if regime == "real_like":
        rl = spec.real_like
        return regime, replace(base, base_frequency=rl.base_frequency, n_periods=rl.n_periods,
                               n_harmonics=rl.n_harmonics, probe_harmonics=tuple(rl.harmonics),
                               probe_budget="split", probe_acceleration_fraction=rl.fraction,
                               probe_weighting=rl.weighting)
    if spec.escalation >= 1:
        accel, fraction = True, 0.8 + 0.2 * (fraction - spec.full.fraction[0]) / max(
            1e-12, spec.full.fraction[1] - spec.full.fraction[0])
    if spec.escalation >= 2:
        lines = spec.full.lines[0]
    f_lo, f_hi = spec.comb_hz
    if f_hi <= 0.0:
        raise ValueError("round8 band not resolved: load the config with load_config (resolve_round8)")
    bottom = max(f_lo, (base.n_harmonics + 1) * base.base_frequency)
    harmonics = tuple(sorted({h for h in log_spaced_probe_harmonics(bottom, f_hi, lines, base.base_frequency)
                              if h > base.n_harmonics and h * base.base_frequency <= f_hi + 1e-9}))
    return regime, replace(base, probe_harmonics=harmonics, probe_budget="additive",
                           probe_acceleration_fraction=min(1.0, fraction),
                           probe_weighting="acceleration" if accel else "position")


# ------------------------------------------------------------- out-of-prior ------

def sample_oop_robots(sampling: Any, seed: int, spec: OopSpec) -> tuple:
    """``spec.robots`` robots ``o00..`` with every K outside the widened prior (``x bands`` of its bounds).

    Per joint the factor is log-uniform over the union of the two bands
    (below the low bound, above the high bound); damping ratio and rotor
    inertia are drawn as the in-prior robots'.
    """
    from .dataset_config import Tier

    rng = np.random.default_rng((int(seed), OOP_STREAM))
    nominal = np.asarray(sampling.stiffness_nominal, dtype=float)
    low, high = nominal * sampling.stiffness_factor[0], nominal * sampling.stiffness_factor[1]
    (a0, a1), (b0, b1) = spec.bands
    width_low, width_high = math.log(a1 / a0), math.log(b1 / b0)
    robots = []
    for index in range(spec.robots):
        stiffness = np.empty(len(nominal))
        for j in range(len(nominal)):
            u = rng.uniform(0.0, width_low + width_high)
            stiffness[j] = (low[j] * a0 * math.exp(u) if u < width_low
                            else high[j] * b0 * math.exp(u - width_low))
        zeta = rng.uniform(sampling.damping_ratio[0], sampling.damping_ratio[1], size=len(nominal))
        if sampling.rotor_inertia_nominal:
            lo, hi = np.log(sampling.rotor_inertia_factor)
            rotor = tuple(np.asarray(sampling.rotor_inertia_nominal) * np.exp(rng.uniform(lo, hi, len(nominal))))
        else:
            rotor = sampling.rotor_inertia
        robots.append(Tier(f"o{index:02d}", stiffness=tuple(stiffness), damping_ratio=tuple(zeta),
                           rotor_inertia=rotor))
    return tuple(robots)


# --------------------------------------------------------------------- modal -----

class ChirpEvaluator:
    """``t -> (q, dq, ddq)`` of one single-joint log chirp (module-level, so a pool can pickle it).

    ``theta_j(t) = q_pose + A(f(t)) w(t) sin phi(t)``, ``f`` log from ``f0`` to
    ``f1`` over ``T``, ``A(f) = a_max / (2 pi f)^2`` clipped by the position,
    velocity and jerk caps, ``w`` a cosine taper over ``fade_s`` at both ends,
    ``hold_s`` at rest before and after.
    """

    def __init__(self, pose: Sequence[float], joint: int, f0: float, f1: float, duration: float,
                 spec: ModalSpec) -> None:
        self.pose = np.asarray(pose, dtype=float)
        self.joint, self.f0, self.f1, self.T = int(joint), float(f0), float(f1), float(duration)
        self.spec = spec
        self.rate = math.log(self.f1 / self.f0) / self.T

    def amplitude(self, f: np.ndarray) -> np.ndarray:
        s, w = self.spec, 2.0 * math.pi * np.asarray(f, dtype=float)
        return np.minimum.reduce([s.a_max / w ** 2, np.full_like(w, s.position_amplitude_max),
                                  s.velocity_max / w, s.jerk_max / w ** 3])

    def _chirp(self, tau: np.ndarray) -> np.ndarray:
        s = self.spec
        inside = (tau >= 0.0) & (tau <= self.T)
        tc = np.clip(tau, 0.0, self.T)
        f = self.f0 * np.exp(self.rate * tc)
        phase = 2.0 * math.pi * self.f0 * (np.exp(self.rate * tc) - 1.0) / self.rate
        fade = np.ones_like(tc)
        if s.fade_s > 0.0:
            fade = np.minimum(fade, 0.5 - 0.5 * np.cos(math.pi * np.clip(tc / s.fade_s, 0.0, 1.0)))
            fade = np.minimum(fade, 0.5 - 0.5 * np.cos(math.pi * np.clip((self.T - tc) / s.fade_s, 0.0, 1.0)))
        return np.where(inside, self.amplitude(f) * fade * np.sin(phase), 0.0)

    def offset(self, time: np.ndarray) -> np.ndarray:
        return self._chirp(np.asarray(time, dtype=float) - self.spec.hold_s)

    @property
    def duration(self) -> float:
        return self.T + 2.0 * self.spec.hold_s

    def __call__(self, t: float):
        h = 1e-5
        x = self.offset(np.asarray([t - h, t, t + h]))
        q = self.pose.copy()
        dq, ddq = np.zeros_like(q), np.zeros_like(q)
        q[self.joint] += x[1]
        dq[self.joint] = (x[2] - x[0]) / (2.0 * h)
        ddq[self.joint] = (x[2] - 2.0 * x[1] + x[0]) / (h * h)
        return q, dq, ddq


def modal_trajectory(spec: ModalSpec, asset: Any, index: int, time_step: float) -> MaterializedTrajectory:
    """Trajectory ``index`` of the modal file: ``spec.segments[index]``."""
    joint_name, pose_index, repeat = spec.segments[index]
    names = tuple(asset.joint_names)
    joint = names.index(joint_name)
    _, f0, f1, duration = next(j for j in spec.joints if j[0] == joint_name)
    pose_name, pose = spec.poses[pose_index]
    if len(pose) != len(names):
        raise ValueError(f"modal pose {pose_name!r} has {len(pose)} joints, the asset {len(names)}")
    evaluator = ChirpEvaluator(pose, joint, f0, f1, duration, spec)
    time = np.arange(0.0, evaluator.duration + 0.5 * time_step, time_step)
    x = evaluator.offset(time)
    position = np.tile(np.asarray(pose, dtype=float), (len(time), 1))
    position[:, joint] += x
    velocity = np.zeros_like(position)
    velocity[:, joint] = np.gradient(x, time_step)
    acceleration = np.zeros_like(position)
    acceleration[:, joint] = np.gradient(velocity[:, joint], time_step)
    metadata = {"kind": "modal_chirp", "joint": joint_name, "pose": pose_name, "pose_index": int(pose_index),
                "repeat": int(repeat), "f0": f0, "f1": f1, "T": duration, "a_max": spec.a_max,
                "base_frequency": f0, "max_acceleration": spec.a_max, "velocity_fraction": 0.0,
                "probe_top_hz": f1, "condition_number": 1.0, "collision_rejections": 0}
    return MaterializedTrajectory(time=time, position=position, velocity=velocity, joint_names=names,
                                  acceleration=acceleration, metadata=metadata, analytic=evaluator)


# -------------------------------------------------------------- verification -----

def _bandpass(x: np.ndarray, low: float, high: float, rate: float) -> np.ndarray | None:
    from scipy.signal import butter, sosfiltfilt

    nyquist = 0.5 * rate
    high = min(high, 0.95 * nyquist)
    if low >= high or low <= 0.0:
        return None
    sos = butter(2, [low / nyquist, high / nyquist], btype="band", output="sos")
    return sosfiltfilt(sos, x, axis=0)


def fit_two_mass(tau_r: np.ndarray, inertia: np.ndarray, target: np.ndarray, kept: np.ndarray, step: float,
                 k0: np.ndarray, d0: np.ndarray, *, iterations: int = 25) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares ``(K, D)`` per joint of the diagonal two-mass filter to ``target`` on ``kept`` rows.

    Joints are independent in the diagonal filter, so one Levenberg-Marquardt
    step updates every joint at once: perturbing every joint's ``log K`` (then
    ``log D``) together costs one filter run per Jacobian column pair.
    """
    from .two_mass import filter_sequence

    def run(logk, logd):
        return filter_sequence(tau_r, inertia, np.exp(logk), np.exp(logd), step)["tau_s"][kept]

    y = target[kept]
    logk, logd = np.log(np.asarray(k0, dtype=float)), np.log(np.asarray(d0, dtype=float))
    lam = np.full(len(logk), 1e-2)
    r = run(logk, logd) - y
    cost = (r ** 2).sum(axis=0)
    eps = 1e-4
    for _ in range(iterations):
        jk = (run(logk + eps, logd) - y - r) / eps
        jd = (run(logk, logd + eps) - y - r) / eps
        a11, a12, a22 = (jk * jk).sum(0), (jk * jd).sum(0), (jd * jd).sum(0)
        g1, g2 = (jk * r).sum(0), (jd * r).sum(0)
        improved = np.zeros(len(logk), dtype=bool)
        for _ in range(6):
            m11, m22 = a11 * (1 + lam), a22 * (1 + lam)
            det = np.where(np.abs(m11 * m22 - a12 ** 2) > 1e-300, m11 * m22 - a12 ** 2, 1e-300)
            dk = np.clip(-(m22 * g1 - a12 * g2) / det, -1.0, 1.0)
            dd = np.clip(-(m11 * g2 - a12 * g1) / det, -1.0, 1.0)
            step_k, step_d = np.where(improved, 0.0, dk), np.where(improved, 0.0, dd)
            r_new = run(logk + step_k, logd + step_d) - y
            cost_new = (r_new ** 2).sum(axis=0)
            better = (cost_new < cost) & ~improved
            logk, logd = np.where(better, logk + step_k, logk), np.where(better, logd + step_d, logd)
            cost = np.where(better, cost_new, cost)
            r = np.where(better[None, :], r_new, r)
            lam = np.where(better, lam / 3.0, lam * 4.0)
            improved |= better
            if improved.all():
                break
        if np.all(np.abs(np.where(improved, np.maximum(np.abs(dk), np.abs(dd)), 0.0)) < 1e-6):
            break
    return np.exp(logk), np.exp(logd)


def fit_two_mass_coupled(tau_r: np.ndarray, inertia: np.ndarray, target: np.ndarray, kept: np.ndarray,
                         step: float, k0: np.ndarray, d0: np.ndarray, free: np.ndarray, *,
                         iterations: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares ``(K, D)`` of the full-coupling two-mass filter; only the ``free`` joints move.

    Levenberg-Marquardt in ``log K``, ``log D`` with a forward-difference
    Jacobian, each joint's residual scaled by its target's RMS so every joint
    weighs the same.  The diagonal filter cannot stand in: on the UR10 lift
    and elbow the coupled modes sit far from ``sqrt(K / M_jj)``, and the
    diagonal fit runs off to the identity (``K -> inf``) instead.
    """
    from .two_mass import filter_sequence

    free = np.flatnonzero(np.asarray(free, dtype=bool))
    y = target[kept]
    scale = 1.0 / np.maximum(np.sqrt(np.mean(y ** 2, axis=0)), 1e-12)
    theta = np.concatenate([np.log(k0), np.log(d0)])
    n = len(k0)
    index = np.concatenate([free, free + n])

    def residual(params):
        out = filter_sequence(tau_r, inertia, np.exp(params[:n]), np.exp(params[n:]), step, coupling="full")
        return ((out["tau_s"][kept] - y) * scale).reshape(-1)

    r = residual(theta)
    cost, lam, eps = float(r @ r), 1e-2, 1e-4
    for _ in range(iterations):
        J = np.empty((len(r), len(index)))
        for c, i in enumerate(index):
            probe = theta.copy()
            probe[i] += eps
            J[:, c] = (residual(probe) - r) / eps
        A, g = J.T @ J, J.T @ r
        for _ in range(8):
            delta = np.clip(-np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), g), -1.0, 1.0)
            trial = theta.copy()
            trial[index] += delta
            r_new = residual(trial)
            cost_new = float(r_new @ r_new)
            if cost_new < cost:
                theta, r, cost, lam = trial, r_new, cost_new, lam / 3.0
                break
            lam *= 4.0
        else:
            break
        if np.max(np.abs(delta)) < 1e-5:
            break
    return np.exp(theta[:n]), np.exp(theta[n:])


def fit_two_mass_multistart(tau_r, inertia, target, kept, step, stiffness, damping, free,
                            starts=((1.5, 0.6), (0.67, 1.5))):
    """:func:`fit_two_mass_coupled` from each ``(K, D)`` factor pair off the truth; the lowest-cost fit."""
    from .two_mass import filter_sequence

    y = target[kept]
    scale = 1.0 / np.maximum(np.sqrt(np.mean(y ** 2, axis=0)), 1e-12)
    best = None
    for k_factor, d_factor in starts:
        k, d = fit_two_mass_coupled(tau_r, inertia, target, kept, step, np.where(free, stiffness * k_factor, stiffness),
                                    np.where(free, damping * d_factor, damping), free)
        r = (filter_sequence(tau_r, inertia, k, d, step, coupling="full")["tau_s"][kept] - y) * scale
        cost = float(np.sum(r ** 2))
        if best is None or cost < best[0]:
            best = (cost, k, d)
    return best[1], best[2]


def verify_bags(frame: pd.DataFrame, manifest: Mapping[str, Any], contract: Mapping[str, Any], asset: Any,
                spec: Round8Spec, *, keep: Sequence[str] | None = None, states: str = "clean") -> pd.DataFrame:
    """Per elastic bag and joint: ``ident_K_relerr``, ``ident_D_relerr``, ``elastic_snr``, ``observable``.

    ``tau_r,true`` = URDF RNEA at the recorded motor state plus the bag's
    true link friction law.  ``states="clean"`` (default) reads the recorded
    *clean* motor state (``q_motor_clean``, ``dq_motor_clean``, SG ``ddq``):
    the fit then asks whether the bag's excitation identifies K and D at all,
    and the sensor noise enters through the SNR only.  On the noisy inputs
    (``states="measured"``) the SG-differentiated ``ddq`` noise, amplified
    ``1/(2 zeta)`` at the mode, biases the fit (errors in variables, D-3.4)
    and mixes the two questions; ``M(theta)`` = URDF CRBA at the recorded
    ``q`` (the plant carries no payload in a ``payload.mode: none`` build).
    The SNR is the band-passed ``tau_s - tau_r,true`` in ``[0.5, 2] f_n`` over
    the band-passed sensor error ``ft - ft_clean``; ``f_n`` is the robot's
    ``sqrt(K / M_jj,median) / 2 pi``.  A joint-bag is resolvable when its
    ``f_n`` is at most ``f_SG``.

    The plant's spring is nonlinear (round 6, ``stiffness_nonlinearity``: the
    sampled K is the middle segment's) and carries the transmission error, so
    the linear ``(K, D)`` a bag can identify is its operating point's.  The
    effective pair ``K_eff``, ``D_eff`` is the least-squares fit of the clean
    transmission torque ``tau_link_clean`` on the true deflection and its
    rate (simulator truth, privileged); ``ident_K_relerr`` is against it,
    ``ident_K_relerr_nominal`` against the sampled K.
    """
    from .diagnostics import savitzky_golay_acceleration
    from .round7 import _block, _model_for, _rigid

    n = int(manifest["n_dof"])
    names = list(manifest["joint_names"])
    step = float(manifest["sample_time_step"])
    rate = 1.0 / step
    window = int((contract.get("differentiation") or {}).get("sg_window") or 11)
    poly = int((contract.get("differentiation") or {}).get("sg_poly") or 3)
    priors = contract.get("transmission_priors") or {}
    head = float(priors.get("burn_in_s", 0.0))
    inertia_median = {r["name"]: np.asarray(r.get("plant_link_inertia_median") or r.get("link_inertia_median"), float)
                      for r in manifest.get("robots", []) if r.get("plant_link_inertia_median") or
                      r.get("link_inertia_median")}
    f_sg = SG_BAND_FRACTION * rate
    model = _model_for(asset, None, {})
    rows = []
    for bag, group in frame.groupby("bag", sort=False):
        if keep is not None and bag not in keep:
            continue
        tier = str(group["tier"].iloc[0])
        if tier == "rigid":
            continue
        clean_states = states == "clean" and "q_motor_clean0" in group.columns
        q = _block(group, "q_motor_clean" if clean_states else "q", n)
        dq = _block(group, "dq_motor_clean" if clean_states else "dq", n)
        ddq = savitzky_golay_acceleration(dq, step, window=window, order=poly)
        target = _block(group, "ft", n)
        clean = _block(group, "ft_clean", n) if "ft_clean0" in group.columns else target
        time = group["t"].to_numpy(dtype=float)
        kept = (time - time[0]) >= head - 1e-9
        tau_rnea, inertia = _rigid(*model, q, dq, ddq)
        viscous = np.asarray([float(group[f"link_viscous__{x}"].iloc[0]) for x in names]) \
            if f"link_viscous__{names[0]}" in group.columns else np.zeros(n)
        coulomb = np.asarray([float(group[f"link_coulomb__{x}"].iloc[0]) for x in names]) \
            if f"link_coulomb__{names[0]}" in group.columns else np.zeros(n)
        from .identification import FrictionModel

        tau_r = tau_rnea + FrictionModel(np.nan_to_num(viscous), np.nan_to_num(coulomb)).torque(dq)
        stiffness = np.asarray([float(group[f"stiffness__{x}"].iloc[0]) for x in names])
        damping = np.asarray([float(group[f"damping__{x}"].iloc[0]) for x in names])
        diagonal = np.diagonal(inertia, axis1=-2, axis2=-1)
        m_med = inertia_median.get(tier, np.median(diagonal, axis=0))
        f_n = np.sqrt(stiffness / m_med) / (2.0 * math.pi)
        # Fit on both sides band-passed in one common band (the filter is linear and coupled, so one band
        # for every joint keeps the pair exact): from FIT_BAND[0] x the lowest resolvable mode, which drops
        # the slow main trajectory and its friction error, to 0.95 f_SG.
        free = f_n <= f_sg
        low = FIT_BAND[0] * float(np.min(f_n[free])) if free.any() else FIT_BAND[0] * float(np.min(f_n))
        band = _bandpass(np.column_stack([tau_r, target]), low, 0.95 * f_sg, rate)
        tau_band, target_band = (tau_r, target) if band is None else (band[:, :n], band[:, n:])
        # Two starts off the truth (K x 1.5, D x 0.6 and K x 0.67, D x 1.5; the lower cost wins), so a fit that
        # never moves cannot pass and close coupled modes do not trap it; joints whose mode is above f_SG
        # stay at the truth (unresolvable: reported, not fitted).
        k_hat, d_hat = fit_two_mass_multistart(tau_band, inertia, target_band, kept, step, stiffness, damping, free)
        elastic = target - tau_r
        noise = target - clean
        k_eff, d_eff = effective_transmission(group, n, stiffness, damping)
        row = {"bag": bag, "tier": tier, "split": str(group["split"].iloc[0])}
        for j, name in enumerate(names):
            band = _bandpass(np.column_stack([elastic[:, j], noise[:, j]]), 0.5 * f_n[j], 2.0 * f_n[j], rate)
            if band is None:
                snr = float("nan")
            else:
                signal_rms = float(np.sqrt(np.mean(band[kept, 0] ** 2)))
                noise_rms = float(np.sqrt(np.mean(band[kept, 1] ** 2)))
                snr = signal_rms / noise_rms if noise_rms > 0 else float("inf")
            # A joint whose mode is above f_SG was not fitted (held at the truth): nothing to score.
            k_err = float(abs(k_hat[j] / k_eff[j] - 1.0)) if free[j] else float("nan")
            d_err = float(abs(d_hat[j] / d_eff[j] - 1.0)) if free[j] and d_eff[j] > 0 else float("nan")
            row.update({f"ident_K__{name}": float(k_hat[j]), f"ident_D__{name}": float(d_hat[j]),
                        f"K_eff__{name}": float(k_eff[j]), f"D_eff__{name}": float(d_eff[j]),
                        f"ident_K_relerr__{name}": k_err, f"ident_D_relerr__{name}": d_err,
                        f"ident_K_relerr_nominal__{name}": (float(abs(k_hat[j] / stiffness[j] - 1.0)) if free[j]
                                                            else float("nan")),
                        f"elastic_snr__{name}": snr, f"f_n_filter__{name}": float(f_n[j]),
                        f"link_inertia_median__{name}": float(m_med[j]),
                        f"zeta_filter__{name}": float(damping[j] / (2.0 * math.sqrt(stiffness[j] * m_med[j]))),
                        f"resolvable__{name}": bool(f_n[j] <= f_sg),
                        f"observable__{name}": bool(k_err <= spec.verification.k_relerr_max
                                                    and np.isfinite(snr) and snr >= spec.verification.snr_min)})
        rows.append(row)
    return pd.DataFrame(rows)


def effective_transmission(group: pd.DataFrame, n: int, stiffness: np.ndarray, damping: np.ndarray
                           ) -> tuple[np.ndarray, np.ndarray]:
    """Per joint, the LS ``(K, D)`` of ``tau_link_clean ~ K defl + D d(defl)/dt`` (clean truth); the sampled
    pair where the clean columns are missing or the fit is degenerate."""
    names = ("tau_link_clean", "q_motor_clean", "q_link_clean", "dq_motor_clean", "dq_link_clean")
    if not all(f"{name}0" in group.columns for name in names):
        return stiffness.copy(), damping.copy()
    from .round7 import _block

    tau = _block(group, "tau_link_clean", n)
    delta = _block(group, "q_motor_clean", n) - _block(group, "q_link_clean", n)
    rate = _block(group, "dq_motor_clean", n) - _block(group, "dq_link_clean", n)
    k_eff, d_eff = stiffness.copy(), damping.copy()
    for j in range(n):
        A = np.column_stack([delta[:, j], rate[:, j], np.ones(len(delta))])
        coef, *_ = np.linalg.lstsq(A, tau[:, j], rcond=None)
        if coef[0] > 0.0:
            k_eff[j] = coef[0]
        if coef[1] > 0.0:
            d_eff[j] = coef[1]
    return k_eff, d_eff


def gate_joints(spec: Round8Spec) -> list[str]:
    """The joints the gate counts: the main joints plus every resolvable wrist (Sec 1.1.3)."""
    resolvable = [row["joint"] for row in spec.band if row["resolvable"]]
    return list(dict.fromkeys(list(spec.verification.main_joints) + resolvable))


def coverage_report(verification: pd.DataFrame, regimes: Mapping[str, str], spec: Round8Spec) -> dict[str, Any]:
    """Observable fraction per joint and regime; the ``full`` gate; the bags to regenerate.

    The gate counts the resolvable joint-bags (``f_n <= f_SG`` for that robot)
    of the gate joints in ``full`` bags.
    """
    table = verification.copy()
    table["regime"] = table["bag"].map(regimes).fillna("unknown")
    joints = gate_joints(spec)
    per_joint: dict[str, Any] = {}
    failing_bags: set[str] = set()
    raise_joints, gate_ok = [], True
    for regime, group in table.groupby("regime"):
        per_joint[regime] = {}
        for joint in [row["joint"] for row in spec.band]:
            resolvable = group[f"resolvable__{joint}"].astype(bool)
            observable = group[f"observable__{joint}"].astype(bool) & resolvable
            n = int(resolvable.sum())
            fraction = float(observable.sum() / n) if n else float("nan")
            per_joint[regime][joint] = {"resolvable_bags": n, "observable_bags": int(observable.sum()),
                                        "observable_fraction": fraction,
                                        "median_K_relerr": float(group.loc[resolvable, f"ident_K_relerr__{joint}"]
                                                                 .median()) if n else float("nan"),
                                        "median_snr": float(group.loc[resolvable, f"elastic_snr__{joint}"]
                                                            .median()) if n else float("nan")}
            if regime == "full" and joint in joints and n:
                if fraction < spec.verification.gate_fraction:
                    gate_ok = False
                if fraction < spec.verification.raise_fraction:
                    raise_joints.append(joint)
                if joint in spec.verification.main_joints:
                    failing_bags |= set(group.loc[resolvable & ~observable, "bag"])
    return {"gate_joints": joints, "per_regime": per_joint, "gate_ok": bool(gate_ok),
            "gate_fraction": spec.verification.gate_fraction, "raise_joints": raise_joints,
            "failing_full_bags": sorted(failing_bags)}


def regimes_of(config: Any, frame: pd.DataFrame) -> dict[str, str]:
    """``bag -> regime`` of an r8 build, recomputed from the seeds (the draw is a pure function of them)."""
    from .dataset_worklist import bag_excitation, split_labels_of

    labels = split_labels_of(config)
    tiers = {t.name: t for t in config.tiers}
    out = {}
    for bag, group in frame.groupby("bag", sort=False):
        tier = tiers[str(group["tier"].iloc[0])]
        index = int(str(bag).split("_")[0][1:])
        out[bag] = bag_excitation(config, tier, index, labels.get(tier.name, "train"))[0]
    return out


def add_bag_columns(frame: pd.DataFrame, regimes: Mapping[str, str], verification: pd.DataFrame | None,
                    attempts: Mapping[str, int] | None = None) -> pd.DataFrame:
    """Write ``probe_regime``, ``regen_attempt`` and the verification columns onto every row of each bag."""
    frame = frame.copy()
    frame["probe_regime"] = frame["bag"].map(regimes).fillna("unknown")
    frame["regen_attempt"] = frame["bag"].map(attempts or {}).fillna(0).astype(int)
    if verification is not None and len(verification):
        columns = [c for c in verification.columns if c not in ("tier", "split")]
        frame = frame.merge(verification[columns], on="bag", how="left")
    return frame


def add_modal_columns(frame: pd.DataFrame, spec: ModalSpec) -> pd.DataFrame:
    """``modal_joint``, ``modal_pose``, ``modal_repeat``, ``modal_f0``, ``modal_f1`` per bag (``tools/modal_fit.py``)."""
    frame = frame.copy()
    bands = {j[0]: (j[1], j[2]) for j in spec.joints}
    index = frame["bag"].map(lambda bag: int(str(bag).split("_")[0][1:]))
    segments = spec.segments
    frame["modal_joint"] = index.map(lambda i: segments[i][0])
    frame["modal_pose"] = index.map(lambda i: spec.poses[segments[i][1]][0])
    frame["modal_repeat"] = index.map(lambda i: segments[i][2]).astype(int)
    frame["modal_f0"] = frame["modal_joint"].map(lambda j: bands[j][0])
    frame["modal_f1"] = frame["modal_joint"].map(lambda j: bands[j][1])
    return frame


def add_inertia_audit(frame: pd.DataFrame, asset: Any, n: int) -> pd.DataFrame:
    """Audit columns ``mjj0..`` = URDF ``M_jj`` at the true link position (the M_phi check, never an input)."""
    from .round7 import _block, _model_for

    pin, model, data = _model_for(asset, None, {})
    q_link = _block(frame, "q_link", n)
    out = np.empty_like(q_link)
    for i in range(len(q_link)):
        out[i] = np.diag(np.asarray(pin.crba(model, data, q_link[i])))
    frame = frame.copy()
    for j in range(n):
        frame[f"mjj{j}"] = out[:, j].astype(np.float32)
    return frame


def urdf_home_inertia(asset: Any, home: Sequence[float]) -> list[float]:
    """``M_ref``: the URDF ``M_jj`` at home, the inertia the derived zeta is reported at (00_plan Sec 3.2)."""
    from .round7 import _model_for

    pin, model, data = _model_for(asset, None, {})
    return [float(v) for v in np.diag(np.asarray(pin.crba(model, data, np.asarray(home, dtype=float))))]


def contract_block(spec: Round8Spec, coverage: Mapping[str, Any] | None, *, variant: str,
                   reference_inertia: Sequence[float], home: Sequence[float]) -> dict[str, Any]:
    """The contract's r8 additions: ``privileged``, ``modal_coverage``, ``round8``."""
    return {
        "privileged": True,
        "privileged_columns": {
            "per_bag": ["stiffness__<joint>", "damping__<joint>", "zeta_filter__<joint>",
                        "link_inertia_median__<joint>", "link_coulomb__<joint>", "link_viscous__<joint>",
                        "K_eff__<joint>", "D_eff__<joint>", "ident_K_relerr_nominal__<joint>",
                        "ident_K__<joint>", "ident_D__<joint>", "ident_K_relerr__<joint>", "ident_D_relerr__<joint>",
                        "elastic_snr__<joint>", "f_n_filter__<joint>", "resolvable__<joint>",
                        "observable__<joint>", "probe_regime", "regen_attempt"],
            "per_sample": ["mjj0..mjj<n-1>"],
            "use": "loss targets and evaluation only (identified_filter/00 Sec 6); the consumer refuses them "
                   "as model inputs",
        },
        "modal_coverage": {"band": [dict(row) for row in spec.band], "comb_hz": list(spec.comb_hz),
                           "rule": "f_n = sqrt(K/M_jj(q))/2pi over the widened K prior and >= 500 URDF "
                                   "configurations in the window; band [0.5 min f_n, min(1.5 max f_n, f_SG)]",
                           "gate": None if coverage is None else dict(coverage)},
        "round8": {**spec.as_dict(), "variant": variant, "reference_inertia_home": list(reference_inertia),
                   "home": list(home)},
    }
