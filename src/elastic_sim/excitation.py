"""Periodic excitation trajectories for dynamic model identification.

Random point-to-point motion moves a robot around but does not necessarily
*excite* its dynamics: whole groups of inertial parameters can stay nearly
unobservable, which makes any later identification ill-conditioned.  The
standard remedy is a finite Fourier series per joint whose coefficients are
optimized so the base-parameter regressor is well conditioned.

For joint ``i`` with base frequency ``w`` and ``H`` harmonics::

    q_i(t)   = q_i0 + sum_k [  a_ik/(k w) sin(k w t) - b_ik/(k w) cos(k w t) ]
    dq_i(t)  =        sum_k [  a_ik       cos(k w t) + b_ik       sin(k w t) ]
    ddq_i(t) =        sum_k [ -a_ik (k w) sin(k w t) + b_ik (k w) cos(k w t) ]

Constraining ``sum_k a_ik = 0`` and ``sum_k k b_ik = 0`` makes velocity and
acceleration vanish at ``t = 0``; because the series is periodic they vanish
again at the end of every period, so the robot starts and stops at rest and a
trajectory can be repeated back to back or averaged over periods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .assets import AssetSpec
from .materialized import MaterializedTrajectory


@dataclass(frozen=True)
class FourierExcitationConfig:
    """Shape and limits of a periodic excitation trajectory."""

    n_harmonics: int = 5
    base_frequency: float = 0.1
    n_periods: int = 1
    time_step: float = 0.002
    limit_margin: float = 0.12
    max_acceleration: float = 3.0
    velocity_fraction: float = 0.6
    # Fraction of the feasible operating band the trajectory centre is drawn
    # from: 0 pins every trajectory to the middle of the joint range, 1 spreads
    # them over the whole band the amplitude leaves free.  Spreading the centre
    # is what varies the gravity load, and so what a dataset learns from.
    centre_jitter: float = 0.0
    settle_time: float = 0.0
    # Small-amplitude high harmonics superimposed on the main series to put
    # energy at the transmission resonances (50-170 Hz on A1-A6), without
    # which the sampled damping ratios leave no signature in the data at all
    # and the stiffness is only visible as the static deflection tau/k
    # (R3_02 F3).  Indices are multiples of base_frequency and must exceed
    # n_harmonics; empty disables the probe.
    probe_harmonics: tuple[int, ...] = ()
    # Fraction of max_acceleration reserved for the probe; the main harmonics
    # get the rest.
    probe_acceleration_fraction: float = 0.2
    # Per-joint ``(lower, upper)`` [rad], overriding the URDF limit for that
    # joint before ``limit_margin`` is applied.  Empty means every joint keeps
    # its URDF limit (the iiwa's historical behaviour).  Needed on robots
    # whose URDF limits are much wider than a sensible excitation band (e.g.
    # the UR10's +-2 pi on five joints): without it, ``centre_jitter`` spreads
    # trajectory centres over the whole +-2 pi box, most of which is either a
    # dynamical duplicate (wrap-around) or collides with the table.  Stored as
    # the raw ``(joint_name, (lo, hi))`` pairs rather than resolved to the
    # active-joint order, because this config has no asset reference of its
    # own; ``joint_bounds``/``effective_position_window`` resolve it once the
    # asset (and so the active-joint order) is known.
    position_window: tuple[tuple[str, tuple[float, float]], ...] = ()
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.n_harmonics < 2:
            raise ValueError("n_harmonics must be at least two")
        if self.base_frequency <= 0.0 or self.time_step <= 0.0:
            raise ValueError("base_frequency and time_step must be positive")
        if self.n_periods < 1:
            raise ValueError("n_periods must be at least one")
        if not 0.0 <= self.limit_margin < 0.5:
            raise ValueError("limit_margin must be in [0, 0.5)")
        if self.max_acceleration <= 0.0:
            raise ValueError("max_acceleration must be positive")
        if not 0.0 < self.velocity_fraction <= 1.0:
            raise ValueError("velocity_fraction must be in (0, 1]")
        if not 0.0 <= self.centre_jitter <= 1.0:
            raise ValueError("centre_jitter must be in [0, 1]")
        if self.probe_harmonics:
            probe = np.asarray(self.probe_harmonics)
            if np.any(np.diff(probe) <= 0):
                raise ValueError("probe_harmonics must be strictly increasing")
            if probe[0] <= self.n_harmonics:
                raise ValueError("probe_harmonics must all be greater than n_harmonics")
            if not 0.0 < self.probe_acceleration_fraction < 1.0:
                raise ValueError("probe_acceleration_fraction must be in (0, 1)")
            # The probe's top frequency must sit below the output grid's
            # Nyquist with margin, or it aliases into the written dataset
            # instead of showing up as the modal response it is meant to be.
            nyquist_margin = 0.4 / self.time_step
            if probe[-1] * self.base_frequency >= nyquist_margin:
                raise ValueError(
                    f"probe_harmonics top frequency {probe[-1] * self.base_frequency:g} Hz would alias "
                    f"the {1.0 / self.time_step:g} Hz output grid (keep it below {nyquist_margin:g} Hz)"
                )
        for name, (lo, hi) in self.position_window:
            if lo >= hi:
                raise ValueError(f"excitation.position_window[{name!r}] = [{lo}, {hi}] must satisfy lo < hi")

    @property
    def period(self) -> float:
        return 1.0 / self.base_frequency

    @property
    def duration(self) -> float:
        return self.n_periods * self.period

    @property
    def harmonic_indices(self) -> np.ndarray:
        """Main-harmonic indices followed by the probe's, as one vector."""
        return np.concatenate([np.arange(1, self.n_harmonics + 1), np.asarray(self.probe_harmonics, dtype=int)])


def project_coefficients(
    a: np.ndarray, b: np.ndarray, indices: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Enforce zero velocity and acceleration at the period boundaries.

    ``dq(0) = sum_k a_ik`` and ``ddq(0) = w * sum_k k b_ik``, so ``a`` is
    projected off the all-ones direction and ``b`` off the harmonic-index
    direction.  ``indices`` are the harmonic index of each column (default
    ``1..H``); a modal probe uses non-consecutive high indices here.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    harmonics = np.arange(1, a.shape[1] + 1, dtype=float) if indices is None else np.asarray(indices, dtype=float)
    a = a - a.mean(axis=1, keepdims=True)
    b = b - np.outer(b @ harmonics / float(harmonics @ harmonics), harmonics)
    return a, b


def evaluate_series(
    a: np.ndarray, b: np.ndarray, offset: np.ndarray, omega: float, time: np.ndarray,
    indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(q, dq, ddq)`` sampled at ``time`` for the Fourier series.

    ``indices`` are the harmonic index of each column of ``a``/``b`` (default
    ``1..a.shape[1]``); a modal probe passes non-consecutive high indices.
    """
    harmonics = np.arange(1, a.shape[1] + 1, dtype=float) if indices is None else np.asarray(indices, dtype=float)
    scaled = harmonics * omega
    phase = np.outer(np.asarray(time, dtype=float), scaled)
    sin, cos = np.sin(phase), np.cos(phase)
    q = sin @ (a / scaled).T - cos @ (b / scaled).T + np.asarray(offset, dtype=float)
    dq = cos @ a.T + sin @ b.T
    ddq = -(sin @ (a * scaled).T) + cos @ (b * scaled).T
    return q, dq, ddq


class _AnalyticEvaluator:
    """Precomputed ``t -> (q, dq, ddq)`` for one Fourier series.

    Matches ``evaluate_series`` exactly at a single time point; used by
    ``MaterializedTrajectory.__call__`` so a rollout samples the analytic
    signal at its own physics time step instead of interpolating the
    trajectory's (typically much coarser) recorded grid (R3_12 Sec 2.1).

    A plain class holding numpy arrays, not a closure: ``generate()``'s
    ``--jobs N`` path sends each bag's ``MaterializedTrajectory`` (analytic
    evaluator included) across a process-pool boundary, which requires
    pickling it -- a closure over local variables is not picklable, this is.
    """

    def __init__(self, a: np.ndarray, b: np.ndarray, offset: np.ndarray, omega: float, indices: np.ndarray):
        harmonics = np.asarray(indices, dtype=float)
        scaled = harmonics * omega
        self._scaled = scaled
        self._a_over_scaled = (a / scaled).T
        self._b_over_scaled = (b / scaled).T
        self._a_t = a.T
        self._b_t = b.T
        self._a_scaled_t = (a * scaled).T
        self._b_scaled_t = (b * scaled).T
        self._offset = np.asarray(offset, dtype=float)

    def __call__(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        phase = self._scaled * float(t)
        sin, cos = np.sin(phase), np.cos(phase)
        q = sin @ self._a_over_scaled - cos @ self._b_over_scaled + self._offset
        dq = cos @ self._a_t + sin @ self._b_t
        ddq = -(sin @ self._a_scaled_t) + cos @ self._b_scaled_t
        return q, dq, ddq


def _make_analytic_evaluator(a: np.ndarray, b: np.ndarray, offset: np.ndarray, omega: float, indices: np.ndarray):
    return _AnalyticEvaluator(a, b, offset, omega, indices)


def log_spaced_probe_harmonics(
    f_min: float, f_max: float, n_lines: int, base_frequency: float
) -> tuple[int, ...]:
    """Harmonic indices for a log-spaced modal probe from ``f_min`` to ``f_max`` Hz.

    A sparse, widely-spaced comb cannot satisfy its own acceptance test: a
    linear system only responds at the excitation frequencies, so the
    deflection peak lands on the nearest *line*, not the true mode, and
    ``zeta`` only changes the response within about the half-power bandwidth
    ``2*zeta*f`` of the mode (~10 Hz at zeta=0.05, f=100 Hz) -- with lines
    much further apart than that, most robots' damping stays unobservable
    even with the probe on (``R3_10 Sec 4.3``). Log spacing keeps adjacent
    -line spacing proportionally tight across the whole band with a modest
    line count, and matches how transmission modes are actually distributed
    (`sqrt(k/J)`, roughly log-uniform in ``k``).
    """
    indices = np.unique(np.round(np.logspace(np.log10(f_min), np.log10(f_max), n_lines) / base_frequency).astype(int))
    return tuple(int(i) for i in indices)


def effective_position_window(asset: AssetSpec, config: FourierExcitationConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return per-active-joint ``(lower, upper)`` [rad], before ``limit_margin``.

    The URDF limit (or ``[-pi, pi]`` when the URDF declares none, or a
    malformed one), intersected with ``config.position_window`` for joints it
    names.  Also used by ``link_inertia_envelope`` so the sampled inertia
    envelope reflects the same joint range the excitation trajectories
    actually live in (R4_03 Sec 1).
    """
    joints = asset.resolve_active_joints()
    window = dict(config.position_window)
    if window:
        unknown = sorted(set(window) - {joint.name for joint in joints})
        if unknown:
            available = ", ".join(joint.name for joint in joints)
            raise ValueError(
                f"excitation.position_window has unknown joint name(s) {unknown}; available: {available}"
            )
    lower, upper = [], []
    for joint in joints:
        lo, hi = joint.lower, joint.upper
        if lo is None or hi is None or not np.isfinite([lo, hi]).all() or lo >= hi:
            lo, hi = -np.pi, np.pi
        if joint.name in window:
            window_lo, window_hi = window[joint.name]
            lo, hi = max(lo, window_lo), min(hi, window_hi)
            if lo >= hi:
                raise ValueError(
                    f"excitation.position_window[{joint.name!r}] = [{window_lo}, {window_hi}] does not "
                    f"intersect the joint's admissible range [{joint.lower}, {joint.upper}]"
                )
        lower.append(float(lo))
        upper.append(float(hi))
    return np.asarray(lower), np.asarray(upper)


def _position_window_metadata_entry(config: FourierExcitationConfig) -> dict:
    """The ``{"position_window": ...}`` entry ``optimize_excitation`` merges into its metadata dict.

    Empty (no key at all, not even ``null``) when ``config.position_window``
    is unset, so a config without one -- every shipped config before this
    round -- gets a bit-identical metadata dict and so an unchanged
    ``trajectory_digest`` (R4_03 Sec 1).  Factored out of
    ``optimize_excitation`` so this can be checked directly without building
    a Pinocchio model.
    """
    if not config.position_window:
        return {}
    return {"position_window": [[name, list(bounds)] for name, bounds in config.position_window]}


def joint_bounds(asset: AssetSpec, config: FourierExcitationConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(safe_lower, safe_upper, velocity_limit)`` for the active joints."""
    joints = asset.resolve_active_joints()
    lower, upper = effective_position_window(asset, config)
    velocity = np.asarray([float(joint.velocity) if joint.velocity else 1.0 for joint in joints])
    span = upper - lower
    return (
        lower + config.limit_margin * span,
        upper - config.limit_margin * span,
        velocity * config.velocity_fraction,
    )


def _scale_block(
    a: np.ndarray, b: np.ndarray, omega: float, time: np.ndarray, indices: np.ndarray | None,
    half_span: np.ndarray, velocity_limit: np.ndarray, acceleration_budget: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Scale one coefficient block to the largest amplitude within its budget.

    Position deviation, velocity and acceleration are all linear in the
    coefficients, so a single per-joint factor makes every limit feasible at
    once and one pass is exact.  Scaling *up* is allowed and wanted: a larger
    feasible amplitude excites the dynamics more.
    """
    q, dq, ddq = evaluate_series(a, b, np.zeros(a.shape[0]), omega, time, indices=indices)
    deviation = 0.5 * (q.max(axis=0) - q.min(axis=0))
    peak_velocity = np.abs(dq).max(axis=0)
    peak_acceleration = np.abs(ddq).max(axis=0)

    def _ratio(limit: np.ndarray, peak: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(peak > 1.0e-12, limit / np.maximum(peak, 1.0e-12), np.inf)

    scale = np.minimum.reduce([
        _ratio(half_span, deviation),
        _ratio(velocity_limit, peak_velocity),
        _ratio(acceleration_budget, peak_acceleration),
    ])
    scale = np.where(np.isfinite(scale), scale, 1.0)
    return a * scale[:, None], b * scale[:, None]


def _fit_to_limits(
    a: np.ndarray,
    b: np.ndarray,
    omega: float,
    time: np.ndarray,
    safe_lower: np.ndarray,
    safe_upper: np.ndarray,
    velocity_limit: np.ndarray,
    max_acceleration: float,
    rng: np.random.Generator | None = None,
    centre_jitter: float = 0.0,
    indices: np.ndarray | None = None,
    probe_count: int = 0,
    probe_acceleration_fraction: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scale each joint's series to the largest feasible amplitude.

    With ``probe_count == 0`` (the default) this is a single-budget scaling,
    bit-identical to before the modal probe existed.  With ``probe_count >
    0``, the last ``probe_count`` columns of ``a``/``b`` are the probe.

    The probe is scaled **first**, against its own acceleration budget alone
    (position and velocity unconstrained): at the probe's high frequencies,
    acceleration is `(2 pi f)^2` times the position amplitude, so its
    position/velocity footprint is negligible (micro-radians at 40+ Hz) and
    reserving it costs the main trajectory nothing measurable.  The main
    harmonics are then scaled against the *remaining* position, velocity and
    ``(1 - probe_acceleration_fraction)`` of the acceleration budget.

    Scaling the main harmonics first and giving the probe only what was
    "left over" (the original approach) is backwards: on any joint where
    position or velocity is the binding limit for the main trajectory --
    true for most of the proximal joints at the shipped settings -- the
    leftover budget is exactly zero and the probe's scale collapses to zero
    with it, silently defeating the whole point of R3_02 (see R3_10 Sec 2.1).
    """
    half_span = 0.5 * (safe_upper - safe_lower)
    n_joints = a.shape[0]
    n_main = a.shape[1] - probe_count
    if probe_count == 0:
        a, b = _scale_block(a, b, omega, time, indices, half_span, velocity_limit,
                            np.full(n_joints, max_acceleration))
    else:
        main_indices = None if indices is None else indices[:n_main]
        probe_indices = None if indices is None else indices[n_main:]
        unconstrained = np.full(n_joints, np.inf)
        # The probe's top line can be tens of Hz; scaling and budget-checking
        # it against the ~2 ms output grid (< 3 samples/period near 190 Hz)
        # can miss the true continuous peak entirely (R3_12 Sec 2.1). Use a
        # much finer grid for the probe block's own scaling and for measuring
        # how much position/velocity headroom it leaves the main harmonics.
        fine_step = min(float(time[1] - time[0]), 1.0e-4)
        fine_time = np.arange(time[0], time[-1] + 0.5 * fine_step, fine_step)
        a_probe, b_probe = _scale_block(
            a[:, n_main:], b[:, n_main:], omega, fine_time, probe_indices, unconstrained, unconstrained,
            np.full(n_joints, probe_acceleration_fraction * max_acceleration),
        )
        q_probe, dq_probe, _ = evaluate_series(
            a_probe, b_probe, np.zeros(a_probe.shape[0]), omega, fine_time, indices=probe_indices
        )
        remaining_half_span = np.maximum(half_span - 0.5 * (q_probe.max(axis=0) - q_probe.min(axis=0)), 0.0)
        remaining_velocity = np.maximum(velocity_limit - np.abs(dq_probe).max(axis=0), 0.0)
        a_main, b_main = _scale_block(
            a[:, :n_main], b[:, :n_main], omega, time, main_indices, remaining_half_span, remaining_velocity,
            np.full(n_joints, (1.0 - probe_acceleration_fraction) * max_acceleration),
        )
        a = np.concatenate([a_main, a_probe], axis=1)
        b = np.concatenate([b_main, b_probe], axis=1)

    q, _, _ = evaluate_series(a, b, np.zeros(a.shape[0]), omega, time, indices=indices)
    # Offsets that keep the achieved range inside the safe band.  Their middle
    # centres the motion in the joint range, as every trajectory did before
    # ``centre_jitter``; anywhere in between is equally feasible.
    lowest, highest = safe_lower - q.min(axis=0), safe_upper - q.max(axis=0)
    middle = 0.5 * (lowest + highest)
    if rng is None or centre_jitter <= 0.0:
        return a, b, middle
    drawn = rng.uniform(np.minimum(lowest, highest), np.maximum(lowest, highest))
    return a, b, middle + centre_jitter * (drawn - middle)


def sample_candidate(
    asset: AssetSpec,
    config: FourierExcitationConfig,
    rng: np.random.Generator,
    time: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw one feasible coefficient set ``(a, b, offset)``.

    ``a``/``b`` carry ``n_harmonics`` main columns followed by
    ``len(probe_harmonics)`` probe columns, in that order; ``offset`` is
    computed from the combined signal.  The two blocks are projected to zero
    velocity/acceleration at ``t=0`` *independently*: they are later scaled
    by different per-joint factors (main harmonics against most of the
    acceleration budget, the probe against what is left), and only an
    independently-zero sub-sum stays zero after an arbitrary rescale -- a
    single projection of the combined matrix would not.
    """
    n = len(asset.joint_names)
    safe_lower, safe_upper, velocity_limit = joint_bounds(asset, config)
    indices = config.harmonic_indices
    n_main = config.n_harmonics
    total = len(indices)
    a = rng.normal(size=(n, total))
    b = rng.normal(size=(n, total))
    a[:, :n_main], b[:, :n_main] = project_coefficients(a[:, :n_main], b[:, :n_main], indices=indices[:n_main])
    if total > n_main:
        a[:, n_main:], b[:, n_main:] = project_coefficients(a[:, n_main:], b[:, n_main:], indices=indices[n_main:])
    omega = 2.0 * np.pi * config.base_frequency
    return _fit_to_limits(
        a, b, omega, time, safe_lower, safe_upper, velocity_limit, config.max_acceleration,
        rng=rng, centre_jitter=config.centre_jitter, indices=indices,
        probe_count=len(config.probe_harmonics), probe_acceleration_fraction=config.probe_acceleration_fraction,
    )


def optimize_excitation(
    asset: AssetSpec,
    config: FourierExcitationConfig | None = None,
    *,
    seed: int = 0,
    n_candidates: int = 96,
    refine_iterations: int = 0,
    kinematics: Any | None = None,
    include_friction: bool = True,
    condition_stride: int = 10,
) -> MaterializedTrajectory:
    """Search for a well-conditioned, feasible periodic excitation trajectory.

    Candidates are drawn at random, scaled to the joint limits and scored by
    the condition number of the base-parameter regressor.  The best feasible
    candidate is returned as a :class:`MaterializedTrajectory`, so trajectory
    JSON, the parquet writer, the collision checker and both simulator
    backends consume it unchanged.
    """
    from . import identification as idn

    config = config or FourierExcitationConfig()
    pin, model, data = idn.build_model(asset)
    basis = idn.base_parameter_basis(pin, model, data, seed=seed, include_friction=include_friction)
    time = np.arange(0.0, config.duration + 0.5 * config.time_step, config.time_step)
    omega = 2.0 * np.pi * config.base_frequency
    rng = np.random.default_rng(seed)

    indices = config.harmonic_indices
    n_main = config.n_harmonics
    best: dict[str, Any] | None = None
    rejected_for_collision = 0
    for index in range(int(n_candidates)):
        a, b, offset = sample_candidate(asset, config, rng, time)
        q, dq, ddq = evaluate_series(a, b, offset, omega, time, indices=indices)
        # Conditioning is scored on the main harmonics only.  The probe
        # exists to excite the transmission modes, not to condition the
        # rigid-body regressor -- including it here would both alias
        # (condition_stride samples the 2 ms grid at ~50 Hz, well below the
        # probe's frequencies) and conflate two separate design objectives
        # (R3_02 Sec 2.6).  A strided subset is otherwise enough, since
        # conditioning is a property of the sampled state distribution.
        q_main, dq_main, ddq_main = evaluate_series(a[:, :n_main], b[:, :n_main], offset, omega, time)
        condition = idn.regressor_condition(
            pin, model, data, q_main[::condition_stride], dq_main[::condition_stride], ddq_main[::condition_stride],
            basis, include_friction=include_friction,
        )
        if not np.isfinite(condition):
            continue
        if best is not None and condition >= best["condition"]:
            continue
        if kinematics is not None:
            report = kinematics.validate_path(
                q,
                margin=float(asset.metadata.get("collision", {}).get("margin", 0.0)),
                max_joint_step=float(asset.metadata.get("collision", {}).get("max_joint_step", 0.05)),
            )
            if not report.valid:
                rejected_for_collision += 1
                continue
        best = {"condition": condition, "a": a, "b": b, "offset": offset, "index": index,
                "q": q, "dq": dq, "ddq": ddq}

    if best is None:
        raise RuntimeError(
            f"No feasible excitation trajectory found for {asset.name!r} in {n_candidates} candidates "
            f"({rejected_for_collision} rejected for collision)"
        )

    metadata = {
        "generator": "fourier_excitation",
        "asset": asset.name,
        "seed": int(seed),
        "condition_number": float(best["condition"]),
        "n_harmonics": int(config.n_harmonics),
        "base_frequency": float(config.base_frequency),
        "n_periods": int(config.n_periods),
        "limit_margin": float(config.limit_margin),
        "max_acceleration": float(config.max_acceleration),
        "velocity_fraction": float(config.velocity_fraction),
        "centre_jitter": float(config.centre_jitter),
        "probe_harmonics": list(int(v) for v in config.probe_harmonics),
        "probe_acceleration_fraction": float(config.probe_acceleration_fraction),
        "probe_top_hz": float(config.probe_harmonics[-1] * config.base_frequency) if config.probe_harmonics else 0.0,
        "candidates": int(n_candidates),
        "candidate_index": int(best["index"]),
        "collision_rejections": int(rejected_for_collision),
        "include_friction": bool(include_friction),
        # Part of the trajectory's identity (it changes the feasible band the
        # coefficients were fit against), so it must enter trajectory_digest()
        # -- but only when set, or the iiwa's metadata dict (and so its
        # digest()) would change with no behaviour change (signal_digest(),
        # which excludes metadata, is unaffected either way) (R4_03 Sec 1).
        **_position_window_metadata_entry(config),
        "coefficients_a": np.asarray(best["a"]).tolist(),
        "coefficients_b": np.asarray(best["b"]).tolist(),
        "offset": np.asarray(best["offset"]).tolist(),
        **dict(config.metadata),
    }
    return MaterializedTrajectory(
        time=time,
        position=best["q"],
        velocity=best["dq"],
        acceleration=best["ddq"],
        joint_names=tuple(asset.joint_names),
        metadata=metadata,
        analytic=_make_analytic_evaluator(best["a"], best["b"], best["offset"], omega, indices),
    )


def trajectory_from_metadata(
    asset: AssetSpec, metadata: dict, *, time_step: float | None = None
) -> MaterializedTrajectory:
    """Rebuild an excitation trajectory from its stored coefficients."""
    a = np.asarray(metadata["coefficients_a"], dtype=float)
    b = np.asarray(metadata["coefficients_b"], dtype=float)
    offset = np.asarray(metadata["offset"], dtype=float)
    base_frequency = float(metadata["base_frequency"])
    n_periods = int(metadata.get("n_periods", 1))
    step = float(time_step if time_step is not None else metadata.get("time_step", 0.002))
    duration = n_periods / base_frequency
    time = np.arange(0.0, duration + 0.5 * step, step)
    n_harmonics = int(metadata["n_harmonics"])
    probe_harmonics = tuple(metadata.get("probe_harmonics", ()) or ())
    indices = np.concatenate([np.arange(1, n_harmonics + 1), np.asarray(probe_harmonics, dtype=int)]) \
        if a.shape[1] != n_harmonics else None
    omega = 2.0 * np.pi * base_frequency
    q, dq, ddq = evaluate_series(a, b, offset, omega, time, indices=indices)
    eval_indices = indices if indices is not None else np.arange(1, n_harmonics + 1)
    return MaterializedTrajectory(
        time=time, position=q, velocity=dq, acceleration=ddq,
        joint_names=tuple(asset.joint_names), metadata=dict(metadata),
        analytic=_make_analytic_evaluator(a, b, offset, omega, eval_indices),
    )
