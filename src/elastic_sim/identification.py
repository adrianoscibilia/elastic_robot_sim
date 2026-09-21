"""Rigid-body identification utilities backed by Pinocchio.

These helpers give dataset generation an analytic reference that is
independent of both simulator backends:

* the feedforward torque that drives a torque-controlled rollout,
* the regressor used to score how well a trajectory excites the dynamics,
* the base-parameter basis used to check that a generated dataset really is
  sufficient to identify the model.

The inertial parameters of a serial arm are not all identifiable from joint
torques: only a lower-dimensional set of linear combinations ("base
parameters") is.  The KUKA LBR iiwa 14 has 70 standard parameters and 43 base
parameters, so any recovery check has to be posed in the base space.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .assets import AssetSpec

# Velocity below which Coulomb friction is blended through zero.  A hard sign
# makes the feedforward torque discontinuous and excites the integrator at
# every zero crossing, so the model uses a smooth approximation instead.
COULOMB_EPSILON = 1.0e-3


@dataclass(frozen=True)
class FrictionModel:
    """Per-joint viscous and Coulomb friction coefficients."""

    viscous: np.ndarray
    coulomb: np.ndarray

    def __post_init__(self) -> None:
        viscous = np.asarray(self.viscous, dtype=float).reshape(-1)
        coulomb = np.asarray(self.coulomb, dtype=float).reshape(-1)
        if viscous.shape != coulomb.shape:
            raise ValueError("viscous and coulomb must have the same length")
        if np.any(viscous < 0.0) or np.any(coulomb < 0.0):
            raise ValueError("friction coefficients must be non-negative")
        object.__setattr__(self, "viscous", viscous)
        object.__setattr__(self, "coulomb", coulomb)

    @property
    def n_dof(self) -> int:
        return int(self.viscous.size)

    def torque(self, dq: np.ndarray, *, epsilon: float = COULOMB_EPSILON) -> np.ndarray:
        dq = np.asarray(dq, dtype=float)
        return self.viscous * dq + self.coulomb * np.tanh(dq / epsilon)

    @classmethod
    def from_asset(cls, asset: AssetSpec) -> "FrictionModel":
        """Read the ``<dynamics damping/friction>`` values from the URDF."""
        joints = asset.resolve_active_joints()
        viscous = [0.0 if joint.damping is None else float(joint.damping) for joint in joints]
        coulomb = [0.0 if joint.friction is None else float(joint.friction) for joint in joints]
        return cls(np.asarray(viscous), np.asarray(coulomb))


def build_model(asset: AssetSpec) -> tuple[Any, Any, Any]:
    """Return ``(pinocchio, model, data)`` for an asset's URDF."""
    import pinocchio as pin

    model = pin.buildModelFromUrdf(str(Path(asset.urdf_path).resolve()))
    expected = tuple(asset.joint_names)
    present = tuple(model.names[index] for index in range(1, model.njoints))
    if present != expected:
        raise ValueError(
            f"Asset {asset.name!r} active joints {expected} do not match the "
            f"Pinocchio joint order {present}; identification assumes they agree"
        )
    return pin, model, model.createData()


def inverse_dynamics(
    pin: Any,
    model: Any,
    data: Any,
    q: np.ndarray,
    dq: np.ndarray,
    ddq: np.ndarray,
    *,
    friction: FrictionModel | None = None,
    epsilon: float = COULOMB_EPSILON,
) -> np.ndarray:
    """Return the joint torque implied by the rigid-body model plus friction."""
    tau = np.asarray(pin.rnea(model, data, np.asarray(q, dtype=float),
                              np.asarray(dq, dtype=float), np.asarray(ddq, dtype=float)), dtype=float)
    if friction is not None:
        tau = tau + friction.torque(dq, epsilon=epsilon)
    return tau


def standard_parameters(pin: Any, model: Any) -> np.ndarray:
    """Return the URDF's 10-per-body inertial parameter vector."""
    return np.concatenate([model.inertias[index].toDynamicParameters() for index in range(1, model.njoints)])


def joint_torque_regressor(
    pin: Any,
    model: Any,
    data: Any,
    q: np.ndarray,
    dq: np.ndarray,
    ddq: np.ndarray,
    *,
    include_friction: bool = False,
    epsilon: float = COULOMB_EPSILON,
) -> np.ndarray:
    """Return the ``(n_dof, n_params)`` regressor for one sample.

    With ``include_friction`` the rigid-body block is augmented by a diagonal
    viscous block and a diagonal (smoothed) Coulomb block, so the parameter
    vector becomes ``[inertial(10*nb), viscous(n), coulomb(n)]``.
    """
    dq = np.asarray(dq, dtype=float)
    block = np.asarray(
        pin.computeJointTorqueRegressor(model, data, np.asarray(q, dtype=float), dq, np.asarray(ddq, dtype=float)),
        dtype=float,
    )
    if not include_friction:
        return block
    return np.hstack([block, np.diag(dq), np.diag(np.tanh(dq / epsilon))])


def stack_regressor(
    pin: Any,
    model: Any,
    data: Any,
    q: np.ndarray,
    dq: np.ndarray,
    ddq: np.ndarray,
    *,
    include_friction: bool = False,
    epsilon: float = COULOMB_EPSILON,
) -> np.ndarray:
    """Stack :func:`joint_torque_regressor` over a sampled trajectory."""
    q = np.atleast_2d(np.asarray(q, dtype=float))
    dq = np.atleast_2d(np.asarray(dq, dtype=float))
    ddq = np.atleast_2d(np.asarray(ddq, dtype=float))
    if not (len(q) == len(dq) == len(ddq)):
        raise ValueError("q, dq and ddq must have the same number of samples")
    return np.vstack([
        joint_torque_regressor(pin, model, data, q[i], dq[i], ddq[i],
                               include_friction=include_friction, epsilon=epsilon)
        for i in range(len(q))
    ])


def base_parameter_basis(
    pin: Any,
    model: Any,
    data: Any,
    *,
    n_samples: int = 400,
    seed: int = 0,
    include_friction: bool = False,
) -> np.ndarray:
    """Return an orthonormal basis ``V1`` of the identifiable parameter space.

    ``W @ theta == (W @ V1) @ (V1.T @ theta)`` for every state, so ``V1``
    converts both the regressor and the true parameters into the base space
    where a least-squares fit is well posed.
    """
    rng = np.random.default_rng(seed)
    # ``pin.randomConfiguration`` draws from Pinocchio's own global RNG, not
    # ``rng``, so the basis depended on how many prior Pinocchio random draws
    # had happened elsewhere in the process -- different between a call made
    # inside ``generate()`` and one made fresh afterwards, changing
    # ``condition_number`` (and so ``digest()``) for a bit-identical
    # trajectory (R3_12 Sec 2.3). Reset Pinocchio's global RNG to a state
    # that depends only on ``seed`` immediately before drawing from it,
    # rather than reimplementing the draw with ``rng.uniform`` over
    # ``[lower, upper]`` (wrong for a continuous joint, whose Pinocchio
    # configuration is `(cos, sin)`, not an angle in a box -- R3_14 Sec 2).
    pin.seed(int(seed))
    rows = []
    for _ in range(int(n_samples)):
        q = pin.randomConfiguration(model)
        dq = rng.normal(size=model.nv)
        ddq = rng.normal(size=model.nv)
        rows.append(joint_torque_regressor(pin, model, data, q, dq, ddq, include_friction=include_friction))
    stacked = np.vstack(rows)
    _, singular, vt = np.linalg.svd(stacked, full_matrices=False)
    tolerance = singular[0] * max(stacked.shape) * np.finfo(float).eps
    rank = int((singular > tolerance).sum())
    return vt[:rank].T


def regressor_condition(
    pin: Any,
    model: Any,
    data: Any,
    q: np.ndarray,
    dq: np.ndarray,
    ddq: np.ndarray,
    basis: np.ndarray,
    *,
    include_friction: bool = False,
) -> float:
    """Condition number of the base regressor over a sampled trajectory.

    This is the standard excitation criterion: the lower it is, the less a
    given torque-measurement error is amplified into parameter error.
    """
    stacked = stack_regressor(pin, model, data, q, dq, ddq, include_friction=include_friction)
    projected = stacked @ basis
    singular = np.linalg.svd(projected, compute_uv=False)
    smallest = float(singular[-1])
    if not np.isfinite(smallest) or smallest <= 0.0:
        return float("inf")
    return float(singular[0] / smallest)


def identify_base_parameters(
    pin: Any,
    model: Any,
    data: Any,
    q: np.ndarray,
    dq: np.ndarray,
    ddq: np.ndarray,
    tau: np.ndarray,
    basis: np.ndarray,
    *,
    include_friction: bool = False,
) -> tuple[np.ndarray, float]:
    """Least-squares fit of base parameters to measured torques.

    Returns ``(theta_base, residual_rms)``.
    """
    stacked = stack_regressor(pin, model, data, q, dq, ddq, include_friction=include_friction)
    projected = stacked @ basis
    target = np.asarray(tau, dtype=float).reshape(-1)
    if len(target) != len(projected):
        raise ValueError("tau must have one row per sample and joint")
    solution, *_ = np.linalg.lstsq(projected, target, rcond=None)
    residual = projected @ solution - target
    return solution, float(np.sqrt(np.mean(residual**2)))
