"""Column layout and per-sample rows of the elastic-joint regressor.

Zollo's model (Spong-type joints, rotor-link coupling dropped: the simulator
is Spong's reduced model, `00_recon.md` Sec 2), with ``delta = theta - q``:

* link rows, one per joint::

      Y(q, dq, ddq) pi + f_l(dq) - K delta - D ddelta = 0          (variant A)
      Y(q, dq, ddq) pi + f_l(dq)                       = tau_J      (variant B)

* second rows, one per joint::

      B ddtheta + f_v dtheta + f_c sgn(dtheta) + K delta + D ddelta = u      (A: motor)
      K delta + D ddelta                                            = tau_J  (B: spring)

``Y`` is Pinocchio's joint-torque regressor (10 standard parameters per
body, ``[m, mc_x, mc_y, mc_z, I_xx, I_xy, I_yy, I_xz, I_yz, I_zz]`` about the
joint frame).  The link friction ``f_l`` and the spring damping ``D`` are not
in Zollo's model; they are in the simulator's, and each has its flag.  The
columns are ordered as Zollo groups them -- every mass, every first moment,
every inertia tensor, then the motor, friction and transmission columns -- so
the elimination keeps the leading column of each dependent set, as
Gauss-Jordan does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

INERTIAL = ("m", "mcx", "mcy", "mcz", "Ixx", "Ixy", "Iyy", "Ixz", "Iyz", "Izz")
FIRST_MOMENT = (1, 2, 3)
INERTIA = (4, 5, 6, 7, 8, 9)

# Groups whose columns may be duplicated per robot (the per-robot draws).
TRANSMISSION_GROUPS = ("rotor", "motor_viscous", "motor_coulomb", "stiffness", "damping")
# The motor-side groups alone: variant A-pr's nuisance parameters (04_variant_A_elastic.md).
MOTOR_GROUPS = ("rotor", "motor_viscous", "motor_coulomb")
VARIANTS = ("A", "B")


@dataclass(frozen=True)
class Column:
    """One unknown: its group, joint (or body), component and robot."""

    group: str
    joint: int
    component: str = ""
    robot: str | None = None

    @property
    def name(self) -> str:
        label = f"{self.group}{'.' + self.component if self.component else ''}[{self.joint + 1}]"
        return label if self.robot is None else f"{label}@{self.robot}"


@dataclass(frozen=True)
class Layout:
    """Which unknowns the regressor carries and in what order."""

    n: int
    variant: str = "B"
    link_friction: bool = True
    spring_damping: bool = True
    robots: tuple[str, ...] | None = None
    per_robot_groups: tuple[str, ...] = TRANSMISSION_GROUPS
    columns: tuple[Column, ...] = field(default=(), compare=False)

    @classmethod
    def build(cls, n: int, variant: str = "B", *, link_friction: bool = True, spring_damping: bool = True,
              robots: Sequence[str] | None = None,
              per_robot_groups: Sequence[str] = TRANSMISSION_GROUPS) -> "Layout":
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
        unknown = set(per_robot_groups) - set(TRANSMISSION_GROUPS)
        if unknown:
            raise ValueError(f"only {TRANSMISSION_GROUPS} can be per robot, got {sorted(unknown)}")
        robots = None if robots is None else tuple(str(r) for r in robots)
        columns: list[Column] = [Column("mass", b, "m") for b in range(n)]
        columns += [Column("first_moment", b, INERTIAL[k]) for b in range(n) for k in FIRST_MOMENT]
        columns += [Column("inertia", b, INERTIAL[k]) for b in range(n) for k in INERTIA]
        groups: list[str] = []
        if variant == "A":
            groups += ["rotor", "motor_viscous", "motor_coulomb"]
        if link_friction:
            groups += ["link_viscous", "link_coulomb"]
        groups += ["stiffness"] + (["damping"] if spring_damping else [])
        for group in groups:
            owners = robots if (robots is not None and group in per_robot_groups) else (None,)
            columns += [Column(group, j, robot=robot) for robot in owners for j in range(n)]
        return cls(n, variant, bool(link_friction), bool(spring_damping), robots, tuple(per_robot_groups),
                   tuple(columns))

    @property
    def size(self) -> int:
        return len(self.columns)

    @property
    def names(self) -> list[str]:
        return [column.name for column in self.columns]

    def index(self, group: str, joint: int, robot: str | None = None) -> int:
        for position, column in enumerate(self.columns):
            if column.group == group and column.joint == joint and column.robot == robot:
                return position
        raise KeyError(f"no column {group}[{joint + 1}]@{robot}")

    def has(self, group: str) -> bool:
        return any(column.group == group for column in self.columns)

    def owner(self, group: str, robot: str | None) -> str | None:
        """The robot whose copy of ``group`` a sample of ``robot`` uses (``None`` when pooled)."""
        if self.robots is None or group not in self.per_robot_groups:
            return None
        if robot not in self.robots:
            raise KeyError(f"robot {robot!r} has no per-robot columns (layout robots {self.robots})")
        return robot

    def active(self, robot: str | None) -> np.ndarray:
        """Layout positions a sample of ``robot`` can touch: the shared columns and its own."""
        return np.asarray([i for i, column in enumerate(self.columns) if column.robot in (None, robot)], dtype=int)

    def inertial_slice(self) -> np.ndarray:
        """Layout positions of the 10n inertial columns, in Pinocchio's parameter order."""
        positions = np.empty(10 * self.n, dtype=int)
        for position, column in enumerate(self.columns):
            if column.group in ("mass", "first_moment", "inertia"):
                positions[10 * column.joint + INERTIAL.index(column.component)] = position
        return positions

    def row_labels(self) -> list[str]:
        second = "motor" if self.variant == "A" else "spring"
        return [f"link[{j + 1}]" for j in range(self.n)] + [f"{second}[{j + 1}]" for j in range(self.n)]

    def describe(self) -> dict[str, Any]:
        return {"n": self.n, "variant": self.variant, "link_friction": self.link_friction,
                "spring_damping": self.spring_damping, "robots": list(self.robots) if self.robots else None,
                "per_robot_groups": list(self.per_robot_groups), "columns": self.names}


def coulomb(velocity: np.ndarray, epsilon: float) -> np.ndarray:
    """``tanh(v / epsilon)``, or ``sign(v)`` for ``epsilon <= 0`` (Zollo's own law)."""
    velocity = np.asarray(velocity, dtype=float)
    return np.sign(velocity) if epsilon <= 0.0 else np.tanh(velocity / epsilon)


def rigid_regressor(pin: Any, model: Any, data: Any, q: np.ndarray, dq: np.ndarray, ddq: np.ndarray) -> np.ndarray:
    """Pinocchio's joint-torque regressor over samples: ``[T, n, 10 n]``."""
    q, dq, ddq = (np.atleast_2d(np.asarray(x, dtype=float)) for x in (q, dq, ddq))
    out = np.empty((len(q), model.nv, 10 * (model.njoints - 1)))
    for i in range(len(q)):
        out[i] = pin.computeJointTorqueRegressor(model, data, q[i], dq[i], ddq[i])
    return out


def sample_rows(layout: Layout, robot: str | None, signals: Mapping[str, np.ndarray], Y: np.ndarray, *,
                epsilon_link: float, epsilon_motor: float
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rows of one bag: ``(W [T, 2n, p], columns [p], target [T, 2n], valid [T, 2n])``.

    ``W`` holds only the ``p`` columns the bag can touch (``columns``, layout
    positions, :meth:`Layout.active`), so a per-robot layout does not
    allocate every other robot's zeros at the full sample rate.
    ``signals`` holds ``theta, dtheta, ddtheta, q, dq, tau_J`` (and ``u`` for
    variant A), each ``[T, n]``; ``Y`` is :func:`rigid_regressor` at the link
    state.  ``robot`` is the bag's tier: ``"rigid"`` has no transmission, so
    its spring rows (B) are invalid and it gives no variant-A rows at all.
    """
    n = layout.n
    T = len(Y)
    columns = layout.active(None if robot == "rigid" else robot)
    local = {int(position): i for i, position in enumerate(columns)}
    W = np.zeros((T, 2 * n, len(columns)))
    target = np.zeros((T, 2 * n))
    valid = np.ones((T, 2 * n), dtype=bool)
    rigid = robot == "rigid"
    if rigid and layout.variant == "A":
        valid[:] = False
        return W, columns, target, valid
    W[:, :n, [local[int(p)] for p in layout.inertial_slice()]] = Y
    q, dq = signals["q"], signals["dq"]
    theta, dtheta = signals["theta"], signals["dtheta"]
    delta, ddelta = theta - q, dtheta - dq
    link_sign = -1.0 if layout.variant == "A" else 0.0
    for j in range(n):
        if layout.link_friction:
            W[:, j, local[layout.index("link_viscous", j)]] = dq[:, j]
            W[:, j, local[layout.index("link_coulomb", j)]] = coulomb(dq[:, j], epsilon_link)
        if rigid:
            continue
        k = local[layout.index("stiffness", j, layout.owner("stiffness", robot))]
        W[:, n + j, k] = delta[:, j]
        if link_sign:
            W[:, j, k] = link_sign * delta[:, j]
        if layout.spring_damping:
            d = local[layout.index("damping", j, layout.owner("damping", robot))]
            W[:, n + j, d] = ddelta[:, j]
            if link_sign:
                W[:, j, d] = link_sign * ddelta[:, j]
        if layout.variant == "A":
            for group, value in (("rotor", signals["ddtheta"][:, j]), ("motor_viscous", dtheta[:, j]),
                                 ("motor_coulomb", coulomb(dtheta[:, j], epsilon_motor))):
                W[:, n + j, local[layout.index(group, j, layout.owner(group, robot))]] = value
    if layout.variant == "A":
        target[:, n:] = signals["u"]
    else:
        target[:, :n] = signals["tau_J"]
        target[:, n:] = signals["tau_J"]
        if rigid:
            valid[:, n:] = False
    return W, columns, target, valid


def nominal_vector(layout: Layout, standard_inertial: np.ndarray,
                   values: Mapping[str, Sequence[float] | Mapping[str, Sequence[float]]] | None = None) -> np.ndarray:
    """A full parameter vector: the URDF inertials plus per-group values.

    ``values[group]`` is a per-joint sequence (pooled) or a mapping robot ->
    sequence (per robot); groups not given are ``nan`` (unknown).
    """
    vector = np.full(layout.size, np.nan)
    vector[layout.inertial_slice()] = np.asarray(standard_inertial, dtype=float)
    for position, column in enumerate(layout.columns):
        if column.group in ("mass", "first_moment", "inertia") or not values or column.group not in values:
            continue
        entry = values[column.group]
        if isinstance(entry, Mapping):
            if column.robot is None:
                vector[position] = float(np.mean([np.asarray(v, dtype=float)[column.joint] for v in entry.values()]))
            elif column.robot in entry:
                vector[position] = float(np.asarray(entry[column.robot], dtype=float)[column.joint])
        else:
            vector[position] = float(np.asarray(entry, dtype=float)[column.joint])
    return vector
