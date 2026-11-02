"""The nominal URDF's joint-space inertia as a table a consumer can evaluate without pinocchio.

``identified_filter/04`` B2 and ``05`` D4: the identified filter's inertia is ``M_f(theta) = M_URDF(theta)``
scaled per joint, so the consumer (``dynamic_model_nn/urdf_inertia.py``) needs ``M_URDF`` of the bare asset at
every sample, on the GPU, inside the training graph.  :func:`chain_table` writes the serial chain pinocchio
builds from the asset's URDF (fixed links already merged into the moving bodies):

* per joint, the placement in its parent joint's frame at ``q = 0`` (``rotation`` 3x3, ``translation`` 3) and
  the revolute ``axis`` (3, in its own frame);
* per body, ``mass``, centre of mass ``com`` (3, body frame) and the rotational inertia about it ``inertia`` (3x3);
* the rotor ``armature`` pinocchio adds to the diagonal (zero for the r8 assets).

``check`` holds a few ``(q, M)`` pairs from ``pinocchio.crba`` so the consumer can verify its own evaluation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

#: The pinocchio joint models a table can describe: revolute about a fixed axis of the joint frame.
_AXES = {"JointModelRX": (1.0, 0.0, 0.0), "JointModelRY": (0.0, 1.0, 0.0), "JointModelRZ": (0.0, 0.0, 1.0)}


def chain_table(asset: Any, check_samples: int = 8, seed: int = 0) -> dict:
    """The asset's serial chain as plain lists, plus ``check_samples`` pinocchio ``(q, M)`` pairs."""
    from .identification import build_model

    pin, model, data = build_model(asset)
    n = model.nv
    if model.nq != n or list(model.parents) != [0] + list(range(n)):
        raise ValueError(f"{asset.name}: chain_table needs a serial chain of 1-DoF joints, got parents "
                         f"{list(model.parents)} (nq {model.nq}, nv {n})")
    joints = []
    for i in range(1, model.njoints):
        joint = model.joints[i]
        kind = joint.shortname()
        if kind in _AXES:
            axis = _AXES[kind]
        elif kind == "JointModelRevoluteUnaligned":
            axis = tuple(float(v) for v in joint.extract().axis)
        else:
            raise ValueError(f"{asset.name}: joint {model.names[i]} is {kind}, not revolute")
        placement, body = model.jointPlacements[i], model.inertias[i]
        joints.append({"name": model.names[i], "rotation": np.asarray(placement.rotation).tolist(),
                       "translation": np.asarray(placement.translation).tolist(), "axis": list(axis),
                       "mass": float(body.mass), "com": np.asarray(body.lever).tolist(),
                       "inertia": np.asarray(body.inertia).tolist()})
    rng = np.random.default_rng(seed)
    low = np.where(np.isfinite(model.lowerPositionLimit), model.lowerPositionLimit, -np.pi)
    high = np.where(np.isfinite(model.upperPositionLimit), model.upperPositionLimit, np.pi)
    check = []
    for _ in range(check_samples):
        q = rng.uniform(low, high)
        matrix = np.asarray(pin.crba(model, data, q))
        check.append({"q": q.tolist(), "M": (np.triu(matrix) + np.triu(matrix, 1).T).tolist()})
    return {"asset": asset.name, "joints": joints, "armature": np.asarray(model.armature, dtype=float).tolist(),
            "check": check}


def crba_reference(asset: Any, q: np.ndarray) -> np.ndarray:
    """``M_URDF(q)`` ``[N, n, n]`` from pinocchio, the reference the table is tested against."""
    from .identification import build_model

    pin, model, data = build_model(asset)
    out = np.empty((len(q), model.nv, model.nv))
    for i, row in enumerate(np.asarray(q, dtype=float)):
        matrix = np.asarray(pin.crba(model, data, row))
        out[i] = np.triu(matrix) + np.triu(matrix, 1).T
    return out
