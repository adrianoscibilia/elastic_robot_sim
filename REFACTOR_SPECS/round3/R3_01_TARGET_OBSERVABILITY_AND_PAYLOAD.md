# R3_01 — Target observability and payload randomization

Addresses findings **F1** (target invariant to the randomized parameters) and
**F2** (no payload; three of seven target channels carry no signal).

---

## 1. The problem, stated precisely

### 1.1 What `ft` actually is

`run_mujoco_elastic_torque` records

```python
tau_spring = -(transmission.stiffness * elastic_q + transmission.damping * elastic_dq)
rows["tau_link"].append(tau_spring)
```

and `rollout_frame` writes that as `ft0..ft{n-1}`.

Contacts are disabled, friction is injected on the **motor** DOF only
(`solver_torque = command.total - friction.torque(motor_dq)` applied at
`addresses[name]["motor_dof"]`), and the rotor inertia is armature on the motor
DOF. The link subtree is therefore driven by the spring torque and nothing else,
so the link-side equation of motion is exactly

```
M(q_link) ddq_link + C(q_link, dq_link) dq_link + g(q_link) = tau_spring
```

i.e.

```
ft  ==  rnea(q_link, dq_link, ddq_link)      with the URDF's nominal inertias
```

This is the same identity that makes the rigid tier's `tau == ft` and that the
existing acceptance test exploits when it recovers base parameters to 1e-4.

### 1.2 Why that makes the elastic randomization inert for part of the model zoo

The inertial parameters on the right-hand side are the **URDF's**, and they are
identical for every sampled robot — only `stiffness`, `damping` and (nominally)
`rotor_inertia` differ. Therefore:

- For a model whose inputs are **link-side only** — `(q, dq, ddq)`, which is the
  `dynamic_model_nn` `*_res` family — the function being learned is *literally the
  same function* on every one of the six robots. Sampling stiffness produces a
  **covariate shift** (which states get visited) and nothing more. It is data
  augmentation, not a family of distinct systems.
- For a model that also consumes the motor torque — `lnn_tau_elastic.forward(q, dq,
  ddq, tau_cmd)`, whose `elastic_state_observer` takes `cat([q, dq, tau_cmd])` —
  the stiffness *is* identifiable, because `tau_motor - ft` depends on the rotor
  acceleration and hence on `k`. Here the randomization is meaningful, subject to
  `R3_02` (the excitation must actually make that dependence visible) and `R3_03`
  (six draws is not enough coverage).

**This is not a bug to fix; it is a claim to state.** The deliverable is that the
manifest and the docs say which model families the dataset can support, and that
the dataset gains an axis that *does* change the link-side map. That axis is
payload.

### 1.3 Why payload is the right axis

`docs/IDENTIFICATION_DATASET.md` already measures the consequence of having none:
link-side torque RMS per joint is about

```
[1.3, 31, 1.7, 9.7, 0.2, 0.2, 0.0] Nm
```

With no tool fitted, the body past the A7 transmission is the bare flange
(`M_77 = 3e-4 kg m²`), so `ft6 == 0` to numerical precision. `ft4` and `ft5` are
at the level of integration noise. The consumer normalizes per channel
(`CustomDataset._normalize`, optionally per bag via `stats_by_bag`), so those three
channels are divided by their own standard deviation and **noise is amplified to
unit variance**. A model trained on this is fitting numerical artifacts on 3/7 of
its outputs.

A randomized payload at the flange:

- changes `M(q)`, `C(q, dq)` and `g(q)` — so the link-side map genuinely differs
  robot to robot, which is what F1 says is missing;
- is the classical excitation for distal inertial parameters in the identification
  literature, so it is easy to defend;
- makes `ft4..ft6` carry real signal, removing the normalization pathology;
- is cheap: no change to the integration step, no change to the excitation search.

---

## 2. Design

### 2.1 Configuration

Add a `payload:` block to `config/identification/kuka_lbr_iiwa_14_r820_table.yaml`,
peer of `transmission:`:

```yaml
# Randomized tool/payload rigidly attached to the flange.  Without one the last
# three link-side torque channels are numerically zero (M_77 = 3e-4 kg m^2) and
# per-channel normalization in the consumer amplifies their noise to unit
# variance.  A payload also makes the link-side dynamics differ between sampled
# robots, which transmission stiffness alone does not (see R3_01 section 1.2).
payload:
  enabled: true
  mass: [0.0, 6.0]            # kg, sampled uniformly; 0 keeps the bare-flange case
  # Offset of the payload centre of mass from the flange frame, in the flange
  # frame [m].  Sampled uniformly and independently per axis.
  offset_x: [-0.08, 0.08]
  offset_y: [-0.08, 0.08]
  offset_z: [0.02, 0.20]
  # The payload is modelled as a uniform box whose principal inertia follows from
  # mass and size; size is sampled uniformly and is isotropic unless you extend
  # this to three ranges.
  size: [0.05, 0.25]          # m, cube edge
  # Draw one payload per robot (the physical reading: a robot ships with a tool),
  # or one per bag (the reading: the same robot is used with many tools).
  per: robot                  # "robot" | "trajectory"
```

Rationale for the numbers: the LBR iiwa 14 R820 is rated 14 kg; a range of
`0-6 kg` stays well inside the effort limits (`effort="200"` on every joint in the
URDF) under `max_acceleration 4.0 rad/s²`, and includes 0 so the bare-flange case
the current dataset produces remains inside the distribution rather than outside it.
**Verify the effort headroom** with the check in `R3_08 §C`, and lower the range if
any bag exceeds 80 % of `effort` on any joint.

### 2.2 The consistency requirement that will break a naive implementation

`body_overrides` already exists (`generic_mujoco_runner._apply_body_overrides`,
threaded into the torque runners through `config["body_overrides"]`). **Do not use
it for the payload.** `identification.build_model` builds the Pinocchio model
straight from `asset.urdf_path`:

```python
model = pin.buildModelFromUrdf(str(Path(asset.urdf_path).resolve()))
```

so a `body_overrides` payload would be seen by the simulator but **not** by
`ComputedTorqueController` / `SeaMotorController`, which are Pinocchio RNEA. The
computed-torque cancellation would stop cancelling, the feedback term would grow
past the 5 % of feedforward that the test suite locks in, and the trajectory
feasibility (`max_acceleration`, effort limits) would be checked against the wrong
model.

**Implement the payload by injecting it into the URDF**, so every consumer of the
asset — Pinocchio, MuJoCo, Newton, the collision checker — sees the same robot by
construction. This is the only approach that is consistent without a per-consumer
audit.

### 2.3 New module: `src/elastic_sim/payload.py`

```python
"""Rigid tool attached to the flange, injected into the URDF.

A payload must be visible to *every* consumer of an asset at once: the
Pinocchio model behind the computed-torque controller, both simulator
importers, and the collision checker.  ``generic_mujoco_runner``'s
``body_overrides`` reaches only the simulator, so a payload applied that way
would silently de-tune the controller.  Injecting a link into the URDF text and
handing back an ``AssetSpec`` that points at it keeps them in step by
construction.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, replace
from pathlib import Path
import tempfile

import numpy as np

from .assets import AssetSpec


@dataclass(frozen=True)
class Payload:
    """A uniform box rigidly attached to the last link."""

    mass: float = 0.0
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    size: float = 0.1

    @property
    def is_empty(self) -> bool:
        return self.mass <= 0.0

    def principal_inertia(self) -> tuple[float, float, float]:
        """Box inertia about its own centre of mass, principal axes."""
        value = self.mass * (2.0 * self.size ** 2) / 12.0
        return (value, value, value)

    def as_dict(self) -> dict:
        return {"mass": float(self.mass), "offset": [float(v) for v in self.offset],
                "size": float(self.size)}


@contextlib.contextmanager
def payload_asset(asset: AssetSpec, payload: Payload | None):
    """Yield ``asset`` with ``payload`` welded to its last link.

    Yields the asset unchanged when the payload is empty, so the bare-flange
    path costs nothing.
    """
    if payload is None or payload.is_empty:
        yield asset
        return
    text = Path(asset.urdf_path).read_text(encoding="utf-8")
    parent = _last_link_name(asset)
    ixx, iyy, izz = payload.principal_inertia()
    x, y, z = payload.offset
    injected = (
        f'<link name="identification_payload">'
        f'<inertial><origin rpy="0 0 0" xyz="0 0 0"/>'
        f'<mass value="{payload.mass:.9g}"/>'
        f'<inertia ixx="{ixx:.9g}" ixy="0" ixz="0" iyy="{iyy:.9g}" iyz="0" izz="{izz:.9g}"/>'
        f'</inertial></link>'
        f'<joint name="identification_payload_joint" type="fixed">'
        f'<parent link="{parent}"/><child link="identification_payload"/>'
        f'<origin rpy="0 0 0" xyz="{x:.9g} {y:.9g} {z:.9g}"/>'
        f'</joint>'
    )
    text = text.replace("</robot>", injected + "</robot>")
    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory) / Path(asset.urdf_path).name
        target.write_text(text, encoding="utf-8")
        # Meshes are resolved relative to the URDF, so keep the original folder
        # reachable: symlink or copy siblings as the existing
        # ``_materialized_mujoco_urdf`` helper already does.
        _mirror_siblings(Path(asset.urdf_path).parent, target.parent)
        yield replace(asset, urdf_path=target)
```

**Implementation notes for whoever writes this.**

- `_last_link_name(asset)` must return the **child link of the last active
  joint**, not the URDF's last declared link and not the `iiwa_link_ee` /
  `iiwa_joint_ee` fixed frame if one exists. Derive it from
  `discover_urdf_joints(asset.urdf_path)` filtered to `asset.joint_names`,
  taking the `child` of the last one. Add an assertion that the chosen parent
  link exists in the URDF text.
- `_mirror_siblings` should reuse whatever `generic_mujoco_runner
  ._materialized_mujoco_urdf` already does for mesh resolution rather than
  inventing a second mechanism; if that helper already writes a temp URDF with
  resolved meshes, prefer **composing** with it (inject first, then hand the
  injected text to that helper) over duplicating it.
- The URDF for this asset is a single line with `</robot>` at the end; a
  `str.replace` on the **last** occurrence is safe. Use
  `text[:text.rindex("</robot>")] + injected + text[text.rindex("</robot>"):]`
  rather than `replace`, so a `</robot>` inside a comment cannot break it.
- If `compose_scene_urdf.py` is later used to produce assets with more than one
  robot, this helper must take the chain explicitly. Raise on
  `len(asset.joint_names) == 0` and document the single-chain assumption.

### 2.4 Sampling

In `src/elastic_sim/dataset.py`:

```python
@dataclass(frozen=True)
class PayloadSampling:
    enabled: bool = False
    mass: tuple[float, float] = (0.0, 0.0)
    offset_x: tuple[float, float] = (0.0, 0.0)
    offset_y: tuple[float, float] = (0.0, 0.0)
    offset_z: tuple[float, float] = (0.0, 0.0)
    size: tuple[float, float] = (0.1, 0.1)
    per: str = "robot"          # "robot" | "trajectory"

    def __post_init__(self) -> None:
        if self.per not in ("robot", "trajectory"):
            raise ValueError("payload.per must be 'robot' or 'trajectory'")
        for name in ("mass", "size"):
            low, high = getattr(self, name)
            if low < 0.0 or high < low:
                raise ValueError(f"payload.{name} must satisfy 0 <= min <= max")
        if self.enabled and self.size[0] <= 0.0:
            raise ValueError("payload.size min must be positive when enabled")


def sample_payloads(sampling: PayloadSampling, seed: int, count: int) -> tuple[Payload, ...]:
    """Draw ``count`` payloads from their own stream, in index order.

    Uses ``default_rng((seed, 2))`` so payloads are independent of the robot
    stream ``(seed, 1)``: adding payloads does not perturb the sampled
    stiffnesses, and vice versa.
    """
```

- Stream key **must** be `(int(seed), 2)`. `(seed, 1)` is already taken by
  `sample_robots`; `(seed,)` plain is taken by `generate`'s friction stream. Adding
  a third stream this way preserves the repo's existing reproducibility promise —
  "a robot depends only on the dataset seed and its index" — which the docs make
  explicitly and which `run_identification_simulation.py --tier e03` relies on.
- `per: robot` → one payload per tier, indexed by tier position among the elastic
  tiers; the rigid tier always gets `Payload()` (empty) so that the rigid
  reference keeps validating against the unmodified URDF, which is the whole point
  of that tier.
- `per: trajectory` → one payload per `(tier, trajectory_index)` pair, drawn in
  `iter_conditions` order.

### 2.5 Threading it through

`run_condition` is the single choke point. Change its signature to accept
`payload: Payload | None = None` and wrap the whole body:

```python
with payload_asset(asset, payload) as asset_p:
    ...   # existing body, but every `asset` becomes `asset_p`
```

Everything downstream — `ComputedTorqueController`, `SeaMotorController`,
`run_mujoco_elastic_torque`, `link_inertia_envelope` — takes the asset as an
argument and will pick the payload up for free.

**`link_inertia_envelope` must be recomputed per payload.** It is currently hoisted
out of the loop in `generate` (correctly, since it is asset-dependent and the asset
was constant). With payloads it is asset-**and**-payload dependent, and it sets both
the derived damping and the integration step. Cache it keyed on the payload:

```python
envelopes: dict[tuple, LinkInertia] = {}
def _envelope(asset, payload):
    key = () if payload is None or payload.is_empty else (payload.mass, payload.offset, payload.size)
    if key not in envelopes:
        with payload_asset(asset, payload) as asset_p:
            envelopes[key] = link_inertia_envelope(asset_p, n_samples=config.transmission.inertia_samples)
    return envelopes[key]
```

With `per: robot` there are at most `robots + 1` envelopes, so the cost is bounded
and small (512 Pinocchio CRBA evaluations each).

A heavier payload **raises** `J_link` at the wrist, which **lowers** the A7 mode and
therefore **relaxes** the integration step. Expect elastic rollouts to get *faster*,
not slower, once a payload is fitted — but assert it rather than assume it
(`R3_08 §C`).

### 2.6 Trajectory feasibility

Excitation candidates are scored on a payload-free model today
(`optimize_excitation(asset, ...)` in `generate`). Two options:

- **(a) Keep trajectories payload-free** (simplest, recommended first): the
  kinematic limits are unchanged by a payload, and the regressor condition number
  is a property of the state trajectory, not of the inertias. The only thing a
  payload changes is the *torque* needed, which is checked separately against the
  effort limit in `R3_08 §C`. Collision geometry is also unchanged, because the
  injected link has no `<collision>` element.
- **(b) Score trajectories per payload** — `n_payloads x n_trajectories`
  optimizations. Only worth it if you later add a payload-aware conditioning
  metric. Do not do this in this round.

Choose **(a)**. State it in the docstring of `generate` so nobody "fixes" it later.

### 2.7 Output schema

Add per-joint-free payload metadata columns to `rollout_frame`, next to the
existing transmission metadata:

```
payload_mass, payload_offset_x, payload_offset_y, payload_offset_z, payload_size
```

Five scalar columns, constant within a bag, `NaN` on the rigid tier. They are
metadata — the loader ignores unrecognized columns — but they are what makes the
manifest self-describing and what a payload-conditioned model would read.

Record the same values per bag in the manifest `records` entries and, for
`per: robot`, in the `robots` list alongside the transmission description.

---

## 3. Optional, higher-value variant: a motor-side target

If the goal is a model that *identifies transmission parameters* rather than one
that regresses link-side torque, the target must involve motor-side quantities.
The dataset already contains everything needed; only the documentation and a
manifest key are missing.

Two learnable maps are available from the existing columns:

| Map | Inputs | Target | Depends on `k`? |
|---|---|---|---|
| A (today) | `q, dq, ddq` (link) | `ft` | **No** — see §1.2 |
| B | `q, dq, ddq, tau` | `ft` | **Yes**, via `tau - ft` |
| C | `q_motor, dq_motor, tau` | `q_motor - q_link` (deflection) | **Yes**, directly |

Map C is the cleanest identification target for the elastic parameters and is one
derived column away: emit

```
defl0..defl{n-1} = q_motor{i} - q_link{i}
```

It costs 7 columns and no simulation time, and it gives a target whose magnitude is
`tau/k` — i.e. a direct readout of the parameter being randomized. Add it; state in
`docs/IDENTIFICATION_DATASET.md` which of A/B/C each `dynamic_model_nn` model family
corresponds to.

---

## 4. Acceptance

- `payload_asset` with an empty payload yields the **same object** (identity check).
- A 5 kg payload at `(0, 0, 0.1)` changes Pinocchio's `M_77` by the analytically
  predicted amount, and MuJoCo's inverse dynamics on the payload-injected asset
  agrees with Pinocchio's to 1e-6 Nm — the same bar the existing suite holds the
  bare asset to.
- With a payload fitted, the recorded `ft4, ft5, ft6` RMS are all `> 0.5 Nm`
  (versus `0.2, 0.2, 0.0` today).
- The feedback/feedforward ratio stays under 5 %, i.e. the controller still cancels
  — this is the test that catches a payload that reached the simulator but not
  Pinocchio.
- Peak `|tau|` stays under 80 % of the URDF `effort` on every joint.

Tests in `R3_08 §C`.
