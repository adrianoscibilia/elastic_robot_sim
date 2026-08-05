# UR10 description

- Upstream: <https://github.com/UniversalRobots/Universal_Robots_ROS2_Description>
- Acquisition: 2026-08-05, shallow checkout of the upstream default branch.
- Licence: BSD-3-Clause; the upstream licence is retained at
  `description/ur_description/LICENSE`.

`description/ur10.urdf` is the repository's pre-existing flattened UR10
description, with its stale machine-specific mesh references replaced by
portable paths to the locally retained upstream UR10 meshes.  The robot's
kinematic calibration remains nominal; use a controller-derived calibration
file for robot-specific real-world accuracy.
