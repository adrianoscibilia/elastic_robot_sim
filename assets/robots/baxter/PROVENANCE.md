# Baxter robot description

- Upstream: <https://github.com/RethinkRobotics/baxter_common>
- Source revision acquired 2026-07-30:
  `6c4b0f375fe4e356a3b12df26ef7c0d5e58df86e`
- Licence: BSD 3-Clause; the upstream `LICENSE` is retained in `source/`.
- Retained ROS description package: `description/baxter_description/`
- URDF: `description/baxter_description/urdf/baxter.urdf`
- Meshes: `description/baxter_description/meshes/`.

Only the description package was retained; it is an archive of the tracked
upstream content at the revision above.  The temporary full clone used for
acquisition is ignored and is not an asset dependency.

The URDF contains both 7-DoF arms plus a head pan joint.  An arm integration
must select one configured chain, rather than treat every non-fixed URDF joint
as an actuated calibration degree of freedom.
