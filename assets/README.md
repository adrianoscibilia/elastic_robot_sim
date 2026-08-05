# Robot assets

This directory contains robot descriptions and small, redistributable benchmark
artifacts needed to reproduce the simulator integrations.  Large raw datasets
are intentionally excluded from version control; each asset-specific
`PROVENANCE.md` records its upstream location, licence, checksum and retrieval
command.

Robot-specific configuration belongs next to the corresponding asset, while
shared loaders/builders must not depend on a particular robot directory.  The
bundled models are discoverable with `AssetRegistry.for_repository()`:

- `fmrr_tecnobody` — original elastic Cartesian platform;
- `ur10` — UR10 description and meshes from Universal Robots' BSD-3-Clause
  ROS 2 description package;
- `baxter_left` / `baxter_right` — Rethink Baxter arm chains and meshes;
- `kuka_kr300_r2500_ultra_se` — six-axis, dataset-compatible primitive-link
  model.  Its KUKA limits are documented, but its link geometry/inertia remain
  an approximation pending an authorised manufacturer CAD/URDF release.

KUKA raw MAT recordings and a Baxter CSV sample are intentionally local under
`assets/datasets/**/raw/`.  They are ignored by Git because of size/licensing,
but scripts can load them in place; their provenance files document retrieval
and the exact schema.
