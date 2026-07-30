# KUKA KR300 R2500 ultra SE model availability

The benchmark dataset identifies the robot as a KUKA KR300 R2500 ultra SE.
KUKA's public download centre directs CAD/model retrieval through authenticated
KUKA Xpert / my.KUKA access.  On 2026-07-30 no authoritative, redistributable
URDF/USD/MJCF for this exact robot was available from the benchmark or KUKA's
public download pages.

Do **not** substitute a different KUKA model (for example an iiwa, KR120, or
KR240) for calibration against this dataset.  Its link geometry, inertia and
transmission characteristics would make the result physically misleading.

Required next asset: an authorised KR300 R2500 ultra SE CAD/robot-description
export, then a checked-in or externally fetched URDF conversion with the six
joint order explicitly verified against the dataset axes.  The official public
technical data sheet is useful for limits and reach only; it is not a dynamic
robot model.
