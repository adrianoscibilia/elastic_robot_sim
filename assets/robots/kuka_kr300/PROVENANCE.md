# KUKA KR300 R2500 ultra SE model availability

The benchmark dataset identifies the robot as a KUKA KR300 R2500 ultra SE.
KUKA's public download centre directs CAD/model retrieval through authenticated
KUKA Xpert / my.KUKA access.  On 2026-08-05 no authoritative, redistributable
URDF/USD/MJCF for this exact robot was available from the benchmark or KUKA's
public download pages.

Do **not** substitute a different KUKA model (for example an iiwa, KR120, or
KR240) for calibration against this dataset.  Its link geometry, inertia and
transmission characteristics would make the result physically misleading.

This repository now includes `description/kuka_kr300_r2500_ultra_se.urdf` so
the benchmark has a local, loadable six-axis model.  It uses primitive link
geometry and approximate inertias; its six joint limits follow KUKA technical
data sheet 0000182713 and its `a1`…`a6` order matches the data files.  It is a
structural simulation/calibration starting point, not an authoritative dynamic
model.  Replace it with an authorised KR300 R2500 ultra SE CAD/robot-description
export before interpreting identified rigid-body values as physical truth.
