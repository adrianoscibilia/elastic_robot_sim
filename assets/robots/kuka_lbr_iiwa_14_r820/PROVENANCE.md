# KUKA LBR iiwa 14 R820 asset provenance

- Source: `https://github.com/lbr-stack/lbr_iiwa14_r820_description`, commit `86ac0532841a90694afe6a65c300f08b55eb1296`.
- License: Apache License 2.0; a copy is retained as `LICENSE`.
- Imported content: joint limits, inertial properties, and all visual/collision meshes.
- Materialization: upstream xacro expanded with robot name `iiwa`; package mesh URIs rewritten to local relative paths.
- Modifications: xacro and ROS package lookup are removed from the runtime asset.
- Effort limits (2026-09-28, R6_02 §6): upstream lists a uniform 200 N·m on every joint; replaced by KUKA's per-axis maxima 320 / 320 / 176 / 176 / 110 / 40 / 40 N·m [S-13]. Velocity limits already match S-13's 85 / 85 / 100 / 75 / 130 / 135 / 135 °/s and are unchanged. The table scene (`kuka_lbr_iiwa_14_r820_table`) carries the same edit.
