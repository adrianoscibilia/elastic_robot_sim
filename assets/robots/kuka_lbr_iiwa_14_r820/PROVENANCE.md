# KUKA LBR iiwa 14 R820 asset provenance

- Source: `https://github.com/lbr-stack/lbr_iiwa14_r820_description`, commit `86ac0532841a90694afe6a65c300f08b55eb1296`.
- License: Apache License 2.0; a copy is retained as `LICENSE`.
- Imported content: joint limits, inertial properties, and all visual/collision meshes.
- Materialization: upstream xacro expanded with robot name `iiwa`; package mesh URIs rewritten to local relative paths.
- Modifications: xacro and ROS package lookup are removed from the runtime asset.
