# KUKA LBR iiwa 7 R800 asset provenance

- Source: `https://github.com/lbr-stack/lbr_iiwa7_r800_description`, commit `64a0cc38708988c631873b74070f3ee418327c68`.
- License: Apache License 2.0; a copy is retained as `LICENSE`.
- Imported content: joint limits, inertial properties, and all visual/collision meshes.
- Materialization: upstream xacro expanded with robot name `iiwa`; package mesh URIs rewritten to local relative paths.
- Modifications: xacro and ROS package lookup are removed from the runtime asset.
