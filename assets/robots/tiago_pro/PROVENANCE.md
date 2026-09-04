# TIAGo Pro dual-arm asset provenance

- Robot source: `https://github.com/pal-robotics/tiago_pro_robot`, commit `59eff25559b0f071ac5cf6d3c4c72c2e33b8951a` (`humble-devel`).
- Arm source: `https://github.com/pal-robotics/pal_sea_arm`, commit `78b544fd2d96b8edbc2472e68f653ea5cce15423` (`humble-devel`).
- License: Apache License 2.0; the upstream license is retained as `LICENSE`.
- Imported content: TIAGo Pro SEA arm inertial properties and the `arm_tiago_pro` visual/collision meshes.
- Materialization: `tiago-pro`, spherical wrist, v2 limits, zero calibration offsets; left and right seven-joint arms are mounted to a fixed simplified torso. Base, torso, head and grippers are intentionally outside the active calibration model.
- Modifications: xacro was expanded to standalone URDF, package mesh URIs were rewritten to local relative paths, and the fixed torso proxy was added. No external ROS package is needed to load the asset.
