# Copyright 2026 elastic_robot_sim contributors
# SPDX-License-Identifier: Apache-2.0
"""iiwa L1/L2/L3 bringup (RR_01 S3.3, T1.3).

``hardware:=mock`` uses ``mock_components/GenericSystem`` (identical
interface names to emulator/real, RR_01 S3.3); ``emulator``/``real`` use
``erd_iiwa/FriPositionSystem``. All three share one controllers YAML and one
JSB, so the same recording code runs unchanged across levels.
"""

from __future__ import annotations

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _launch_setup(context, *args, **kwargs):
    hardware = LaunchConfiguration("hardware").perform(context)
    robot_ip = LaunchConfiguration("robot_ip").perform(context)
    fri_port = LaunchConfiguration("fri_port").perform(context)
    lab_config_path = LaunchConfiguration("lab_config").perform(context)
    run_dir = LaunchConfiguration("run_dir").perform(context)

    # RR_06 P3-2c: outside mock, the JTC's path/goal tolerances are a safety
    # parameter, not an optional extra -- launching without them would abort
    # on no real tracking deviation at all (controllers.yaml carries no
    # per-joint `constraints.<joint>` entries of its own).
    if hardware != "mock" and not lab_config_path:
        raise RuntimeError(
            f"iiwa.launch.py hardware:={hardware!r} refused: lab_config:= is required outside "
            "hardware:=mock (RR_06 P3-2c) -- pass the same erd.lab/1 YAML the recording run uses."
        )

    from erd_iiwa.description import descriptions

    initial_positions = None
    if hardware == "mock" and lab_config_path:
        from erd_recording.config import load_lab_config
        mock_config = load_lab_config(lab_config_path)
        initial_positions = {driver: mock_config.poses.home[mock_config.joint_order.index(sim)]
                             for driver, sim in mock_config.description.joint_map.items()}
    urdf_xml, srdf_xml = descriptions(hardware=hardware, robot_ip=robot_ip, fri_port=fri_port,
                                    initial_positions=initial_positions)

    from ament_index_python.packages import get_package_share_directory
    from pathlib import Path

    controllers_yaml = str(Path(get_package_share_directory("erd_iiwa")) / "config" / "controllers.yaml")

    controller_manager_parameters = [{"robot_description": urdf_xml}, controllers_yaml]
    if hardware != "mock":
        controller_manager_parameters.append({
            "hardware_synchronization.expect_blocking_read_write": True,
            "hardware_synchronization.minimum_cycle_time": 0.00001,
        })
    if lab_config_path:
        # RR_04 A-5: per-joint path/goal tolerances from limits.abort.tracking_rad,
        # generated fresh from whichever lab config this run uses -- never
        # hand-edited into the checked-in controllers.yaml.
        from erd_recording.config import load_lab_config
        from erd_recording.tolerances import write_tolerance_overrides

        config = load_lab_config(lab_config_path)
        driver_joint_names = tuple(config.description.joint_map.keys())
        # RR_06 P3-2c: written into the run folder when the orchestrator
        # (whoever launches this already knowing the run directory, e.g. a
        # `--run-id` that matches `run_dir:=`) gives one; otherwise a stable
        # per-user location, never the shared, world-writable `/tmp` the prior
        # pass used.
        if run_dir:
            override_dir = Path(run_dir)
        else:
            override_dir = Path.home() / ".ros" / "erd"
        override_path = override_dir / f"erd_iiwa_tolerances_{Path(lab_config_path).stem}.yaml"
        write_tolerance_overrides(config, "erd_arm_controller", driver_joint_names, override_path)
        print(f"[iiwa.launch.py] JTC tolerance override written to {override_path}")
        controller_manager_parameters.append(str(override_path))

    robot_state_publisher = Node(
        package="robot_state_publisher", executable="robot_state_publisher", output="screen",
        parameters=[{"robot_description": urdf_xml}],
    )
    controller_manager = Node(
        package="controller_manager", executable="ros2_control_node", output="screen",
        parameters=controller_manager_parameters,
        remappings=[("/joint_state_broadcaster/joint_states", "/joint_states")],
    )
    joint_state_broadcaster_spawner = Node(
        package="controller_manager", executable="spawner", arguments=["joint_state_broadcaster"],
    )
    # RR_04 A-4: outside mock, the JTC is spawned inactive and only activated
    # by `preflight` once the FRI session is COMMANDING_ACTIVE with finite
    # states -- spawning it active would let it latch NaN or a stale IPO hold
    # position before the first real packet arrives. Mock keeps
    # auto-activation (there is no session to wait for).
    arm_controller_args = ["erd_arm_controller"] if hardware == "mock" else ["erd_arm_controller", "--inactive"]
    arm_controller_spawner = Node(
        package="controller_manager", executable="spawner", arguments=arm_controller_args,
    )
    recording_spawner = Node(
        package="controller_manager", executable="spawner", arguments=["erd_state_broadcaster"],
    )
    return [robot_state_publisher, controller_manager, joint_state_broadcaster_spawner,
            recording_spawner, arm_controller_spawner]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument("hardware", default_value="mock", description="mock | emulator | real"),
        DeclareLaunchArgument("robot_ip", default_value="192.170.10.2"),
        DeclareLaunchArgument("fri_port", default_value="30200"),
        DeclareLaunchArgument("lab_config", default_value="",
                              description="path to the erd.lab/1 YAML this run uses, for the JTC tolerance "
                                          "override (RR_04 A-5); required outside hardware:=mock (RR_06 P3-2c)"),
        DeclareLaunchArgument("run_dir", default_value="",
                              description="run folder to write the tolerance override into, when the orchestrator "
                                          "already knows it; empty falls back to ~/.ros/erd/ (RR_06 P3-2c)"),
        OpaqueFunction(function=_launch_setup),
    ])
