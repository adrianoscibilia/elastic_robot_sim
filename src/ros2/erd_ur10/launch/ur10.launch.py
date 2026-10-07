"""UR10 CB3 L1/L2/L3 bringup (RR_01 S3.4, T1.4): wraps
``ur_robot_driver/ur_control.launch.py`` with erd's own controllers file.

The RTDE sidecar logger is **not** started here (RR_04 A-12): it now writes
straight into the run folder (``<run>/rtde.parquet``) and is started/stopped
by the orchestrator around each run's bag, exactly like ``ros2 bag record``,
so its lifetime matches one recording, not the whole launch session.
"""

from __future__ import annotations

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression


def _controllers_file_for(lab_config_path: str, run_dir: str) -> str:
    """The static ``controllers.yaml``, or -- when a lab config is given -- a
    generated copy with ``scaled_joint_trajectory_controller``'s per-joint
    path/goal tolerances filled in from ``limits.abort.tracking_rad``
    (RR_04 A-5). ``ur_control.launch.py`` only accepts one ``controllers_file``,
    so the override is merged into a full copy rather than layered as a
    second parameter file. Upstream's own ``$(var tf_prefix)`` join convention
    is kept in the generated per-joint keys so the launch system's own
    substitution (``tf_prefix`` defaults to ``''``) still resolves them --
    this file is loaded through the exact same ``ParameterFile`` mechanism as
    the static one, just with different content.
    """
    base_path = Path(get_package_share_directory("erd_ur10")) / "config" / "controllers.yaml"
    if not lab_config_path:
        return str(base_path)

    import yaml

    from erd_recording.config import load_lab_config
    from erd_recording.tolerances import GOAL_TIME_S, GOAL_TOLERANCE_RAD

    config = load_lab_config(lab_config_path)
    tracking_rad = config.limits.abort.tracking_rad
    data = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    controller_params = data["scaled_joint_trajectory_controller"]["ros__parameters"]
    constraints = dict(controller_params.get("constraints", {}))
    constraints["goal_time"] = GOAL_TIME_S
    for driver_joint_name in config.description.joint_map.keys():
        constraints[f"$(var tf_prefix){driver_joint_name}"] = {
            "trajectory": tracking_rad, "goal": GOAL_TOLERANCE_RAD,
        }
    controller_params["constraints"] = constraints

    # RR_06 P3-1 item 2, live finding: the static controllers.yaml quotes
    # every bare `"$(var tf_prefix)"` value (so it survives as an explicit
    # empty string once launch substitutes `tf_prefix:=''`); a plain
    # `yaml.safe_load`/`yaml.safe_dump` round-trip through this function
    # drops those quotes (PyYAML doesn't know the string needs to survive a
    # later, non-YAML substitution pass), and `rcl_yaml_param_parser` then
    # refuses the file outright ("No value at line N") once launch's
    # `ParameterFile(allow_substs=True)` resolves `tf_prefix` to `''` --
    # found live, launching `hardware:=mock lab_config:=...` for the first
    # time this file's own override path was actually exercised. Re-quote
    # every line whose value is *exactly* `$(var tf_prefix)` (never the
    # compound uses like `$(var tf_prefix)tcp_fts_sensor`, which don't
    # collapse to a blank value and never needed quotes).
    import re

    dumped = yaml.safe_dump(data, sort_keys=False)
    dumped = re.sub(r'^(\s*\S+:\s*)\$\(var tf_prefix\)$', r'\1"$(var tf_prefix)"', dumped, flags=re.MULTILINE)

    # RR_06 P3-2c: written into the run folder when the orchestrator already
    # knows it, otherwise a stable per-user location -- never the shared,
    # world-writable `/tmp` the prior pass used.
    override_dir = Path(run_dir) if run_dir else Path.home() / ".ros" / "erd"
    output_path = override_dir / f"erd_ur10_controllers_{Path(lab_config_path).stem}.yaml"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(dumped, encoding="utf-8")
    print(f"[ur10.launch.py] JTC tolerance override written to {output_path}")
    return str(output_path)


def _launch_setup(context, *args, **kwargs):
    hardware = LaunchConfiguration("hardware")
    hardware_value = hardware.perform(context)
    robot_ip = LaunchConfiguration("robot_ip")
    reverse_ip = LaunchConfiguration("reverse_ip")
    lab_config_path = LaunchConfiguration("lab_config").perform(context)
    run_dir = LaunchConfiguration("run_dir").perform(context)

    # RR_06 P3-2c: outside mock, the JTC's path/goal tolerances are a safety
    # parameter, not an optional extra.
    if hardware_value != "mock" and not lab_config_path:
        raise RuntimeError(
            f"ur10.launch.py hardware:={hardware_value!r} refused: lab_config:= is required outside "
            "hardware:=mock (RR_06 P3-2c) -- pass the same erd.lab/1 YAML the recording run uses."
        )

    controllers_file = _controllers_file_for(lab_config_path, run_dir)
    ur_control_launch = str(Path(get_package_share_directory("ur_robot_driver")) / "launch" / "ur_control.launch.py")

    driver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(ur_control_launch),
        launch_arguments={
            "ur_type": "ur10",
            "robot_ip": robot_ip,
            "reverse_ip": reverse_ip,
            "use_mock_hardware": PythonExpression(["'", hardware, "' == 'mock'"]),
            "headless_mode": LaunchConfiguration("headless_mode"),
            "controllers_file": controllers_file,
            "initial_joint_controller": "scaled_joint_trajectory_controller",
            "launch_rviz": "false",
        }.items(),
    )

    # RR_12 A-3a: on hardware: real, preflight refuses any active controller
    # outside the profile's expected set. ur_control.launch.py activates
    # three more in the same spawner call as the ones erd needs (removing
    # them from the controllers file kills that call, RR_11 S3.6), so they
    # are deactivated here once the spawner has activated all of them.
    # force_torque_sensor_broadcaster stays: joint_state_broadcaster is
    # chained to it and cannot run without it.
    extras = " ".join(EXTRA_ACTIVE_CONTROLLERS)
    deactivate_extras = ExecuteProcess(
        cmd=["bash", "-c",
             "is_active() { echo \"$1\" | grep -qE \"^$2 +.* active\"; }; "
             "for i in $(seq 1 120); do "
             "  listed=$(ros2 control list_controllers 2>/dev/null); ready=1; "
             f"  for c in scaled_joint_trajectory_controller {extras}; do is_active \"$listed\" $c || ready=0; done; "
             "  [ $ready = 1 ] && break; sleep 1; "
             "done; "
             "for attempt in 1 2 3 4 5; do "
             f"  ros2 control switch_controllers --deactivate {extras} --strict "
             f"  && echo 'erd_ur10: deactivated {extras} (RR_12 A-3a)' && exit 0; sleep 2; "
             "done; echo 'erd_ur10: could not deactivate the extra controllers; preflight will refuse' >&2; exit 1"],
        output="screen",
    )
    return [driver, deactivate_extras]


#: Controllers ur_robot_driver 3.9 activates that the recording must not run
#: with (RR_12 A-3a); none of them is needed by erd.
EXTRA_ACTIVE_CONTROLLERS = ("gravity_update_controller", "friction_model_controller", "tcp_pose_broadcaster")


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument("hardware", default_value="mock", description="mock | ursim | real"),
        DeclareLaunchArgument("robot_ip", default_value="192.168.56.10"),
        DeclareLaunchArgument("reverse_ip", default_value="192.168.56.1"),
        DeclareLaunchArgument("headless_mode", default_value="false"),
        DeclareLaunchArgument("lab_config", default_value="",
                              description="path to the erd.lab/1 YAML this run uses, for the JTC tolerance "
                                          "override (RR_04 A-5); required outside hardware:=mock (RR_06 P3-2c)"),
        DeclareLaunchArgument("run_dir", default_value="",
                              description="run folder to write the tolerance override into, when the orchestrator "
                                          "already knows it; empty falls back to ~/.ros/erd/ (RR_06 P3-2c)"),
        OpaqueFunction(function=_launch_setup),
    ])
