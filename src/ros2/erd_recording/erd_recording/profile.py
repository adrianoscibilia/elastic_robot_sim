"""Per-robot profile: interface names, controller, rate (RR_01 S5).

``record_iiwa``/``record_ur10`` are thin wrappers over :mod:`erd_recording.pipeline`
with one of these; everything stage-generic lives in ``pipeline``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RobotProfile:
    robot: str                              # config.LabConfig.robot value
    controller_name: str                    # the JTC controller to send goals to
    controller_action: str                  # action name under the controller
    joint_state_topic: str                  # e.g. /dynamic_joint_states
    driver_joint_order: tuple[str, ...]     # the ros2_control / driver joint order
    rate_hz: float
    hardware_choices: tuple[str, ...]       # valid `hardware:` values
    command_interfaces: tuple[str, ...]     # e.g. ("position",)
    state_interfaces: tuple[str, ...]
    #: RR_04 A-1/A-11: iiwa exposes FRI session health via the "fri" GPIO on
    #: `/dynamic_joint_states`; the UR10 has no such interface.
    has_fri_gpio: bool = False
    #: RR_04 A-1/A-11: the UR10's speed-scaling/robot-mode/safety-mode topics
    #: (io_and_status_controller, speed_scaling_state_broadcaster); the iiwa
    #: has none of these.
    has_ur_status_topics: bool = False
    #: RR_04 A-1/B-3: the torque-fraction abort check only applies where a
    #: raw torque measurement exists (iiwa `effort`); `None` on the UR10,
    #: whose `effort` is a current in amps, not yet converted to N*m online.
    torque_abort_state_interface: str | None = None
    #: RR_04 A-4: whether the controller is spawned active (mock: no session
    #: to wait for) or inactive, activated by `preflight` once ready.
    controller_spawns_active: bool = True
    #: RR_06 P3-3c: node names to never flag as a "competing" action client
    #: (RR_04 B-3's "no other follow_joint_trajectory action client" check),
    #: because they are a standard, always-present part of this robot's own
    #: driver launch, not an independent commander -- e.g. ur_robot_driver's
    #: own `trajectory_until_node`, which internally bridges
    #: `ur_msgs/FollowJointTrajectoryUntil` onto our exact
    #: `control_msgs/FollowJointTrajectory` action and is therefore always a
    #: client on it, with or without any operator interference.
    competing_action_client_allowlist: tuple[str, ...] = ()


IIWA_PROFILE = RobotProfile(
    robot="kuka_lbr_iiwa_14_r820",
    controller_name="erd_arm_controller",
    controller_action="follow_joint_trajectory",
    joint_state_topic="/dynamic_joint_states",
    driver_joint_order=("joint_a1", "joint_a2", "joint_a3", "joint_a4", "joint_a5", "joint_a6", "joint_a7"),
    rate_hz=1000.0,
    hardware_choices=("mock", "emulator", "real"),
    command_interfaces=("position",),
    state_interfaces=("position", "velocity", "effort", "commanded_effort"),
    has_fri_gpio=True,
    torque_abort_state_interface="effort",
    controller_spawns_active=False,
)

UR10_PROFILE = RobotProfile(
    robot="ur10_cb3",
    controller_name="scaled_joint_trajectory_controller",
    controller_action="follow_joint_trajectory",
    joint_state_topic="/dynamic_joint_states",
    driver_joint_order=("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"),
    rate_hz=125.0,
    hardware_choices=("mock", "ursim", "real"),
    command_interfaces=("position",),
    state_interfaces=("position", "velocity"),
    has_ur_status_topics=True,
    controller_spawns_active=True,
    competing_action_client_allowlist=("trajectory_until_node",),
)


def profile_for(robot: str) -> RobotProfile:
    if robot == IIWA_PROFILE.robot:
        return IIWA_PROFILE
    if robot == UR10_PROFILE.robot:
        return UR10_PROFILE
    raise ValueError(f"unknown robot {robot!r}")
