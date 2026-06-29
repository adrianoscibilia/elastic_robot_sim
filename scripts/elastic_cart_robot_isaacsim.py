import numpy as np
import pandas as pd
import argparse
import yaml
from yaml import SafeLoader
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from elastic_sim.trajectory import (
    load_trajectory_settings,
    validate_trajectory_settings,
    make_sinusoidal_trajectory,
    make_ptp_trajectory,
    make_hold_trajectory,
)

from warp import tau

# Parse command line arguments
parser = argparse.ArgumentParser(description="Elastic cart model simulation")
parser.add_argument("--csv", action="store_true", help="Save simulation data to CSV")
parser.add_argument("--plot", action="store_true", help="Show debug plots")
parser.add_argument("--headless", action="store_true", help="Run simulation in headless mode (no GUI)")
args = parser.parse_args()

# Store command line arguments as module-level variables
SAVE_CSV = args.csv
SHOW_PLOT = args.plot
HEADLESS = args.headless
CUT_OFF_TIME = 0  # Time to skip at the beginning of the simulation for data collection

from isaaclab.app import AppLauncher
app_launcher = AppLauncher({
    "headless": HEADLESS
    })

simulation_app = app_launcher.app

from isaacsim.core.api import World  #type: ignore
from isaacsim.core.api.objects import DynamicCuboid  #type: ignore
from isaacsim.core.utils.prims import create_prim  #type: ignore
from isaacsim.core.utils.types import ArticulationAction  #type: ignore
from isaacsim.core.api.robots import Robot  # type: ignore
from pxr import UsdPhysics
from pxr import PhysxSchema
import matplotlib.pyplot as plt

# Load settings from YAML
config_path = os.path.join(os.path.dirname(__file__), "..", "config", "settings.yaml")
with open(config_path, 'r') as f:
    settings = yaml.load(f, Loader=SafeLoader)

# Extract elastic drive parameters
SIM_TIME = settings['simulation']['sim_time']
TIME_STEP = settings['simulation']['time_step']
REF_MODE = settings['simulation']['ref_mode']
SEED = settings['simulation']['seed']

DRIVE_X_STIFFNESS = settings['elastic_drives']['drive_x']['stiffness']
DRIVE_X_DAMPING   = settings['elastic_drives']['drive_x']['damping']
DRIVE_Y_STIFFNESS = settings['elastic_drives']['drive_y']['stiffness']
DRIVE_Y_DAMPING   = settings['elastic_drives']['drive_y']['damping']
DRIVE_Z_STIFFNESS = settings['elastic_drives']['drive_z']['stiffness']
DRIVE_Z_DAMPING   = settings['elastic_drives']['drive_z']['damping']
PAYLOAD = settings['elastic_drives']['payload']

def set_random_seed(seed):
    if seed != -1:
        np.random.seed(seed)
    else:
        seed = np.random.randint(0, 10000)
        np.random.seed(seed)
    print(f"Random seed set to: {seed}")
    return seed

#################################################################################
################################ Debug Plots ####################################
#################################################################################

def debug_kinematics(ref_pos, ref_vel, pos, vel, time):
    plt.figure(figsize=(14, 12))
    
    # Row 1: X (motor vs link)
    plt.subplot(3, 2, 1)
    plt.plot(time, ref_pos[:, 0], label="X ref", linestyle="dashed")
    plt.plot(time, pos[:, 0], label="X link")
    plt.title("Position - X")
    plt.ylabel("Position (m)")
    plt.legend()
    plt.grid()
    
    plt.subplot(3, 2, 2)
    plt.plot(time, ref_vel[:, 0], label="VX ref", linestyle="dashed")
    plt.plot(time, vel[:, 0], label="VX link")
    plt.title("Velocity - X")
    plt.ylabel("Velocity (m/s)")
    plt.legend()
    plt.grid()
    
    # Row 2: Y (motor vs link)
    plt.subplot(3, 2, 3)
    plt.plot(time, ref_pos[:, 1], label="Y ref", linestyle="dashed")
    plt.plot(time, pos[:, 1], label="Y link")
    plt.title("Position - Y")
    plt.ylabel("Position (m)")
    plt.legend()
    plt.grid()
    
    plt.subplot(3, 2, 4)
    plt.plot(time, ref_vel[:, 1], label="VY ref", linestyle="dashed")
    plt.plot(time, vel[:, 1], label="VY link")
    plt.title("Velocity - Y")
    plt.ylabel("Velocity (m/s)")
    plt.legend()
    plt.grid()
    
    # Row 3: Z (motor vs link)
    plt.subplot(3, 2, 5)
    plt.plot(time, ref_pos[:, 2], label="Z ref", linestyle="dashed")
    plt.plot(time, pos[:, 2], label="Z link")
    plt.title("Position - Z")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.legend()
    plt.grid()
    
    plt.subplot(3, 2, 6)
    plt.plot(time, ref_vel[:, 2], label="VZ ref", linestyle="dashed")
    plt.plot(time, vel[:, 2], label="VZ link")
    plt.title("Velocity - Z")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.legend()
    plt.grid()
    
    plt.tight_layout()

def debug_dynamics(tau_x, tau_y, tau_z, tau_x_nom, tau_y_nom, tau_z_nom, time):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(time, tau_x, label="Tau X")
    plt.plot(time, tau_x_nom, label="Tau X (Nominal)", linestyle="dashed")
    plt.title("Torque - X")
    plt.ylabel("Torque (N*m)")
    plt.legend()
    plt.grid()

    plt.subplot(1, 3, 2)
    plt.plot(time, tau_y, label="Tau Y")
    plt.plot(time, tau_y_nom, label="Tau Y (Nominal)", linestyle="dashed")
    plt.title("Torque - Y")
    plt.ylabel("Torque (N*m)")
    plt.legend()
    plt.grid()

    plt.subplot(1, 3, 3)
    plt.plot(time, tau_z, label="Tau Z")
    plt.plot(time, tau_z_nom, label="Tau Z (Nominal)", linestyle="dashed")
    plt.title("Torque - Z")
    plt.ylabel("Torque (N*m)")
    plt.legend()
    plt.grid()

    plt.tight_layout()

#################################################################################
########################### Trajectory Generation ###############################
#################################################################################

# TODO (isaacsim): heavy isaaclab/isaacsim imports at the module top prevent running
# this backend in CI. The trajectory math is imported from elastic_sim.trajectory
# at runtime (below); only the isaaclab simulation loop itself remains here.

#################################################################################
############################# Main Simulation Loop ##############################
#################################################################################

def main():
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    physics_context = world.get_physics_context()
    physics_context.enable_gpu_dynamics(True)
    physics_context.set_physics_dt(dt=TIME_STEP, substeps=8)
    physics_context.set_solver_type("TGS")
    physics_context.enable_stablization(True)
    physics_context.set_gravity(-9.81)

    world.set_simulation_dt(
        physics_dt=TIME_STEP,
        rendering_dt=TIME_STEP
    )

    stage = world.stage

    scene_prim = stage.GetPrimAtPath("/physicsScene")
    if not scene_prim:
        raise RuntimeError("Physics scene not found")
    
    physx_scene = PhysxSchema.PhysxSceneAPI.Apply(scene_prim)
    physx_scene.CreateMaxPositionIterationCountAttr(128)
    physx_scene.CreateMaxVelocityIterationCountAttr(32)
    physx_scene.CreateEnableStabilizationAttr(True)

    seed = set_random_seed(SEED)

    # --------------------------------------------------
    # Robot root
    # --------------------------------------------------
    create_prim("/World/Robot", "Xform")
    UsdPhysics.ArticulationRootAPI.Apply(
        world.stage.GetPrimAtPath("/World/Robot")
    )

    # --------------------------------------------------
    # Base (fixed)
    # --------------------------------------------------
    create_prim("/World/Robot/Base", "Xform")
    base_prim = world.stage.GetPrimAtPath("/World/Robot/Base")
    UsdPhysics.RigidBodyAPI.Apply(base_prim)

    # Fix base to world
    create_prim("/World/Robot/BaseFixedJoint", "PhysicsFixedJoint")
    fixed_joint = UsdPhysics.FixedJoint.Get(
        world.stage, "/World/Robot/BaseFixedJoint"
    )
    fixed_joint.CreateBody0Rel().SetTargets(["/World/Robot/Base"])

    # --------------------------------------------------
    # LINK X - MOTOR
    # --------------------------------------------------
    link_x_motor = DynamicCuboid(
        prim_path="/World/Robot/Link_X_motor",
        position=np.array([1.5, 0.0, 1.5]),
        size=0.2,
        mass=2.0,
    )

    create_prim("/World/Robot/Joint_X_motor", "PhysicsPrismaticJoint")
    joint_x_motor = UsdPhysics.PrismaticJoint.Get(
        world.stage, "/World/Robot/Joint_X_motor"
    )
    joint_x_motor.CreateBody0Rel().SetTargets(["/World/Robot/Base"])
    joint_x_motor.CreateBody1Rel().SetTargets(["/World/Robot/Link_X_motor"])
    joint_x_motor.CreateAxisAttr("X")
    joint_x_motor.CreateLowerLimitAttr(-1.25)
    joint_x_motor.CreateUpperLimitAttr(1.25)

    drive_x_motor = UsdPhysics.DriveAPI.Apply(joint_x_motor.GetPrim(), "linear")
    drive_x_motor.CreateStiffnessAttr(30000.0)
    drive_x_motor.CreateDampingAttr(500.0)
    drive_x_motor.CreateMaxForceAttr(2000.0)

    # --------------------------------------------------
    # LINK X - ELASTIC
    # --------------------------------------------------
    link_x = DynamicCuboid(
            prim_path="/World/Robot/Link_X",
            position=np.array([1.5, 0.0, 1.5]),
            size=0.2,
            mass=2.0,
        )

    create_prim("/World/Robot/Joint_X", "PhysicsPrismaticJoint")
    joint_x = UsdPhysics.PrismaticJoint.Get(
        world.stage, "/World/Robot/Joint_X"
    )

    joint_x.CreateBody0Rel().SetTargets(["/World/Robot/Link_X_motor"])
    joint_x.CreateBody1Rel().SetTargets(["/World/Robot/Link_X"])
    joint_x.CreateAxisAttr("X")

    drive_x = UsdPhysics.DriveAPI.Apply(
        joint_x.GetPrim(), "linear"
    )
    drive_x.CreateStiffnessAttr(DRIVE_X_STIFFNESS)
    drive_x.CreateDampingAttr(DRIVE_X_DAMPING)
    drive_x.CreateMaxForceAttr(2000.0)

    # --------------------------------------------------
    # LINK Y - MOTOR
    # --------------------------------------------------
    link_y_motor = DynamicCuboid(
        prim_path="/World/Robot/Link_Y_motor",
        position=np.array([0.0, 1.5, 1.5]),
        size=0.2,
        mass=1.5,
    )

    create_prim("/World/Robot/Joint_Y_motor", "PhysicsPrismaticJoint")
    joint_y_motor = UsdPhysics.PrismaticJoint.Get(
        world.stage, "/World/Robot/Joint_Y_motor"
    )
    joint_y_motor.CreateBody0Rel().SetTargets(["/World/Robot/Link_X"])
    joint_y_motor.CreateBody1Rel().SetTargets(["/World/Robot/Link_Y_motor"])
    joint_y_motor.CreateAxisAttr("Y")
    joint_y_motor.CreateLowerLimitAttr(-1.25)
    joint_y_motor.CreateUpperLimitAttr(1.25)

    drive_y_motor = UsdPhysics.DriveAPI.Apply(joint_y_motor.GetPrim(), "linear")
    drive_y_motor.CreateStiffnessAttr(30000.0)
    drive_y_motor.CreateDampingAttr(500.0)
    drive_y_motor.CreateMaxForceAttr(2000.0)

    # --------------------------------------------------
    # LINK Y - ELASTIC
    # --------------------------------------------------
    link_y = DynamicCuboid(
            prim_path="/World/Robot/Link_Y",
            position=np.array([0.0, 1.5, 1.5]),
            size=0.2,
            mass=2.0,
    )

    create_prim("/World/Robot/Joint_Y", "PhysicsPrismaticJoint")
    joint_y = UsdPhysics.PrismaticJoint.Get(
        world.stage, "/World/Robot/Joint_Y"
    )

    joint_y.CreateBody0Rel().SetTargets(["/World/Robot/Link_Y_motor"])
    joint_y.CreateBody1Rel().SetTargets(["/World/Robot/Link_Y"])
    joint_y.CreateAxisAttr("Y")

    drive_y = UsdPhysics.DriveAPI.Apply(
        joint_y.GetPrim(), "linear"
    )
    drive_y.CreateStiffnessAttr(DRIVE_Y_STIFFNESS)
    drive_y.CreateDampingAttr(DRIVE_Y_DAMPING)
    drive_y.CreateMaxForceAttr(2000.0)

    # --------------------------------------------------
    # LINK Z - MOTOR
    # --------------------------------------------------
    link_z_motor = DynamicCuboid(
        prim_path="/World/Robot/Link_Z_motor",
        position=np.array([1.5, 1.5, 1.5]),
        size=0.2,
        mass=0.5,
    )

    link_z_motor.set_collision_enabled(False)

    create_prim("/World/Robot/Joint_Z_motor", "PhysicsPrismaticJoint")
    joint_z_motor = UsdPhysics.PrismaticJoint.Get(
        world.stage, "/World/Robot/Joint_Z_motor"
    )
    joint_z_motor.CreateBody0Rel().SetTargets(["/World/Robot/Link_Y"])
    joint_z_motor.CreateBody1Rel().SetTargets(["/World/Robot/Link_Z_motor"])
    joint_z_motor.CreateAxisAttr("Z")
    joint_z_motor.CreateLowerLimitAttr(-1.25)
    joint_z_motor.CreateUpperLimitAttr(1.25)

    drive_z_motor = UsdPhysics.DriveAPI.Apply(joint_z_motor.GetPrim(), "linear")
    drive_z_motor.CreateStiffnessAttr(30000.0)
    drive_z_motor.CreateDampingAttr(500.0)
    drive_z_motor.CreateMaxForceAttr(2000.0)

    # --------------------------------------------------
    # LINK Z - ELASTIC
    # --------------------------------------------------
    link_z = DynamicCuboid(
        prim_path="/World/Robot/Link_Z",
        position=np.array([1.5, 1.5, 1.5]),
        size=0.2,
        mass=0.5,
    )

    create_prim("/World/Robot/Joint_Z", "PhysicsPrismaticJoint")
    joint_z = UsdPhysics.PrismaticJoint.Get(
        world.stage, "/World/Robot/Joint_Z"
    )

    joint_z.CreateBody0Rel().SetTargets(["/World/Robot/Link_Z_motor"])
    joint_z.CreateBody1Rel().SetTargets(["/World/Robot/Link_Z"])
    joint_z.CreateAxisAttr("Z")

    drive_z = UsdPhysics.DriveAPI.Apply(
        joint_z.GetPrim(), "linear"
    )
    drive_z.CreateStiffnessAttr(DRIVE_Z_STIFFNESS)
    drive_z.CreateDampingAttr(DRIVE_Z_DAMPING)
    drive_z.CreateMaxForceAttr(2000.0)

    # --------------------------------------------------
    # Payload (dynamic)
    # --------------------------------------------------
    payload = DynamicCuboid(
        prim_path="/World/Robot/Payload",
        position=np.array([1.5, 1.5, 1.4]),
        color=np.array([1.0, 0.0, 0.0]),
        size=0.15,
        mass=PAYLOAD,
    )

    create_prim("/World/Robot/PayloadJoint", "PhysicsFixedJoint")

    payload_joint = UsdPhysics.FixedJoint.Get(
        world.stage,
        "/World/Robot/PayloadJoint"
    )

    payload_joint.CreateBody0Rel().SetTargets(["/World/Robot/Link_Z"])
    payload_joint.CreateBody1Rel().SetTargets(["/World/Robot/Payload"])

    # --------------------------------------------------
    # Simulation
    # --------------------------------------------------

    robot = Robot(
        prim_path="/World/Robot",
        name="elastic_cartesian"
    )

    world.scene.add(robot)

    world.reset()
    robot.initialize()
    time = 0.0
    print("robot dof_names:", robot.dof_names)
    print("Physics dt:", world.get_physics_dt())
    print("Rendering dt:", world.get_rendering_dt())
    print("GPU dynamics:", physics_context.is_gpu_dynamics_enabled())

    if SAVE_CSV:
        data = []
    if SHOW_PLOT:
        time_history = []
        ref_pos_plt = []
        ref_vel_plt = []
        pos_plt = []
        vel_plt = []
        tau_plt = []
        tau_nom_plt = []

    stiffness_vector = np.array([DRIVE_X_STIFFNESS, DRIVE_Y_STIFFNESS, DRIVE_Z_STIFFNESS])
    damping_vector = np.array([DRIVE_X_DAMPING, DRIVE_Y_DAMPING, DRIVE_Z_DAMPING])

    # Build reference trajectory from settings.yaml (no magic numbers)
    _config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "settings.yaml")
    with open(_config_path, encoding="utf-8") as _f:
        _raw_settings = yaml.load(_f, Loader=SafeLoader)
    tset = load_trajectory_settings(_raw_settings)
    validate_trajectory_settings(tset)
    if REF_MODE == 0:
        _traj_obj = make_hold_trajectory(np.zeros(3), SIM_TIME, tset)
        print("\nReference mode 0: holding zero position.\n")
    elif REF_MODE == 1:
        _traj_obj = make_sinusoidal_trajectory(tset, SIM_TIME, seed, time_step=TIME_STEP)
        print(f"\nSinusoidal trajectory: peak={_traj_obj.config.executed_peak_velocity_ms:.3f} m/s  "
              f"factor={_traj_obj.config.global_speed_factor:.3f}\n")
    elif REF_MODE == 2:
        _traj_obj = make_ptp_trajectory(tset, SIM_TIME, seed, time_step=TIME_STEP)
        print(f"\nPTP trajectory: peak={_traj_obj.config.executed_peak_velocity_ms:.3f} m/s  "
              f"factor={_traj_obj.config.global_speed_factor:.3f}\n")
    else:
        raise ValueError(f"Invalid ref_mode: {REF_MODE}")

    # -----------------------------
    # Sensor noise parameters
    # -----------------------------
    # Position sensor
    ENCODER_RESOLUTION = 1.2e-8
    ENCODER_NOISE_POS = 5e-6

    # Velocity
    ENCODER_NOISE_VEL = 1e-2

    # Torque
    TORQUE_NOISE_REL = 0.03
    TORQUE_NOISE_ABS = 0.01

    while simulation_app.is_running() and time < SIM_TIME:
        world.step()

        # Retrieve joint states
        q = robot.get_joint_positions()
        dq = robot.get_joint_velocities()
        tau = robot.get_measured_joint_efforts()

        # Add sensor noise to measurements
        q_meas = np.round(q / ENCODER_RESOLUTION) * ENCODER_RESOLUTION
        q_meas += np.random.normal(0, ENCODER_NOISE_POS, size=q.shape)
        dq_meas = dq + np.random.normal(0, ENCODER_NOISE_VEL, size=dq.shape)
        tau_meas = tau + np.random.normal(
            0,
            TORQUE_NOISE_REL*np.abs(tau) + TORQUE_NOISE_ABS,
            size=tau.shape
        )

        q_motor = q_meas[[0,2,4]]
        q_link  = q_meas[[1,3,5]]

        dq_motor = dq_meas[[0,2,4]]
        dq_link  = dq_meas[[1,3,5]]

        tau_motor = tau_meas[[0,2,4]]
        tau_link  = tau_meas[[1,3,5]]

        tau_nominal = (
            stiffness_vector * q_link +
            damping_vector * dq_link
        )

        # Compute reference trajectory
        _q_ref, _dq_ref = _traj_obj(time)
        ref_x, ref_y, ref_z = _q_ref
        vel_x, vel_y, vel_z = _dq_ref

        action = ArticulationAction(
            joint_positions=np.array([ref_x, 0.0, ref_y, 0.0, ref_z, 0.0]),
            joint_velocities=np.array([vel_x, 0.0, vel_y, 0.0, vel_z, 0.0])
        )

        robot.apply_action(action)

        # Log or plot data
        time += world.get_physics_dt()

        if SAVE_CSV:
            if time >= CUT_OFF_TIME:
                data.append([
                time,
                ref_x, ref_y, ref_z,
                vel_x, vel_y, vel_z,
                q_motor[0], q_link[0],
                q_motor[1], q_link[1],
                q_motor[2], q_link[2],
                dq_motor[0], dq_link[0],
                dq_motor[1], dq_link[1],
                dq_motor[2], dq_link[2],
                tau_motor[0], tau_link[0],
                tau_motor[1], tau_link[1],
                tau_motor[2], tau_link[2] 
            ])
        if SHOW_PLOT:
            if time >= CUT_OFF_TIME:
                time_history.append(time)
                ref_pos_plt.append([ref_x, ref_y, ref_z])
                ref_vel_plt.append([vel_x, vel_y, vel_z])
                pos_plt.append([q_motor[i] + q_link[i] for i in range(3)])
                vel_plt.append([dq_motor[i] + dq_link[i] for i in range(3)])
                tau_plt.append([tau_motor[i] + tau_link[i] for i in range(3)])
                tau_nom_plt.append(-tau_nominal)

    if SAVE_CSV:
        df = pd.DataFrame(
            data,
            columns=[
                "t",
                "ref_x","ref_y","ref_z",
                "vel_x","vel_y","vel_z",
                "q_motor_x","q_link_x",
                "q_motor_y","q_link_y",
                "q_motor_z","q_link_z",
                "dq_motor_x","dq_link_x",
                "dq_motor_y","dq_link_y",
                "dq_motor_z","dq_link_z",
                "tau_motor_x","tau_link_x",
                "tau_motor_y","tau_link_y",
                "tau_motor_z","tau_link_z"
            ]
        )
        timestamp = datetime.now().strftime("%H%M%S")
        payload_str = f"{PAYLOAD:.1f}kg"
        drive_x_str = f"X{DRIVE_X_STIFFNESS:.0f}_{DRIVE_X_DAMPING:.0f}"
        drive_y_str = f"Y{DRIVE_Y_STIFFNESS:.0f}_{DRIVE_Y_DAMPING:.0f}"
        drive_z_str = f"Z{DRIVE_Z_STIFFNESS:.0f}_{DRIVE_Z_DAMPING:.0f}"
        trajectory_str = f"ref{REF_MODE}"
        csv_filename = os.path.join(os.path.dirname(__file__), "..", "data", f"d{timestamp}_s{seed}_p{payload_str}_{drive_x_str}_{drive_y_str}_{drive_z_str}_{trajectory_str}.csv")
        df.to_csv(csv_filename, index=False)
        print(f"Elastic dataset saved as {csv_filename}")

    # Debug plots
    if SHOW_PLOT:
        debug_kinematics(
            np.array(ref_pos_plt),
            np.array(ref_vel_plt),
            np.array(pos_plt),
            np.array(vel_plt),
            time_history
        )
        debug_dynamics(
            np.array(tau_plt)[:, 0],
            np.array(tau_plt)[:, 1],
            np.array(tau_plt)[:, 2],
            np.array(tau_nom_plt)[:, 0],
            np.array(tau_nom_plt)[:, 1],
            np.array(tau_nom_plt)[:, 2],
            time_history
        )
        plt.show()

    simulation_app.close()

if __name__ == "__main__":
    main()