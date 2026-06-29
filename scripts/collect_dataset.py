#!/usr/bin/env python3
"""Collect real-robot calibration data: trajectory configs + real recordings.

Reads default parameters from config/settings.yaml (collection section).
Every CLI argument overrides the corresponding settings.yaml value.

The script generates random sinusoidal / PTP trajectories, saves their
configs (with seeds) for full reproducibility, then executes each one on
the real robot via ROS 2 and records the joint/force signals.

Motor lifecycle:
  - Motors are enabled ONCE before the first trajectory via
    /ethercat_checker/start_motors (Trigger).
  - Motors are disabled ONCE after the last trajectory (or on interrupt/error)
    via /ethercat_checker/stop_motors (Trigger).

The resulting files are the ground truth used by run_calibration.py:

    data/recordings/<session>/
        trajectory_20240605_143022.json   ← seed + waypoints for reproducibility
        real_20240605_143022.parquet      ← recorded joint states + F/T
        trajectory_20240605_143058.json
        real_20240605_143058.parquet
        ...

Usage
-----
# Use all settings from config/settings.yaml (output_dir included)
ros2 run elastic_robot_sim collect_dataset

# Override output directory and number of trajectories
ros2 run elastic_robot_sim collect_dataset \\
    --output-dir data/recordings/session_01 \\
    --num-trajectories 20

# Override specific settings on the CLI
ros2 run elastic_robot_sim collect_dataset \\
    --output-dir data/recordings/session_01 \\
    --modes ptp --sim-time 12.0

# Skip motor enable/disable (e.g. motors already on, or dry testing)
ros2 run elastic_robot_sim collect_dataset --no-motor-control

# Point at a different settings file
ros2 run elastic_robot_sim collect_dataset \\
    --settings /path/to/settings.yaml
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from datetime import datetime

import yaml

# ---------------------------------------------------------------------------
# Path bootstrap for running from the source tree (no-op when installed)
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, ".."))
_SRC = os.path.join(_REPO, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from elastic_sim.trajectory import (
    load_trajectory_settings,
    validate_trajectory_settings,
    make_ptp_trajectory,
    make_sinusoidal_trajectory,
)

_HAS_ROS = False
try:
    import rclpy
    from rclpy.node import Node
    from std_srvs.srv import Trigger
    from elastic_sim.ros_recorder import RealRobotRecorder, FT_TOPIC_DEFAULT
    _HAS_ROS = True
except ImportError:
    FT_TOPIC_DEFAULT = "/ft_sensor_command_broadcaster/wrench"


# ---------------------------------------------------------------------------
# Motor controller node (session-level, created once)
# ---------------------------------------------------------------------------

if _HAS_ROS:
    class MotorController(Node):
        """Thin node that wraps /ethercat_checker/start_motors and stop_motors."""

        _START_SRV = "/ethercat_checker/start_motors"
        _STOP_SRV  = "/ethercat_checker/stop_motors"

        def __init__(self) -> None:
            super().__init__("dataset_motor_controller")
            self._start = self.create_client(Trigger, self._START_SRV)
            self._stop  = self.create_client(Trigger, self._STOP_SRV)

        def enable(self, timeout: float = 15.0) -> bool:
            """Call start_motors. Returns True on success."""
            self.get_logger().info("Waiting for start_motors service...")
            if not self._start.wait_for_service(timeout_sec=timeout):
                self.get_logger().error(
                    f"{self._START_SRV} not available after {timeout:.0f} s."
                )
                return False
            future = self._start.call_async(Trigger.Request())
            rclpy.spin_until_future_complete(self, future)
            result: Trigger.Response = future.result()
            if result.success:
                self.get_logger().info("Motors enabled.")
            else:
                self.get_logger().error(
                    f"start_motors returned failure: {result.message}"
                )
            return result.success

        def disable(self, timeout: float = 15.0) -> bool:
            """Call stop_motors. Returns True on success (best-effort in finally)."""
            self.get_logger().info("Waiting for stop_motors service...")
            if not self._stop.wait_for_service(timeout_sec=timeout):
                self.get_logger().error(
                    f"{self._STOP_SRV} not available after {timeout:.0f} s."
                )
                return False
            future = self._stop.call_async(Trigger.Request())
            rclpy.spin_until_future_complete(self, future)
            result: Trigger.Response = future.result()
            if result.success:
                self.get_logger().info("Motors disabled.")
            else:
                self.get_logger().error(
                    f"stop_motors returned failure: {result.message}"
                )
            return result.success


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

def _default_settings_path() -> str:
    """Return the installed settings.yaml path, falling back to the source tree."""
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(
            get_package_share_directory("elastic_robot_sim"), "config", "settings.yaml"
        )
    except Exception:
        return os.path.join(_REPO, "config", "settings.yaml")


def _load_settings(path: str | None) -> dict:
    if path is None:
        path = _default_settings_path()
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Collect real-robot calibration recordings via ROS 2.  "
            "Defaults are read from config/settings.yaml (collection section)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--output-dir", default=None,
        help="Directory where trajectory JSON files and parquet files are saved "
             "(default: collection.output_dir in settings.yaml).",
    )
    p.add_argument(
        "--settings", default=None, metavar="PATH",
        help="Path to a settings YAML file.",
    )
    p.add_argument(
        "--num-trajectories", type=int, default=None, metavar="N",
        help="Number of trajectories to collect (0 = run until Ctrl-C).",
    )
    p.add_argument(
        "--modes", nargs="+", choices=["sin", "ptp"], default=None,
        help="Trajectory modes to randomly sample.",
    )
    p.add_argument(
        "--sim-time", type=float, default=None,
        help="Duration of each trajectory in seconds.",
    )
    p.add_argument(
        "--master-seed", type=int, default=None,
        help="Seed for the trajectory-seed RNG (null = random).",
    )
    p.add_argument(
        "--ft-topic", default=None,
        help="ROS 2 force-torque topic name.",
    )
    p.add_argument(
        "--speed-override", type=float, default=None, metavar="PCT",
        help="Trajectory speed as %% of nominal (1–100). Default from settings.yaml.",
    )
    p.add_argument(
        "--no-motor-control", action="store_true",
        help="Skip motor enable/disable (use when motors are already on, or for dry testing).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Robot subprocess
# ---------------------------------------------------------------------------

def _start_robot(
    traj_json: str,
    output_file: str,
    ros_python: str,
    ft_topic: str,
    record_rate_hz: float,
) -> subprocess.Popen:
    script = os.path.join(_REPO, "real_robot", "record_real_rollout.py")
    cmd = [
        ros_python, script,
        "--traj-config", traj_json,
        "--output-file", output_file,
        "--ft-topic", ft_topic,
        "--record-rate-hz", str(record_rate_hz),
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    settings = _load_settings(args.settings)
    col = settings.get("collection", {})

    # Load and validate trajectory settings (single source of truth)
    tsettings = load_trajectory_settings(settings)
    validate_trajectory_settings(tsettings)

    # Resolve effective collection config (CLI wins over settings.yaml)
    num_trajectories: int = args.num_trajectories if args.num_trajectories is not None else col.get("num_trajectories", 10)
    modes: list[str]      = args.modes            if args.modes is not None            else col.get("modes", ["sin", "ptp"])
    sim_time: float       = args.sim_time          if args.sim_time is not None          else col.get("sim_time", 15.0)
    master_seed           = args.master_seed       if args.master_seed is not None       else col.get("master_seed", None)
    ros_python: str       = args.ros_python        if args.ros_python is not None        else col.get("ros_python", "python3")
    ft_topic: str         = args.ft_topic          if args.ft_topic is not None          else col.get("ft_topic", "/ft_sensor/wrench")
    speed_override: float = col.get("speed_override", 100.0)
    record_rate_hz: float = col.get("record_rate_hz", 100.0)
    time_step: float      = settings.get("simulation", {}).get("time_step", 0.01)

    os.makedirs(output_dir, exist_ok=True)

    rng = random.Random(master_seed)
    infinite = num_trajectories == 0

    print(f"Output dir      : {output_dir}")
    print(f"Trajectory modes: {modes}")
    print(f"Duration        : {sim_time} s")
    print(f"Joint limits    : {dict(tsettings.joint_limits)}")
    print(f"Peak vel limit  : {tsettings.peak_cartesian_velocity_ms} m/s")
    print(f"Speed override  : {speed_override}%")
    print(f"Record rate     : {record_rate_hz} Hz")
    print(f"Master seed     : {master_seed if master_seed is not None else 'random'}")
    print(f"F/T topic       : {ft_topic}")
    print(f"Speed override  : {speed_override:.0f}%")
    print(f"Motor control   : {'enabled' if motor_control else 'disabled (--no-motor-control)'}")
    print(f"Trajectories    : {'unlimited (Ctrl-C to stop)' if infinite else num_trajectories}")

    if not _HAS_ROS:
        print("ERROR: ROS 2 packages not found. Source your ROS 2 workspace first.")
        sys.exit(1)

    rclpy.init()
    motors: MotorController | None = None
    count = 0

    try:
        # ------------------------------------------------------------------
        # Enable motors once for the whole session
        # ------------------------------------------------------------------
        if motor_control:
            motors = MotorController()
            print("\n[motors] Enabling motors...")
            if not motors.enable():
                print("[motors] ERROR: failed to enable motors — aborting.")
                sys.exit(1)
            print("[motors] Motors enabled.\n")

        # ------------------------------------------------------------------
        # Trajectory collection loop
        # ------------------------------------------------------------------
        while infinite or count < num_trajectories:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            seed = rng.randint(0, 2**31 - 1)
            mode_str = rng.choice(modes)

            print(f"\n{'='*60}")
            print(
                f"  Recording {count + 1}"
                f"{'/' + str(num_trajectories) if not infinite else ''}"
                f"  ts={ts}  mode={mode_str}  seed={seed}"
            )

            # Generate and save trajectory — baked config is self-describing
            if mode_str == "sin":
                traj = make_sinusoidal_trajectory(
                    tsettings, sim_time, seed,
                    speed_override=speed_override, time_step=time_step,
                )
            else:
                traj = make_ptp_trajectory(
                    tsettings, sim_time, seed,
                    speed_override=speed_override, time_step=time_step,
                )

            cfg = traj.config
            nom_peak = cfg.executed_peak_velocity_ms / cfg.global_speed_factor if cfg.global_speed_factor > 0 else 0.0
            print(f"  speed: nominal_peak={nom_peak:.3f} "
                  f"factor={cfg.global_speed_factor:.3f} "
                  f"executed_peak={cfg.executed_peak_velocity_ms:.3f} m/s "
                  f"duration={cfg.sim_time:.2f}s")

            traj_json = os.path.join(output_dir, f"trajectory_{ts}.json")
            traj.config.save(traj_json)
            print(f"  Trajectory saved: trajectory_{ts}.json")

            # Execute on real robot and record
            real_out = os.path.join(output_dir, f"real_{ts}.parquet")
            print("  [real] Connecting to action server...", flush=True)
            node = RealRobotRecorder(traj.config, ft_topic=ft_topic, speed_override=speed_override)
            try:
                ok = node.send_trajectory()
                if ok:
                    # Allow final sensor samples to arrive before saving
                    rclpy.spin_once(node, timeout_sec=1.0)
                    node.save_rollout(output_file=real_out)
                    print(f"  [real] Saved: real_{ts}.parquet")
                else:
                    print("  [real] FAILED — trajectory config kept for debugging")
            finally:
                node.destroy_node()

            count += 1

    except KeyboardInterrupt:
        print(f"\nInterrupted after {count} completed recordings.")

    finally:
        # ------------------------------------------------------------------
        # Disable motors once after all trajectories (or on error/interrupt)
        # ------------------------------------------------------------------
        if motor_control and motors is not None:
            print("\n[motors] Disabling motors...")
            motors.disable()
            motors.destroy_node()

        rclpy.shutdown()

    print(f"\nCollection complete: {count} trajectories saved to {output_dir}")
    print(
        f"\nNext — run calibration:\n"
        f"  python scripts/run_calibration.py --recordings-dir {output_dir}"
    )


if __name__ == "__main__":
    main()
