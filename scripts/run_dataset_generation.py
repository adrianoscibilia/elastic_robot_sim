import subprocess
import yaml
import copy
import random
import os
import math

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

newton_python = os.path.join(
    project_root,
    "..",
    "venvs",
    "env_newton",
    "Scripts",
    "python.exe",
)

sim_script = os.path.join(
    project_root,
    "scripts",
    "elastic_cart_robot_newton.py",
)

settings_file = os.path.join(
    project_root,
    "config",
    "settings.yaml",
)

# -----------------------------
# Parameter intervals
# -----------------------------
# Use interval extremes, then sample random values inside them. The ranges are
# chosen to stay plausible for an elastic Cartesian platform while still
# covering enough compliance variation to be informative for learning.
# Keys match the URDF joint names discovered by the generic builder.
STIFFNESS_INTERVALS = {
    "joint_x": (3000.0, 9000.0),
    "joint_y": (3000.0, 9000.0),
    "joint_z": (2500.0, 7000.0),
}

# Sample damping ratios rather than raw damping values so the resulting robot
# configurations span underdamped to slightly overdamped responses in a more
# meaningful way.  The actual damping coefficient is computed at simulation time
# inside elastic_cart_robot_newton.py via d = 2 * zeta * sqrt(k * m).
DAMPING_RATIO_INTERVALS = {
    "joint_x": (0.45, 1.05),
    "joint_y": (0.45, 1.05),
    "joint_z": (0.55, 1.20),
}

# Mapping from internal joint names to the config drive keys expected by
# elastic_cart_robot_newton.py.
JOINT_TO_DRIVE = {
    "joint_x": "drive_x",
    "joint_y": "drive_y",
    "joint_z": "drive_z",
}

PAYLOAD_INTERVAL = (0.0, 6.0)
REF_MODES = [1, 2]  # 1 sinusoidal, 2 ptp
RNG = random.Random(42)
ROBOT_CONFIG_COUNT = 10
TRIALS_PER_ROBOT = 10
PAYLOAD_LEVEL_COUNT = 0


def sample_uniform(interval):
    lower, upper = interval
    return RNG.uniform(lower, upper)


def sample_stiffness(axis_name):
    # Log-uniform sampling covers both compliant and stiff behaviors without
    # over-sampling the high end of the interval.
    lower, upper = STIFFNESS_INTERVALS[axis_name]
    return 10.0 ** RNG.uniform(math.log10(lower), math.log10(upper))


def sample_damping_ratio(axis_name):
    return sample_uniform(DAMPING_RATIO_INTERVALS[axis_name])


def build_payload_levels():
    lower, upper = PAYLOAD_INTERVAL
    if PAYLOAD_LEVEL_COUNT < 2:
        return [round(lower, 3)]
    step = (upper - lower) / (PAYLOAD_LEVEL_COUNT - 1)
    return [round(lower + idx * step, 3) for idx in range(PAYLOAD_LEVEL_COUNT)]


def build_trial_plan():
    plan = []
    if PAYLOAD_LEVEL_COUNT != 0:
        payload_levels = build_payload_levels()
        for payload in payload_levels:
            for ref_mode in REF_MODES:
                plan.append((payload, ref_mode))
    else:
        for _ in range(TRIALS_PER_ROBOT):
            payload = 0.0
            ref_mode = RNG.choice(REF_MODES)
            plan.append((payload, ref_mode))
    if len(plan) != TRIALS_PER_ROBOT:
        raise ValueError(
            f"Trial plan has {len(plan)} entries but TRIALS_PER_ROBOT={TRIALS_PER_ROBOT}."
        )
    return plan

# -----------------------------
# Load template config
# -----------------------------
with open(settings_file) as f:
    base_config = yaml.safe_load(f)

N_EXPERIMENTS = ROBOT_CONFIG_COUNT * TRIALS_PER_ROBOT


def main():
    subprocess_env = os.environ.copy()

    trial_plan_template = build_trial_plan()

    try:
        experiment_idx = 0
        joint_names = list(STIFFNESS_INTERVALS.keys())
        for robot_idx in range(ROBOT_CONFIG_COUNT):
            robot_params = {}
            for jname in joint_names:
                k = round(sample_stiffness(jname), 2)
                zeta = round(sample_damping_ratio(jname), 4)
                robot_params[jname] = {"stiffness": k, "damping_ratio": zeta}

            param_str = " ".join(
                f"{jn}=(k={robot_params[jn]['stiffness']:.2f}, zeta={robot_params[jn]['damping_ratio']:.4f})"
                for jn in joint_names
            )
            print(f"\nRobot {robot_idx + 1}/{ROBOT_CONFIG_COUNT}: {param_str}")

            trial_plan = trial_plan_template.copy()
            RNG.shuffle(trial_plan)

            for trial_idx, (payload, ref_mode) in enumerate(trial_plan):
                config = copy.deepcopy(base_config)

                for jname in joint_names:
                    drive_key = JOINT_TO_DRIVE[jname]
                    config["elastic_drives"][drive_key] = robot_params[jname]

                config["elastic_drives"]["payload"] = payload
                config["simulation"]["ref_mode"] = ref_mode
                config["simulation"]["seed"] = 1000 + experiment_idx

                print(
                    f"  Trial {trial_idx + 1}/{TRIALS_PER_ROBOT} "
                    f"[{experiment_idx + 1}/{N_EXPERIMENTS}] "
                    f"payload={payload:.3f} ref_mode={ref_mode} seed={config['simulation']['seed']}"
                )

                with open(settings_file, "w") as f:
                    yaml.dump(config, f, sort_keys=False)

                subprocess.run(
                    [
                        newton_python,
                        sim_script,
                        "--csv",
                        "--headless",
                    ],
                    check=True,
                    env=subprocess_env,
                )
                experiment_idx += 1
    finally:
        with open(settings_file, "w") as f:
            yaml.dump(base_config, f, sort_keys=False)

    print("\nDataset generation completed.")


if __name__ == "__main__":
    main()
