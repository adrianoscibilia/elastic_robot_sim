cmake_minimum_required(VERSION 3.8)
project(elastic_robot_sim)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# Install the elastic_sim Python library (required by both executables).
# All other simulation code (scripts/sim backends, tests/) is left untouched.
ament_python_install_package(elastic_sim
  PACKAGE_DIR src/elastic_sim
)

# Install config/ so ament_index_python can locate settings.yaml at runtime.
install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

# Install ROS 2 executables without the .py suffix (ros2 run matches by exact name):
#   ros2 run elastic_robot_sim record_real_rollout  — single trajectory
#   ros2 run elastic_robot_sim collect_dataset      — full dataset collection loop
install(PROGRAMS real_robot/record_real_rollout.py
  RENAME record_real_rollout
  DESTINATION lib/${PROJECT_NAME}
)
install(PROGRAMS scripts/collect_dataset.py
  RENAME collect_dataset
  DESTINATION lib/${PROJECT_NAME}
)
install(PROGRAMS scripts/generate_serial_trajectory.py
  RENAME generate_serial_trajectory
  DESTINATION lib/${PROJECT_NAME}
)
install(PROGRAMS scripts/run_dataset_calibration.py
  RENAME run_dataset_calibration
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
