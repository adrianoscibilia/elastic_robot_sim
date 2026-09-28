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
install(DIRECTORY assets/robots/
  DESTINATION share/${PROJECT_NAME}/assets/robots
  PATTERN "source" EXCLUDE
)
install(DIRECTORY docs/
  DESTINATION share/${PROJECT_NAME}/docs
)
install(FILES README.md assets/README.md
  DESTINATION share/${PROJECT_NAME}
)

# Install ROS 2 executables without the .py suffix (ros2 run matches by exact name):
#   ros2 run elastic_robot_sim run_experiment       — paired sim/real run
#   ros2 run elastic_robot_sim run_calibration      — paired-artifact calibration
install(PROGRAMS scripts/run_experiment.py
  RENAME run_experiment
  DESTINATION lib/${PROJECT_NAME}
)
install(PROGRAMS scripts/run_calibration.py
  RENAME run_calibration
  DESTINATION lib/${PROJECT_NAME}
)
install(PROGRAMS scripts/run_asset_simulation.py scripts/run_asset_dataset_generation.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
