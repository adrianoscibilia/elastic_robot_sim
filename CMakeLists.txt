cmake_minimum_required(VERSION 3.8)
project(elastic_robot_sim)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# Install elastic_sim so record_real_rollout.py can import it after installation.
# All other simulation code (scripts/, tests/) is left untouched by colcon.
ament_python_install_package(elastic_sim
  PACKAGE_DIR src/elastic_sim
)

# Install the single ROS 2 node as an executable (ros2 run elastic_robot_sim record_real_rollout)
install(PROGRAMS
  real_robot/record_real_rollout.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
