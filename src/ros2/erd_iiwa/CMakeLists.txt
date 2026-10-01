cmake_minimum_required(VERSION 3.16)
project(erd_iiwa)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
add_compile_options(-Wall -Wextra)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)

# The bundled FRI 1.11 SDK from the pinned `iiwa_ros2` checkout (RR_01 S3.2):
# read its headers/sources at build time from the workspace, never copied
# into this (public) repository -- KUKA's SDK license permits use with KUKA
# products, not redistribution. Point ERD_FRI_SDK_ROOT elsewhere if your
# workspace layout differs from `~/projects/erd_ws/src/{erd,iiwa_ros2}`.
set(ERD_FRI_SDK_ROOT "$ENV{HOME}/projects/erd_ws/src/iiwa_ros2/iiwa_hardware/external/libFRI"
    CACHE PATH "Path to the iiwa_ros2-bundled libFRI 1.11 SDK (headers + nanopb sources)")
if(NOT EXISTS "${ERD_FRI_SDK_ROOT}/include/friLBRState.h")
  message(FATAL_ERROR
    "ERD_FRI_SDK_ROOT (${ERD_FRI_SDK_ROOT}) does not contain friLBRState.h. "
    "Run `vcs import src < src/erd/erd.repos` in the erd_ws workspace first, "
    "or pass -DERD_FRI_SDK_ROOT=<path> to colcon build.")
endif()
set(FRI_INCLUDES
  ${ERD_FRI_SDK_ROOT}/include
  ${ERD_FRI_SDK_ROOT}/src/protobuf_gen
  ${ERD_FRI_SDK_ROOT}/src/nanopb-0.2.8
  ${ERD_FRI_SDK_ROOT}/src/protobuf
  ${ERD_FRI_SDK_ROOT}/src/base
  ${ERD_FRI_SDK_ROOT}/src/connection
  ${ERD_FRI_SDK_ROOT}/src/client_lbr
  ${ERD_FRI_SDK_ROOT}/src/client_trafo
)
file(GLOB FRI_SOURCES
  ${ERD_FRI_SDK_ROOT}/src/base/*.cpp
  ${ERD_FRI_SDK_ROOT}/src/connection/*.cpp
  ${ERD_FRI_SDK_ROOT}/src/client_lbr/*.cpp
  ${ERD_FRI_SDK_ROOT}/src/client_trafo/*.cpp
  ${ERD_FRI_SDK_ROOT}/src/protobuf/*.cpp
  ${ERD_FRI_SDK_ROOT}/src/protobuf/*.c
  ${ERD_FRI_SDK_ROOT}/src/protobuf_gen/*.c
  ${ERD_FRI_SDK_ROOT}/src/nanopb-0.2.8/*.c
)

add_library(erd_iiwa_hardware SHARED
  src/erd_fri_client.cpp
  src/fri_position_system.cpp
  ${FRI_SOURCES}
)
target_compile_features(erd_iiwa_hardware PUBLIC cxx_std_17)
target_include_directories(erd_iiwa_hardware PRIVATE include ${FRI_INCLUDES})
target_compile_definitions(erd_iiwa_hardware PRIVATE HAVE_SOCKLEN_T PB_FIELD_16BIT)
# -Wall/-Wextra apply to our sources only (RR_01 S3.2); the bundled FRI SDK
# sources compile with their own (much noisier) upstream warnings suppressed.
set_source_files_properties(${FRI_SOURCES} PROPERTIES COMPILE_OPTIONS "-w")
ament_target_dependencies(erd_iiwa_hardware hardware_interface pluginlib rclcpp rclcpp_lifecycle)
pluginlib_export_plugin_description_file(hardware_interface hardware_plugins.xml)

install(TARGETS erd_iiwa_hardware DESTINATION lib)
install(DIRECTORY launch config description sunrise DESTINATION share/${PROJECT_NAME})
ament_python_install_package(erd_iiwa)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_description_interfaces test/test_description_interfaces.py)
  ament_add_gtest(test_fri_position_system test/test_fri_position_system.cpp)
  target_include_directories(test_fri_position_system PRIVATE include ${FRI_INCLUDES})
  target_compile_definitions(test_fri_position_system PRIVATE HAVE_SOCKLEN_T PB_FIELD_16BIT)
  target_link_libraries(test_fri_position_system erd_iiwa_hardware)
  ament_target_dependencies(test_fri_position_system hardware_interface pluginlib rclcpp rclcpp_lifecycle)
endif()

ament_package()
