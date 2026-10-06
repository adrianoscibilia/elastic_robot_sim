cmake_minimum_required(VERSION 3.16)
project(erd_fri_emulator)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
add_compile_options(-Wall -Wextra)

find_package(ament_cmake REQUIRED)
find_package(pinocchio REQUIRED)

# Same bundled FRI 1.11 SDK as erd_iiwa (RR_01 S3.2/S3.5), read at build time
# from the workspace, never copied into this (public) repository. The
# emulator only needs the message definitions (nanopb runtime + the
# generated FRIMessages.pb.* + pb_frimessages_callbacks) -- it speaks raw UDP
# itself (erd_fri_emulator/emulator.cpp), not KUKA::FRI::UdpConnection/
# ClientApplication, which are client-only abstractions (S1.9 finding).
# Default: $ERD_FRI_SDK_ROOT (exported by workspace_setup.sh), else the
# in-repo workspace checkout <repo>/ros2_ws/src/external/iiwa_ros2 (resolved
# from this file's real path, since ros2_ws/src/erd is a symlink to src/ros2).
if(DEFINED ENV{ERD_FRI_SDK_ROOT})
  set(_erd_fri_default "$ENV{ERD_FRI_SDK_ROOT}")
else()
  get_filename_component(_erd_pkg_dir "${CMAKE_CURRENT_LIST_DIR}" REALPATH)
  set(_erd_fri_default
    "${_erd_pkg_dir}/../../../ros2_ws/src/external/iiwa_ros2/iiwa_hardware/external/libFRI")
endif()
set(ERD_FRI_SDK_ROOT "${_erd_fri_default}"
  CACHE PATH "Path to the iiwa_ros2-bundled libFRI 1.11 SDK (headers + nanopb sources)")
if(NOT EXISTS "${ERD_FRI_SDK_ROOT}/include/friLBRState.h")
  message(FATAL_ERROR
    "ERD_FRI_SDK_ROOT (${ERD_FRI_SDK_ROOT}) does not contain friLBRState.h. "
    "Run `source workspace_setup.sh` from the repository root first (it fetches "
    "src/ros2/erd.repos into ros2_ws/src/external), "
    "or pass -DERD_FRI_SDK_ROOT=<path> to colcon build.")
endif()
set(FRI_MESSAGE_INCLUDES
  ${ERD_FRI_SDK_ROOT}/src/protobuf_gen
  ${ERD_FRI_SDK_ROOT}/src/nanopb-0.2.8
  ${ERD_FRI_SDK_ROOT}/src/protobuf
)
set(FRI_MESSAGE_SOURCES
  ${ERD_FRI_SDK_ROOT}/src/protobuf_gen/FRIMessages.pb.c
  ${ERD_FRI_SDK_ROOT}/src/protobuf/pb_frimessages_callbacks.c
  ${ERD_FRI_SDK_ROOT}/src/nanopb-0.2.8/pb_encode.c
  ${ERD_FRI_SDK_ROOT}/src/nanopb-0.2.8/pb_decode.c
)

add_library(erd_fri_emulator_core STATIC
  src/robot_side_messages.cpp
  src/kinematic_plant.cpp
  src/emulator.cpp
  ${FRI_MESSAGE_SOURCES}
)
target_include_directories(erd_fri_emulator_core PUBLIC include ${FRI_MESSAGE_INCLUDES})
target_compile_definitions(erd_fri_emulator_core PUBLIC HAVE_SOCKLEN_T PB_FIELD_16BIT)
target_link_libraries(erd_fri_emulator_core PUBLIC pinocchio::pinocchio)
set_source_files_properties(${FRI_MESSAGE_SOURCES} PROPERTIES COMPILE_OPTIONS "-w")

add_executable(erd_fri_emulator_node src/main.cpp)
target_link_libraries(erd_fri_emulator_node erd_fri_emulator_core)
set_target_properties(erd_fri_emulator_node PROPERTIES OUTPUT_NAME erd_fri_emulator)

install(TARGETS erd_fri_emulator_node DESTINATION lib/${PROJECT_NAME})
install(PROGRAMS scripts/run_emulator.sh DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(Threads REQUIRED)
  # The test cross-checks wire compatibility against the real FRI SDK's own
  # client-side codec classes (T1.9); these two .cpp files are the SDK's,
  # compiled only into the test binary (production erd_fri_emulator code
  # never needs the client-direction codecs).
  ament_add_gtest(test_emulator
    test/test_emulator.cpp
    ${ERD_FRI_SDK_ROOT}/src/protobuf/friCommandMessageEncoder.cpp
    ${ERD_FRI_SDK_ROOT}/src/protobuf/friMonitoringMessageDecoder.cpp
  )
  target_include_directories(test_emulator PRIVATE ${ERD_FRI_SDK_ROOT}/include ${FRI_MESSAGE_INCLUDES})
  # Default test URDF: the bare asset in this repository, located from this
  # file's real path (ros2_ws/src/erd is a symlink); ERD_TEST_URDF overrides.
  get_filename_component(_erd_test_pkg_dir "${CMAKE_CURRENT_LIST_DIR}" REALPATH)
  get_filename_component(_erd_test_urdf
    "${_erd_test_pkg_dir}/../../../assets/robots/kuka_lbr_iiwa_14_r820/description/kuka_lbr_iiwa_14_r820.urdf"
    ABSOLUTE)
  target_compile_definitions(test_emulator PRIVATE HAVE_SOCKLEN_T PB_FIELD_16BIT
    "ERD_TEST_URDF_DEFAULT=\"${_erd_test_urdf}\"")
  target_link_libraries(test_emulator erd_fri_emulator_core Threads::Threads)
endif()

ament_package()
