// Copyright 2026 elastic_robot_sim contributors
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <memory>
#include <vector>

#include "hardware_interface/system_interface.hpp"
#include "rclcpp_lifecycle/state.hpp"

#include "erd_iiwa/erd_fri_client.hpp"

namespace erd_iiwa
{

namespace detail
{

/// Pure decision function for RR_04 A-4: `read()`'s NaN-before-first-packet
/// behaviour is fine while the session has never reached COMMANDING_ACTIVE,
/// but once it has, a `step()` failure or leaving that state is a hardware
/// error -- the controller manager must deactivate the JTC rather than let
/// it keep holding (or extrapolating from) a stale/NaN setpoint. Kept free of
/// FRI I/O so the state-sequence behaviour is unit-testable without a live
/// connection (`test_fri_position_system.cpp`). `has_been_active` is updated
/// in place: once true, it never goes back to false for the life of the
/// activation (a fresh `on_activate()` resets it).
inline hardware_interface::return_type fri_read_return(
  bool step_ok, KUKA::FRI::ESessionState state, bool & has_been_active)
{
  if (!step_ok) {
    return has_been_active ? hardware_interface::return_type::ERROR :
           hardware_interface::return_type::OK;
  }
  const bool active_now = (state == KUKA::FRI::COMMANDING_ACTIVE);
  if (has_been_active && !active_now) {
    return hardware_interface::return_type::ERROR;
  }
  if (active_now) {
    has_been_active = true;
  }
  return hardware_interface::return_type::OK;
}

}  // namespace detail

/// `erd_iiwa/FriPositionSystem`: one ros2_control cycle per FRI packet
/// (RR_01 S3.3). `read()` blocks on `ErdFriClient::step()` (receive + send
/// the previous cycle's target), so the robot paces the loop 1:1; `write()`
/// only stages the new target for the next cycle (see erd_fri_client.hpp).
/// No filtering anywhere: `effort` is the raw measured torque, `velocity` is an
/// unfiltered finite difference of the raw measured position, for the JTC's
/// state interface only -- never part of the recorded/written signal.
class FriPositionSystem : public hardware_interface::SystemInterface
{
public:
  using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

  CallbackReturn on_init(const hardware_interface::HardwareInfo & info) override;
  CallbackReturn on_activate(const rclcpp_lifecycle::State & previous_state) override;
  CallbackReturn on_deactivate(const rclcpp_lifecycle::State & previous_state) override;
  std::vector<hardware_interface::StateInterface> export_state_interfaces() override;
  std::vector<hardware_interface::CommandInterface> export_command_interfaces() override;
  hardware_interface::return_type read(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;
  hardware_interface::return_type write(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;

private:
  std::unique_ptr<ErdFriClient> client_;

  std::array<double, kNumJoints> position_{};
  std::array<double, kNumJoints> velocity_{};
  std::array<double, kNumJoints> effort_{};
  std::array<double, kNumJoints> commanded_effort_{};
  std::array<double, kNumJoints> external_effort_{};
  std::array<double, kNumJoints> commanded_position_{};
  std::array<double, kNumJoints> ipo_position_{};
  std::array<double, kNumJoints> position_command_{};
  std::array<double, kNumJoints> previous_position_{};
  bool have_previous_{false};
  bool has_been_active_{false};

  // GPIO "fri" interfaces (RR_01 S3.3).
  double fri_time_sec_{0.0};
  double fri_time_nsec_{0.0};
  double fri_sample_time_{0.0};
  double fri_session_state_{0.0};
  double fri_command_mode_{0.0};
  double fri_connection_quality_{0.0};
  double fri_tracking_performance_{0.0};
  double fri_safety_state_{0.0};
  double fri_operation_mode_{0.0};
  double fri_drive_state_{0.0};
  double fri_cycle_{0.0};
  double fri_received_cycle_{0.0};

  void invalidate();
};

}  // namespace erd_iiwa
