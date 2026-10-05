// Copyright 2026 elastic_robot_sim contributors
// SPDX-License-Identifier: Apache-2.0
#include "erd_iiwa/fri_position_system.hpp"

#include <limits>
#include <set>
#include "rclcpp/rclcpp.hpp"

#include "pluginlib/class_list_macros.hpp"

namespace erd_iiwa
{

namespace
{
constexpr double kNan = std::numeric_limits<double>::quiet_NaN();
}

void FriPositionSystem::invalidate()
{
  position_.fill(kNan);
  velocity_.fill(kNan);
  effort_.fill(kNan);
  commanded_effort_.fill(kNan);
  external_effort_.fill(kNan);
  commanded_position_.fill(kNan);
  ipo_position_.fill(kNan);
  have_previous_ = false;
  fri_time_sec_ = fri_time_nsec_ = kNan;
  fri_session_state_ = fri_command_mode_ = fri_connection_quality_ = kNan;
  fri_tracking_performance_ = fri_safety_state_ = fri_operation_mode_ = fri_drive_state_ = kNan;
}

FriPositionSystem::CallbackReturn FriPositionSystem::on_init(
  const hardware_interface::HardwareInfo & info)
{
  if (hardware_interface::SystemInterface::on_init(info) != CallbackReturn::SUCCESS) {
    return CallbackReturn::ERROR;
  }
  if (info_.joints.size() != static_cast<size_t>(kNumJoints)) {
    return CallbackReturn::ERROR;
  }
  client_ = std::make_unique<ErdFriClient>();
  invalidate();
  position_command_.fill(kNan);
  fri_cycle_ = 0.0;
  return CallbackReturn::SUCCESS;
}

std::vector<hardware_interface::StateInterface> FriPositionSystem::export_state_interfaces()
{
  std::vector<hardware_interface::StateInterface> result;
  for (int i = 0; i < kNumJoints; ++i) {
    const auto & name = info_.joints[i].name;
    result.emplace_back(name, "position", &position_[i]);
    result.emplace_back(name, "velocity", &velocity_[i]);
    result.emplace_back(name, "effort", &effort_[i]);
    result.emplace_back(name, "commanded_effort", &commanded_effort_[i]);
    result.emplace_back(name, "external_effort", &external_effort_[i]);
    result.emplace_back(name, "commanded_position", &commanded_position_[i]);
    result.emplace_back(name, "ipo_position", &ipo_position_[i]);
  }
  result.emplace_back("fri", "time_sec", &fri_time_sec_);
  result.emplace_back("fri", "time_nsec", &fri_time_nsec_);
  result.emplace_back("fri", "sample_time", &fri_sample_time_);
  result.emplace_back("fri", "session_state", &fri_session_state_);
  result.emplace_back("fri", "command_mode", &fri_command_mode_);
  result.emplace_back("fri", "connection_quality", &fri_connection_quality_);
  result.emplace_back("fri", "tracking_performance", &fri_tracking_performance_);
  result.emplace_back("fri", "safety_state", &fri_safety_state_);
  result.emplace_back("fri", "operation_mode", &fri_operation_mode_);
  result.emplace_back("fri", "drive_state", &fri_drive_state_);
  result.emplace_back("fri", "cycle", &fri_cycle_);
  result.emplace_back("fri", "received_cycle", &fri_received_cycle_);
  return result;
}

std::vector<hardware_interface::CommandInterface> FriPositionSystem::export_command_interfaces()
{
  std::vector<hardware_interface::CommandInterface> result;
  for (int i = 0; i < kNumJoints; ++i) {
    result.emplace_back(info_.joints[i].name, "position", &position_command_[i]);
  }
  return result;
}

FriPositionSystem::CallbackReturn FriPositionSystem::on_activate(const rclcpp_lifecycle::State &)
{
  invalidate();
  position_command_.fill(kNan);
  has_been_active_ = false;  // RR_04 A-4: a fresh activation starts a fresh session
  try {
    const auto & params = info_.hardware_parameters;
    const int port = std::stoi(params.at("fri_port"));
    if (!client_->connect(port, params.at("robot_ip").c_str())) {
      return CallbackReturn::ERROR;
    }
  } catch (const std::exception &) {
    return CallbackReturn::ERROR;
  }
  return CallbackReturn::SUCCESS;
}

FriPositionSystem::CallbackReturn FriPositionSystem::on_deactivate(const rclcpp_lifecycle::State &)
{
  client_->disconnect();
  invalidate();
  return CallbackReturn::SUCCESS;
}

hardware_interface::return_type FriPositionSystem::read(
  const rclcpp::Time &, const rclcpp::Duration &)
{
  // One blocking FRI cycle: receives the new monitoring message, dispatches
  // to ErdFriClientImpl's callback (which sends the *previous* write()'s
  // target -- see erd_fri_client.hpp), and updates `sample()`.
  const bool step_ok = client_->step();
  const auto return_code = detail::fri_read_return(
    step_ok, step_ok ? client_->current_state() : KUKA::FRI::IDLE, has_been_active_);
  if (return_code == hardware_interface::return_type::ERROR) {
    // Session was COMMANDING_ACTIVE and is no longer (or the packet failed
    // to arrive at all): invalidate and tell the controller manager, so it
    // deactivates the JTC instead of holding/extrapolating a stale setpoint
    // (RR_04 A-4).
    RCLCPP_ERROR(rclcpp::get_logger("erd_fri"),
      "FRI read ERROR at received cycle %.0f", fri_received_cycle_);
    invalidate();
    return hardware_interface::return_type::ERROR;
  }
  if (!step_ok) {
    invalidate();
    return hardware_interface::return_type::OK;  // MONITORING_WAIT/IDLE: not yet an error
  }
  const auto & sample = client_->sample();
  position_ = sample.measured_position;
  effort_ = sample.measured_torque;              // raw, unfiltered (RR_01 S3.3)
  commanded_effort_ = sample.commanded_torque;
  external_effort_ = sample.external_torque;
  commanded_position_ = sample.commanded_position;
  ipo_position_ = sample.ipo_position;

  if (have_previous_ && sample.sample_time > 0.0) {
    for (int i = 0; i < kNumJoints; ++i) {
      // unfiltered finite diff
      velocity_[i] = (position_[i] - previous_position_[i]) / sample.sample_time;
    }
  }
  previous_position_ = position_;
  have_previous_ = true;

  fri_time_sec_ = static_cast<double>(sample.time_sec);
  fri_time_nsec_ = static_cast<double>(sample.time_nsec);
  fri_sample_time_ = sample.sample_time;
  fri_session_state_ = static_cast<double>(sample.session_state);
  fri_command_mode_ = static_cast<double>(sample.command_mode);
  fri_connection_quality_ = static_cast<double>(sample.connection_quality);
  fri_tracking_performance_ = sample.tracking_performance;
  fri_safety_state_ = static_cast<double>(sample.safety_state);
  fri_operation_mode_ = static_cast<double>(sample.operation_mode);
  fri_drive_state_ = static_cast<double>(sample.drive_state);
  // Remote packet sequence, not successful read count: lost packets remain visible.
  fri_cycle_ = static_cast<double>(sample.sequence_counter);
  fri_received_cycle_ += 1.0;
  return hardware_interface::return_type::OK;
}

hardware_interface::return_type FriPositionSystem::write(
  const rclcpp::Time &, const rclcpp::Duration &)
{
  // Stages the target for the *next* read()'s step() to send; see
  // erd_fri_client.hpp for why read()/write() do not map 1:1 onto FRI's
  // receive/send. Outside COMMANDING_ACTIVE, ErdFriClientImpl mirrors the
  // IPO position regardless of what is staged here (RR_01 S3.3).
  if (client_->current_state() == KUKA::FRI::COMMANDING_ACTIVE) {
    bool all_finite = true;
    for (double v : position_command_) {
      all_finite = all_finite && std::isfinite(v);
    }
    if (all_finite) {
      client_->set_target_position(position_command_);
    }
  }
  return hardware_interface::return_type::OK;
}

}  // namespace erd_iiwa

PLUGINLIB_EXPORT_CLASS(erd_iiwa::FriPositionSystem, hardware_interface::SystemInterface)
