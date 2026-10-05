// Copyright 2026 elastic_robot_sim contributors
// SPDX-License-Identifier: Apache-2.0
#include <atomic>
#include <chrono>
#include <limits>
#include <thread>
#include <utility>
#include <vector>

#include "control_msgs/msg/dynamic_joint_state.hpp"
#include "controller_interface/controller_interface.hpp"
#include "pluginlib/class_list_macros.hpp"

namespace erd_iiwa
{
// The standard JSB's realtime publisher may skip a sample when its worker
// owns the message. Recording requires every hardware read, so retain a
// bounded FIFO and serialize in a separate thread. The monitor still uses
// depth 1 and the original timestamp; queuing never makes old data fresh.
class BufferedStateBroadcaster : public controller_interface::ControllerInterface
{
public:
  ~BufferedStateBroadcaster() override {stop();}

  controller_interface::InterfaceConfiguration command_interface_configuration() const override
  {
    return {controller_interface::interface_configuration_type::NONE, {}};
  }
  controller_interface::InterfaceConfiguration state_interface_configuration() const override
  {
    return {controller_interface::interface_configuration_type::ALL, {}};
  }
  CallbackReturn on_init() override {return CallbackReturn::SUCCESS;}
  CallbackReturn on_configure(const rclcpp_lifecycle::State &) override
  {
    publisher_ = get_node()->create_publisher<control_msgs::msg::DynamicJointState>(
      "/dynamic_joint_states", rclcpp::QoS(2000).reliable());
    return CallbackReturn::SUCCESS;
  }
  CallbackReturn on_activate(const rclcpp_lifecycle::State &) override
  {
    control_msgs::msg::DynamicJointState message;
    indices_.clear();
    for (const auto & state : state_interfaces_) {
      const auto name = state.get_prefix_name();
      size_t joint = 0;
      while (joint < message.joint_names.size() && message.joint_names[joint] != name) {
        ++joint;
      }
      if (joint == message.joint_names.size()) {
        message.joint_names.push_back(name);
        message.interface_values.emplace_back();
      }
      auto & values = message.interface_values[joint];
      indices_.emplace_back(joint, values.values.size());
      values.interface_names.push_back(state.get_interface_name());
      values.values.push_back(0.0);
    }
    buffer_.assign(kCapacity, message);
    read_.store(0);
    write_.store(0);
    running_.store(true);
    worker_ = std::thread([this]() {
          while (running_.load() || read_.load() != write_.load()) {
            const auto index = read_.load(std::memory_order_relaxed);
            if (index == write_.load(std::memory_order_acquire)) {
              std::this_thread::sleep_for(std::chrono::microseconds(100));
              continue;
            }
            try {
              publisher_->publish(buffer_[index % kCapacity]);
            } catch (const std::exception & error) {
              if (rclcpp::ok()) {
                RCLCPP_ERROR(get_node()->get_logger(), "Telemetry publication failed: %s",
                error.what());
              }
              running_.store(false);
              break;
            }
            read_.store(index + 1, std::memory_order_release);
          }
    });
    return CallbackReturn::SUCCESS;
  }
  CallbackReturn on_deactivate(const rclcpp_lifecycle::State &) override
  {
    stop();
    return CallbackReturn::SUCCESS;
  }
  controller_interface::return_type update(
    const rclcpp::Time & time, const rclcpp::Duration &) override
  {
    const auto index = write_.load(std::memory_order_relaxed);
    if (!running_.load() || index - read_.load(std::memory_order_acquire) >= kCapacity) {
      RCLCPP_ERROR(get_node()->get_logger(),
          "Recording telemetry FIFO overflow; refusing silent loss");
      return controller_interface::return_type::ERROR;
    }
    auto & message = buffer_[index % kCapacity];
    message.header.stamp = time;
    for (size_t i = 0; i < indices_.size(); ++i) {
      double value = std::numeric_limits<double>::quiet_NaN();
      state_interfaces_[i].get_value(value);
      message.interface_values[indices_[i].first].values[indices_[i].second] = value;
    }
    write_.store(index + 1, std::memory_order_release);
    return controller_interface::return_type::OK;
  }

private:
  void stop()
  {
    running_.store(false);
    if (worker_.joinable()) {
      worker_.join();
    }
  }
  static constexpr size_t kCapacity = 8192;
  std::vector<control_msgs::msg::DynamicJointState> buffer_;
  std::vector<std::pair<size_t, size_t>> indices_;
  std::atomic<size_t> read_{0};
  std::atomic<size_t> write_{0};
  std::atomic<bool> running_{false};
  std::thread worker_;
  rclcpp::Publisher<control_msgs::msg::DynamicJointState>::SharedPtr publisher_;
};
}  // namespace erd_iiwa

PLUGINLIB_EXPORT_CLASS(erd_iiwa::BufferedStateBroadcaster,
  controller_interface::ControllerInterface)
