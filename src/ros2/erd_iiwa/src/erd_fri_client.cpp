// Copyright 2026 elastic_robot_sim contributors
// SPDX-License-Identifier: Apache-2.0
// See erd_fri_client.hpp for why this subclasses LBRClient directly.
#include "erd_iiwa/erd_fri_client.hpp"

#include <cstring>
#include <limits>
#include "friClientData.h"

namespace erd_iiwa
{

KUKA::FRI::ClientData * ErdFriClientImpl::createData()
{
  sdk_data_ = KUKA::FRI::LBRClient::createData();
  return sdk_data_;
}

void ErdFriClientImpl::onStateChange(KUKA::FRI::ESessionState, KUKA::FRI::ESessionState new_state)
{
  current_state_ = new_state;
}

void ErdFriClientImpl::capture_sample()
{
  const auto & state = robotState();
  sample_.valid = true;
  sample_.sequence_counter = sdk_data_->monitoringMsg.header.sequenceCounter;
  std::memcpy(
    sample_.measured_position.data(), state.getMeasuredJointPosition(),
      kNumJoints * sizeof(double));
  std::memcpy(
    sample_.commanded_position.data(), state.getCommandedJointPosition(),
      kNumJoints * sizeof(double));
  // The real controller sends IPO positions only in the commanding states;
  // in MONITORING_* LBRState::getIpoJointPosition() throws FRIException (not a
  // std::exception), which controller_manager reports as "Unknown exception
  // thrown during read" and deactivates the hardware. Our emulator always sent
  // them, so L2 never hit this (found on the real iiwa, 2026-10-08).
  if (sdk_data_->monitoringMsg.ipoData.has_jointPosition) {
    std::memcpy(sample_.ipo_position.data(), state.getIpoJointPosition(),
        kNumJoints * sizeof(double));
  } else {
    sample_.ipo_position.fill(std::numeric_limits<double>::quiet_NaN());
  }
  std::memcpy(
    sample_.measured_torque.data(), state.getMeasuredTorque(), kNumJoints * sizeof(double));
  std::memcpy(
    sample_.commanded_torque.data(), state.getCommandedTorque(), kNumJoints * sizeof(double));
  std::memcpy(
    sample_.external_torque.data(), state.getExternalTorque(), kNumJoints * sizeof(double));
  sample_.sample_time = state.getSampleTime();
  sample_.time_sec = state.getTimestampSec();
  sample_.time_nsec = state.getTimestampNanoSec();
  sample_.session_state = static_cast<int>(state.getSessionState());
  sample_.connection_quality = static_cast<int>(state.getConnectionQuality());
  sample_.command_mode = static_cast<int>(state.getClientCommandMode());
  sample_.safety_state = static_cast<int>(state.getSafetyState());
  sample_.operation_mode = static_cast<int>(state.getOperationMode());
  sample_.drive_state = static_cast<int>(state.getDriveState());
  sample_.tracking_performance = state.getTrackingPerformance();
}

void ErdFriClientImpl::monitor()
{
  capture_sample();
}

void ErdFriClientImpl::waitForCommand()
{
  capture_sample();
  robotCommand().setJointPosition(robotState().getIpoJointPosition());
}

void ErdFriClientImpl::command()
{
  capture_sample();
  if (target_valid_) {
    robotCommand().setJointPosition(target_position_.data());
  } else {
    robotCommand().setJointPosition(robotState().getIpoJointPosition());
  }
  target_valid_ = false;
}

void ErdFriClientImpl::set_target_position(const std::array<double, kNumJoints> & position)
{
  target_position_ = position;
  target_valid_ = true;
}

ErdFriClient::ErdFriClient()
// Bound a lost peer without treating ordinary non-RT scheduling jitter as loss.
// ERROR-to-controller-deactivation is separately bounded by controller_manager.
: connection_(20), app_(connection_, client_) {}

bool ErdFriClient::connect(int port, const char * remote_host)
{
  return app_.connect(port, remote_host);
}

void ErdFriClient::disconnect()
{
  app_.disconnect();
}

bool ErdFriClient::step()
{
  return app_.step();
}

void ErdFriClient::set_target_position(const std::array<double, kNumJoints> & position)
{
  client_.set_target_position(position);
}

}  // namespace erd_iiwa
