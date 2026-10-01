// Copyright 2026 elastic_robot_sim contributors
// SPDX-License-Identifier: Apache-2.0
//
// `LBRState`/`LBRCommand`'s message-wiring members (`_message`, `_cmdMessage`,
// `_monMessage`, the `LBR{MONITOR,COMMAND}MESSAGEID` constants) are
// `protected` and additionally `friend`-scoped to the literal class name
// `IRDFClient` in the iiwa_ros2-bundled FRI SDK headers -- so IRDFClient's
// own composition-based wiring (RR_01 S1.2) cannot be reused by another
// class without patching those headers, which RR_01 S3.2 forbids ("no
// upstream edits"). The SDK-idiomatic alternative -- and what every other
// FRI driver (LBR-Stack included) actually does -- is to subclass
// `KUKA::FRI::LBRClient` directly: its protected `_robotState`/`_robotCommand`
// are then ordinary inherited-protected members, and `LBRClient::createData()`
// does the message wiring FriPositionSystem would otherwise have needed
// friendship for. `KUKA::FRI::ClientApplication` drives one connect/step/
// disconnect cycle exactly as IRDFClient's own `updateFromRobot`+`updateToRobot`
// pair did, just as a single call (`step()`) instead of two; see the .cpp for
// how `read()`/`write()` map onto that.
#pragma once

#include <array>
#include <cstddef>  // NULL, used by friClientApplication.h's default argument
#include <cstdint>

#include "friClientApplication.h"
#include "friLBRClient.h"
#include "friUdpConnection.h"

namespace erd_iiwa
{

constexpr int kNumJoints = 7;

/// One FRI cycle's raw state, exactly as the monitoring message carries it.
struct ErdFriSample
{
  bool valid{false};
  std::array<double, kNumJoints> measured_position{};
  std::array<double, kNumJoints> commanded_position{};
  std::array<double, kNumJoints> ipo_position{};
  std::array<double, kNumJoints> measured_torque{};      // raw, unfiltered
  std::array<double, kNumJoints> commanded_torque{};
  std::array<double, kNumJoints> external_torque{};
  double sample_time{0.0};
  uint32_t sequence_counter{0};
  uint32_t time_sec{0};
  uint32_t time_nsec{0};
  int session_state{0};
  int connection_quality{0};
  int command_mode{0};
  int safety_state{0};
  int operation_mode{0};
  int drive_state{0};
  double tracking_performance{0.0};
};

/// `KUKA::FRI::LBRClient` subclass capturing the full monitoring message into
/// an `ErdFriSample` and applying `target_position` in COMMANDING_ACTIVE,
/// mirroring the IPO position otherwise (RR_01 S3.3 "Command" rule) -- the
/// same dispatch IRDFClient's `monitor`/`waitForCommand`/`command` implement,
/// just without needing its friendship.
class ErdFriClientImpl : public KUKA::FRI::LBRClient
{
public:
  void monitor() override;
  void waitForCommand() override;
  void command() override;
  void onStateChange(
    KUKA::FRI::ESessionState old_state, KUKA::FRI::ESessionState new_state) override;

  void set_target_position(const std::array<double, kNumJoints> & position);
  const ErdFriSample & sample() const {return sample_;}
  KUKA::FRI::ESessionState current_state() const {return current_state_;}

protected:
  KUKA::FRI::ClientData * createData() override;

private:
  KUKA::FRI::ClientData * sdk_data_{nullptr};  // owned by ClientApplication
  ErdFriSample sample_;
  KUKA::FRI::ESessionState current_state_{KUKA::FRI::IDLE};
  std::array<double, kNumJoints> target_position_{};
  bool target_valid_{false};

  void capture_sample();
};

/// Thin RAII wrapper over `UdpConnection` + `ErdFriClientImpl` +
/// `ClientApplication` (RR_01 S3.3): `read()` calls `step()` (one blocking
/// receive, dispatch, and send of the *previous* cycle's target -- a FRI
/// packet's worth of latency, same as every synchronous FRI driver);
/// `write()` stages the new target for the next `step()`.
class ErdFriClient
{
public:
  ErdFriClient();

  bool connect(int port, const char * remote_host);
  void disconnect();
  bool step();

  void set_target_position(const std::array<double, kNumJoints> & position);
  const ErdFriSample & sample() const {return client_.sample();}
  KUKA::FRI::ESessionState current_state() const {return client_.current_state();}

private:
  KUKA::FRI::UdpConnection connection_;
  ErdFriClientImpl client_;
  KUKA::FRI::ClientApplication app_;
};

}  // namespace erd_iiwa
