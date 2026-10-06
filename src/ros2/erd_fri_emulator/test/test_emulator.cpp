// SPDX-License-Identifier: Apache-2.0
// T1.9: wire-format compatibility against the real FRI SDK client-side codec
// classes (no friend restriction on MonitoringMessageDecoder/
// CommandMessageEncoder, unlike LBRState/LBRCommand -- see
// robot_side_messages.hpp), the session state machine, and a live UDP
// loopback round trip.
#include <gtest/gtest.h>

#include <arpa/inet.h>
#include <atomic>
#include <cstring>
#include <thread>
#include <unistd.h>

#include "erd_fri_emulator/emulator.hpp"
#include "erd_fri_emulator/kinematic_plant.hpp"
#include "erd_fri_emulator/robot_side_messages.hpp"
#include "friCommandMessageEncoder.h"
#include "friMonitoringMessageDecoder.h"

namespace
{
std::string test_urdf_path()
{
  const char * env = std::getenv("ERD_TEST_URDF");
  if (env) {
    return env;
  }
  return ERD_TEST_URDF_DEFAULT;  // set by CMakeLists.txt from the repository path
}
}  // namespace

TEST(KinematicPlant, CommandedTorqueNeverEqualsMeasuredTorqueAtRest)
{
  // Regression for a real finding (T1.9, live testing against erd_iiwa's
  // actual FriPositionSystem): at rest (dq=ddq=0), rotor_inertia*ddq and the
  // friction term both vanish, so without an *independent* noise draw on
  // the commanded-torque path, `commanded_torque` was bit-identical to
  // `measured_torque` -- exactly the I-9 defect pattern RR_01's own
  // `tau != ft` check exists to catch.
  erd_fri_emulator::KinematicPlant plant(test_urdf_path(), 0.001, 42);
  erd_fri_emulator::JointArray zero{};
  bool saw_difference = false;
  for (int i = 0; i < 20; ++i) {
    auto sample = plant.step(zero);
    for (int j = 0; j < plant.n_joints(); ++j) {
      if (std::abs(sample.measured_torque[j] - sample.commanded_torque[j]) > 1e-12) {
        saw_difference = true;
      }
    }
  }
  EXPECT_TRUE(saw_difference);
}

TEST(RobotSideMessages, MonitoringEncodeDecodesWithRealSdkDecoder)
{
  FRIMonitoringMessage message{};
  erd_fri_emulator::MonitoringMessageEncoder encoder(&message);
  auto * position = static_cast<tRepeatedDoubleArguments *>(
    message.monitorData.measuredJointPosition.value.arg);
  auto * torque = static_cast<tRepeatedDoubleArguments *>(message.monitorData.measuredTorque.value.arg);
  for (int i = 0; i < 7; ++i) {
    position->value[i] = 0.1 * (i + 1);
    torque->value[i] = -2.0 * (i + 1);
  }
  message.connectionInfo.sessionState = FRISessionState_COMMANDING_ACTIVE;
  message.header.sequenceCounter = 42;

  char buffer[erd_fri_emulator::kFriMonitorMsgMaxSize];
  int size = 0;
  ASSERT_TRUE(encoder.encode(buffer, size));

  FRIMonitoringMessage decoded{};
  KUKA::FRI::MonitoringMessageDecoder sdk_decoder(&decoded, 7);
  ASSERT_TRUE(sdk_decoder.decode(buffer, size));

  EXPECT_EQ(decoded.header.sequenceCounter, 42u);
  EXPECT_EQ(decoded.connectionInfo.sessionState, FRISessionState_COMMANDING_ACTIVE);
  auto * decoded_position = static_cast<tRepeatedDoubleArguments *>(
    decoded.monitorData.measuredJointPosition.value.arg);
  auto * decoded_torque = static_cast<tRepeatedDoubleArguments *>(
    decoded.monitorData.measuredTorque.value.arg);
  for (int i = 0; i < 7; ++i) {
    EXPECT_NEAR(decoded_position->value[i], 0.1 * (i + 1), 1e-9);
    EXPECT_NEAR(decoded_torque->value[i], -2.0 * (i + 1), 1e-9);
  }
}

TEST(RobotSideMessages, CommandDecodesRealSdkEncoderOutput)
{
  FRICommandMessage message{};
  KUKA::FRI::CommandMessageEncoder sdk_encoder(&message, 7);
  auto * position = static_cast<tRepeatedDoubleArguments *>(message.commandData.jointPosition.value.arg);
  for (int i = 0; i < 7; ++i) {
    position->value[i] = 0.05 * (i - 3);
  }
  message.commandData.has_jointPosition = true;
  message.has_commandData = true;
  message.header.sequenceCounter = 7;

  char buffer[erd_fri_emulator::kFriCommandMsgMaxSize];
  int size = 0;
  ASSERT_TRUE(sdk_encoder.encode(buffer, size));

  FRICommandMessage decoded{};
  erd_fri_emulator::CommandMessageDecoder decoder(&decoded);
  ASSERT_TRUE(decoder.decode(buffer, size));

  EXPECT_EQ(decoded.header.sequenceCounter, 7u);
  ASSERT_TRUE(decoded.has_commandData);
  ASSERT_TRUE(decoded.commandData.has_jointPosition);
  auto * decoded_position = static_cast<tRepeatedDoubleArguments *>(
    decoded.commandData.jointPosition.value.arg);
  for (int i = 0; i < 7; ++i) {
    EXPECT_NEAR(decoded_position->value[i], 0.05 * (i - 3), 1e-9);
  }
}

class LiveLoopback : public ::testing::Test
{
protected:
  void SetUp() override
  {
    erd_fri_emulator::EmulatorOptions options;
    options.port = 32011;  // fixed test-only port, unlikely to collide
    options.urdf_path = test_urdf_path();
    options.settle_cycles = 5;
    options.send_period_s = 0.001;
    emulator_ = std::make_unique<erd_fri_emulator::FriEmulator>(options);
    ASSERT_TRUE(emulator_->open());
    keep_running_.store(true);
    thread_ = std::thread([this] { emulator_->run(keep_running_); });

    client_fd_ = socket(AF_INET, SOCK_DGRAM, 0);
    sockaddr_in local{};
    local.sin_family = AF_INET;
    local.sin_addr.s_addr = htonl(INADDR_ANY);
    local.sin_port = 0;  // ephemeral -- the test client does not need the
                        // real setup's "same port both sides" convention
                        // (that only matters for two different hosts).
    ASSERT_EQ(bind(client_fd_, reinterpret_cast<sockaddr *>(&local), sizeof(local)), 0);
    server_addr_.sin_family = AF_INET;
    server_addr_.sin_port = htons(static_cast<uint16_t>(options.port));
    inet_pton(AF_INET, "127.0.0.1", &server_addr_.sin_addr);

    // The decoder must outlive every `receive_monitoring()` call: its
    // constructor wires `monitoring_.*.value.arg` to buffers owned by the
    // decoder itself (`map_repeatedDouble`), freed in its destructor. A
    // decoder recreated per call left `monitoring_`'s pointers dangling
    // the moment the temporary went out of scope -- caught by this test.
    monitoring_decoder_ = std::make_unique<KUKA::FRI::MonitoringMessageDecoder>(&monitoring_, 7);
  }

  void TearDown() override
  {
    keep_running_.store(false);
    if (thread_.joinable()) {
      thread_.join();
    }
    if (client_fd_ >= 0) {
      close(client_fd_);
    }
  }

  void send_command(double target)
  {
    FRICommandMessage message{};
    KUKA::FRI::CommandMessageEncoder encoder(&message, 7);
    auto * position = static_cast<tRepeatedDoubleArguments *>(message.commandData.jointPosition.value.arg);
    for (int i = 0; i < 7; ++i) {
      position->value[i] = target;
    }
    message.commandData.has_jointPosition = true;
    message.has_commandData = true;
    message.header.sequenceCounter = send_sequence_++;
    char buffer[erd_fri_emulator::kFriCommandMsgMaxSize];
    int size = 0;
    encoder.encode(buffer, size);
    sendto(client_fd_, buffer, static_cast<size_t>(size), 0,
          reinterpret_cast<sockaddr *>(&server_addr_), sizeof(server_addr_));
  }

  /// Decodes into the fixture's own `monitoring_`, valid until the next call
  /// (see the dangling-pointer note in `SetUp`).
  bool receive_monitoring(int timeout_ms = 50)
  {
    timeval tv{};
    tv.tv_sec = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;
    setsockopt(client_fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    char buffer[erd_fri_emulator::kFriMonitorMsgMaxSize];
    ssize_t size = recvfrom(client_fd_, buffer, sizeof(buffer), 0, nullptr, nullptr);
    if (size <= 0) {
      return false;
    }
    return monitoring_decoder_->decode(buffer, static_cast<int>(size));
  }

  std::unique_ptr<erd_fri_emulator::FriEmulator> emulator_;
  std::atomic<bool> keep_running_{true};
  std::thread thread_;
  int client_fd_{-1};
  sockaddr_in server_addr_{};
  uint32_t send_sequence_{0};
  FRIMonitoringMessage monitoring_{};
  std::unique_ptr<KUKA::FRI::MonitoringMessageDecoder> monitoring_decoder_;
};

TEST_F(LiveLoopback, ReachesCommandingActiveAfterSustainedCommands)
{
  bool reached_active = false;
  // 5 settle cycles per state x 3 transitions = 15 cycles minimum; give it
  // generous headroom for scheduling jitter on a non-RT kernel.
  for (int i = 0; i < 200; ++i) {
    send_command(0.0);
    if (!receive_monitoring(20)) {
      continue;
    }
    if (monitoring_.connectionInfo.sessionState == FRISessionState_COMMANDING_ACTIVE) {
      reached_active = true;
      break;
    }
  }
  EXPECT_TRUE(reached_active);
}

TEST_F(LiveLoopback, ReflectsSequenceCounterAndAppliesCommandedPosition)
{
  // Drive to COMMANDING_ACTIVE first.
  for (int i = 0; i < 100; ++i) {
    send_command(0.0);
    if (receive_monitoring(20) &&
        monitoring_.connectionInfo.sessionState == FRISessionState_COMMANDING_ACTIVE) {
      break;
    }
  }
  ASSERT_EQ(monitoring_.connectionInfo.sessionState, FRISessionState_COMMANDING_ACTIVE);

  const uint32_t sent_sequence = send_sequence_;
  send_command(0.3);
  bool saw_reflected = false;
  double last_commanded = 0.0;
  for (int i = 0; i < 50; ++i) {
    if (!receive_monitoring(20)) {
      continue;
    }
    if (monitoring_.header.reflectedSequenceCounter == sent_sequence) {
      saw_reflected = true;
    }
    auto * commanded = static_cast<tRepeatedDoubleArguments *>(
      monitoring_.monitorData.commandedJointPosition.value.arg);
    last_commanded = commanded->value[0];
    send_command(0.3);
  }
  EXPECT_TRUE(saw_reflected);
  EXPECT_NEAR(last_commanded, 0.3, 1e-6);
}

// RR_04 B-8: --drop-every N deterministically drops every Nth outgoing
// monitoring packet (a reproducible alternative to --drop-probability), and
// every drop is counted in stats(), never silently lost.
TEST(FriEmulatorDropEvery, DropsEveryNthPacketAndCountsIt)
{
  erd_fri_emulator::EmulatorOptions options;
  options.port = 32012;  // a different fixed test-only port from LiveLoopback's 32011
  options.urdf_path = test_urdf_path();
  options.settle_cycles = 2;
  options.send_period_s = 0.001;
  options.drop_every = 4;
  erd_fri_emulator::FriEmulator emulator(options);
  ASSERT_TRUE(emulator.open());
  std::atomic<bool> keep_running{true};
  std::thread thread([&] { emulator.run(keep_running); });

  int client_fd = socket(AF_INET, SOCK_DGRAM, 0);
  sockaddr_in local{};
  local.sin_family = AF_INET;
  local.sin_addr.s_addr = htonl(INADDR_ANY);
  local.sin_port = 0;
  ASSERT_EQ(bind(client_fd, reinterpret_cast<sockaddr *>(&local), sizeof(local)), 0);
  sockaddr_in server_addr{};
  server_addr.sin_family = AF_INET;
  server_addr.sin_port = htons(static_cast<uint16_t>(options.port));
  inet_pton(AF_INET, "127.0.0.1", &server_addr.sin_addr);

  int received = 0;
  timeval tv{};
  tv.tv_sec = 0;
  tv.tv_usec = 20000;
  setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
  for (int i = 0; i < 200 && received < 20; ++i) {
    FRICommandMessage message{};
    KUKA::FRI::CommandMessageEncoder encoder(&message, 7);
    auto * position = static_cast<tRepeatedDoubleArguments *>(message.commandData.jointPosition.value.arg);
    for (int j = 0; j < 7; ++j) {
      position->value[j] = 0.0;
    }
    message.commandData.has_jointPosition = true;
    message.has_commandData = true;
    message.header.sequenceCounter = static_cast<uint32_t>(i);
    char command_buffer[erd_fri_emulator::kFriCommandMsgMaxSize];
    int command_size = 0;
    encoder.encode(command_buffer, command_size);
    sendto(client_fd, command_buffer, static_cast<size_t>(command_size), 0,
          reinterpret_cast<sockaddr *>(&server_addr), sizeof(server_addr));

    char buffer[erd_fri_emulator::kFriMonitorMsgMaxSize];
    ssize_t size = recvfrom(client_fd, buffer, sizeof(buffer), 0, nullptr, nullptr);
    if (size > 0) {
      received++;
    }
  }
  keep_running.store(false);
  thread.join();
  close(client_fd);

  const auto & stats = emulator.stats();
  EXPECT_GT(stats.packets_dropped_injected, 0u);
  // Every 4th of the cycles actually run should have been dropped, +/- one
  // for whichever cycle the loop happened to stop on.
  double expected_fraction = 1.0 / static_cast<double>(options.drop_every);
  double actual_fraction = static_cast<double>(stats.packets_dropped_injected) / static_cast<double>(stats.cycles);
  EXPECT_NEAR(actual_fraction, expected_fraction, 0.05);
}

int main(int argc, char ** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
