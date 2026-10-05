// SPDX-License-Identifier: Apache-2.0
#include "erd_fri_emulator/emulator.hpp"

#include <arpa/inet.h>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <time.h>
#include <unistd.h>

namespace erd_fri_emulator
{

namespace
{
timespec add_seconds(timespec t, double seconds)
{
  long whole = static_cast<long>(seconds);
  double frac = seconds - static_cast<double>(whole);
  t.tv_sec += whole;
  t.tv_nsec += static_cast<long>(frac * 1e9);
  if (t.tv_nsec >= 1000000000L) {
    t.tv_nsec -= 1000000000L;
    t.tv_sec += 1;
  }
  return t;
}
}  // namespace

FriEmulator::FriEmulator(EmulatorOptions options)
: options_(std::move(options)),
  encoder_(&monitoring_message_),
  decoder_(&command_message_),
  plant_(options_.urdf_path, options_.send_period_s, options_.seed),
  drop_rng_(options_.seed + 1),
  jitter_rng_(options_.seed + 2)
{
}

FriEmulator::~FriEmulator()
{
  if (socket_fd_ >= 0) {
    ::close(socket_fd_);
  }
}

bool FriEmulator::open()
{
  socket_fd_ = ::socket(AF_INET, SOCK_DGRAM, 0);
  if (socket_fd_ < 0) {
    std::fprintf(stderr, "erd_fri_emulator: socket() failed: %s\n", std::strerror(errno));
    return false;
  }
  sockaddr_in address{};
  address.sin_family = AF_INET;
  if (options_.bind_address == "0.0.0.0") {
    address.sin_addr.s_addr = htonl(INADDR_ANY);
  } else if (::inet_pton(AF_INET, options_.bind_address.c_str(), &address.sin_addr) != 1) {
    std::fprintf(stderr, "erd_fri_emulator: invalid --bind-address %s\n", options_.bind_address.c_str());
    return false;
  }
  address.sin_port = htons(static_cast<uint16_t>(options_.port));
  if (::bind(socket_fd_, reinterpret_cast<sockaddr *>(&address), sizeof(address)) < 0) {
    std::fprintf(stderr, "erd_fri_emulator: bind(%d) failed: %s\n", options_.port, std::strerror(errno));
    return false;
  }
  int flags = fcntl(socket_fd_, F_GETFL, 0);
  fcntl(socket_fd_, F_SETFL, flags | O_NONBLOCK);

  if (!options_.client_address.empty()) {
    client_addr_.sin_family = AF_INET;
    client_addr_.sin_port = htons(static_cast<uint16_t>(options_.client_port ? options_.client_port : options_.port));
    if (::inet_pton(AF_INET, options_.client_address.c_str(), &client_addr_.sin_addr) != 1) {
      std::fprintf(stderr, "erd_fri_emulator: invalid --client-address %s\n", options_.client_address.c_str());
      return false;
    }
    client_known_ = true;
  }
  return true;
}

void FriEmulator::poll_incoming()
{
  char buffer[kFriCommandMsgMaxSize];
  sockaddr_in from{};
  socklen_t from_len = sizeof(from);
  for (;;) {
    ssize_t size = ::recvfrom(socket_fd_, buffer, sizeof(buffer), 0,
                              reinterpret_cast<sockaddr *>(&from), &from_len);
    if (size <= 0) {
      break;  // EAGAIN/EWOULDBLOCK: nothing more pending this cycle
    }
    if (!client_known_) {
      client_addr_ = from;
      client_known_ = true;
    }
    stats_.packets_received++;
    if (!decoder_.decode(buffer, static_cast<int>(size))) {
      stats_.decode_failures++;
      continue;
    }
    last_received_sequence_ = command_message_.header.sequenceCounter;
    if (command_message_.has_commandData && command_message_.commandData.has_jointPosition) {
      auto * values = static_cast<tRepeatedDoubleArguments *>(
        command_message_.commandData.jointPosition.value.arg);
      for (int i = 0; i < plant_.n_joints() && i < kNumJoints; ++i) {
        target_position_[i] = values->value[i];
      }
      has_target_ = true;
    }
  }
}

void FriEmulator::advance_state_machine()
{
  cycles_in_state_++;
  switch (state_) {
    case FRISessionState_IDLE:
      if (options_.auto_activate && client_known_) {
        state_ = FRISessionState_MONITORING_WAIT;
        cycles_in_state_ = 0;
      }
      break;
    case FRISessionState_MONITORING_WAIT:
      if (cycles_in_state_ >= options_.settle_cycles) {
        state_ = FRISessionState_MONITORING_READY;
        cycles_in_state_ = 0;
      }
      break;
    case FRISessionState_MONITORING_READY:
      if (options_.auto_activate && cycles_in_state_ >= options_.settle_cycles) {
        state_ = FRISessionState_COMMANDING_WAIT;
        cycles_in_state_ = 0;
      }
      break;
    case FRISessionState_COMMANDING_WAIT:
      if (cycles_in_state_ >= options_.settle_cycles) {
        state_ = FRISessionState_COMMANDING_ACTIVE;
        cycles_in_state_ = 0;
      }
      break;
    case FRISessionState_COMMANDING_ACTIVE:
    default:
      break;
  }
}

void FriEmulator::send_monitoring_message()
{
  if (!client_known_) {
    return;
  }
  const bool drop_by_probability =
    options_.drop_probability > 0.0 && unit_interval_(drop_rng_) < options_.drop_probability;
  // `send_sequence_` is the packet's own outgoing index, so `--drop-every N`
  // drops packets 0, N, 2N, ... deterministically regardless of when the
  // session reached COMMANDING_ACTIVE (RR_04 B-8).
  const bool drop_by_schedule =
    options_.drop_every > 0 && (send_sequence_ % static_cast<uint32_t>(options_.drop_every)) == 0;
  const bool enabled = options_.drops_enabled == nullptr || *options_.drops_enabled != 0;
  const bool drop = enabled && (drop_by_probability || drop_by_schedule);

  // Hold the plant's own current position (mirrors IPO) until a target has
  // ever been received, exactly as the real client mirrors IPO position
  // outside COMMANDING_ACTIVE (RR_01 S3.3); once a target exists, always
  // drive the plant toward it. One `step()` call per cycle.
  JointArray setpoint = has_target_ ? target_position_ : plant_.current_position();
  KinematicPlant::Sample sample = plant_.step(setpoint);

  timespec now{};
  clock_gettime(CLOCK_REALTIME, &now);

  FRIMonitoringMessage * message = encoder_.message();
  message->connectionInfo.sessionState = state_;
  message->connectionInfo.sendPeriod = static_cast<uint32_t>(options_.send_period_s * 1000.0);
  message->header.sequenceCounter = send_sequence_++;
  message->header.reflectedSequenceCounter = last_received_sequence_;
  message->monitorData.timestamp.sec = static_cast<uint32_t>(now.tv_sec);
  message->monitorData.timestamp.nanosec = static_cast<uint32_t>(now.tv_nsec);
  message->robotInfo.controlMode = ControlMode_POSITION_CONTROLMODE;
  message->ipoData.clientCommandMode = ClientCommandMode_POSITION;

  auto * measured_position = static_cast<tRepeatedDoubleArguments *>(
    message->monitorData.measuredJointPosition.value.arg);
  auto * measured_torque = static_cast<tRepeatedDoubleArguments *>(
    message->monitorData.measuredTorque.value.arg);
  auto * commanded_position = static_cast<tRepeatedDoubleArguments *>(
    message->monitorData.commandedJointPosition.value.arg);
  auto * commanded_torque = static_cast<tRepeatedDoubleArguments *>(
    message->monitorData.commandedTorque.value.arg);
  auto * external_torque = static_cast<tRepeatedDoubleArguments *>(
    message->monitorData.externalTorque.value.arg);
  auto * ipo_position = static_cast<tRepeatedDoubleArguments *>(message->ipoData.jointPosition.value.arg);
  auto * drive_state = static_cast<tRepeatedIntArguments *>(message->robotInfo.driveState.arg);

  for (int i = 0; i < plant_.n_joints() && i < kNumJoints; ++i) {
    measured_position->value[i] = sample.measured_position[i];
    measured_torque->value[i] = sample.measured_torque[i];
    commanded_position->value[i] = has_target_ ? target_position_[i] : sample.measured_position[i];
    commanded_torque->value[i] = sample.commanded_torque[i];
    external_torque->value[i] = sample.external_torque[i];
    ipo_position->value[i] = has_target_ ? target_position_[i] : sample.measured_position[i];
    if (state_ == FRISessionState_COMMANDING_ACTIVE) {
      drive_state->value[i] = DriveState_ACTIVE;
    } else if (state_ == FRISessionState_IDLE) {
      drive_state->value[i] = DriveState_OFF;
    } else {
      drive_state->value[i] = DriveState_TRANSITIONING;
    }
  }

  if (drop) {
    stats_.packets_dropped_injected++;
    return;
  }
  char buffer[kFriMonitorMsgMaxSize];
  int size = 0;
  if (encoder_.encode(buffer, size)) {
    ::sendto(socket_fd_, buffer, static_cast<size_t>(size), 0,
            reinterpret_cast<sockaddr *>(&client_addr_), sizeof(client_addr_));
  }
}

void FriEmulator::run(std::atomic<bool> & keep_running)
{
  timespec next{};
  clock_gettime(CLOCK_MONOTONIC, &next);
  while (keep_running.load()) {
    poll_incoming();
    advance_state_machine();
    send_monitoring_message();
    stats_.cycles++;
    double period = options_.send_period_s;
    if (options_.jitter_us > 0.0) {
      period += jitter_unit_(jitter_rng_) * options_.jitter_us * 1e-6;
    }
    next = add_seconds(next, period);
    clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next, nullptr);
  }
}

}  // namespace erd_fri_emulator
