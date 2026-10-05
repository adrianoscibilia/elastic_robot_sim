// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <atomic>
#include <cstdint>
#include <csignal>
#include <netinet/in.h>
#include <random>
#include <string>

#include "erd_fri_emulator/kinematic_plant.hpp"
#include "erd_fri_emulator/robot_side_messages.hpp"

namespace erd_fri_emulator
{

struct EmulatorOptions
{
  int port{30200};
  // "0.0.0.0" (wildcard) for real hardware, where the emulator and the
  // client are different hosts. On a same-machine (L2a) loopback test the
  // real FRI client (KUKA::FRI::UdpConnection::open(), unmodified SDK code)
  // *also* binds its local socket to 0.0.0.0:port -- two wildcard binds to
  // the same port from different processes always conflict. Binding the
  // emulator to a specific loopback address instead (e.g. "127.0.0.2") lets
  // both coexist: Linux's UDP bind conflict check only rejects two sockets
  // whose bound addresses actually overlap for the same port, and
  // wildcard-vs-specific does not (found live testing T1.9's integration
  // with erd_iiwa's real client, "binding port number 30200 failed!").
  std::string bind_address{"0.0.0.0"};
  // The real FRI client only *reacts* to a received monitoring message --
  // KUKA::FRI::ClientApplication::step() decodes, dispatches to the
  // client's callback, and replies; it never sends anything unprompted.
  // A robot-side emulator therefore cannot learn the client's address from
  // an incoming packet (there won't be one until the robot speaks first):
  // it must be told, the same way a real Sunrise/FRI setup is configured
  // with the client's IP (RR_01 S4's `connection.client_ip`). Empty means
  // "learn opportunistically from the first received packet", kept only
  // for test harnesses that send speculatively (T1.9's gtest).
  std::string client_address;
  int client_port{0};  // 0 = same as `port` (the real setup's convention)
  std::string urdf_path;
  bool auto_activate{true};
  int settle_cycles{10};        // cycles spent in each of MONITORING_WAIT/READY/COMMANDING_WAIT
  double drop_probability{0.0}; // fraction of outgoing monitoring packets dropped (RR_01 S3.5)
  // Deterministic alternative to `drop_probability` (RR_04 B-8): every
  // `drop_every`-th outgoing monitoring packet is dropped (0 = disabled).
  // Reproducible drop *timing* is what the drop-policy test needs -- a fixed
  // cycle number invalidates a known segment window, unlike a probabilistic
  // drop whose exact cycle varies run to run even with a fixed seed's
  // aggregate rate.
  int drop_every{0};
  const volatile std::sig_atomic_t * drops_enabled{nullptr};
  // Timing jitter (RR_04 B-8): each cycle's send period is perturbed by a
  // uniform random offset in [-jitter_us, +jitter_us] microseconds, to
  // exercise the drop/gap-detection policy against unevenly spaced samples,
  // not just perfectly periodic ones.
  double jitter_us{0.0};
  double send_period_s{0.001};
  unsigned seed{0};
};

/// The Sunrise/robot side of FRI 1.11 over UDP (RR_01 S3.5). Binds
/// `port`, learns the client's address from its first command packet (UDP,
/// same pattern the real robot/client pair uses -- both sides bind the same
/// port number, see `friUdpConnection.cpp`), and runs the session state
/// machine IDLE -> MONITORING_WAIT -> MONITORING_READY -> COMMANDING_WAIT ->
/// COMMANDING_ACTIVE, sending one `FRIMonitoringMessage` per `send_period_s`
/// and decoding whatever `FRICommandMessage` arrived since the last cycle.
class FriEmulator
{
public:
  explicit FriEmulator(EmulatorOptions options);
  ~FriEmulator();

  /// Binds the UDP socket. Returns false on failure (port in use, etc.).
  bool open();

  /// Blocking main loop at `send_period_s`, until `request_stop()` or a
  /// signal sets the shared atomic flag passed in.
  void run(std::atomic<bool> & keep_running);

  /// Diagnostics, read after `run()` returns (T1.9 acceptance: zero lost
  /// cycles in an idle run; injected drops are counted, not silently lost).
  struct Stats
  {
    uint64_t cycles{0};
    uint64_t packets_received{0};
    uint64_t decode_failures{0};
    uint64_t packets_dropped_injected{0};
  };
  const Stats & stats() const { return stats_; }

private:
  EmulatorOptions options_;
  int socket_fd_{-1};
  sockaddr_in client_addr_{};
  bool client_known_{false};

  FRISessionState state_{FRISessionState_IDLE};
  int cycles_in_state_{0};
  uint32_t send_sequence_{0};
  uint32_t last_received_sequence_{0};

  FRIMonitoringMessage monitoring_message_{};
  FRICommandMessage command_message_{};
  MonitoringMessageEncoder encoder_;
  CommandMessageDecoder decoder_;
  KinematicPlant plant_;
  std::mt19937 drop_rng_;
  std::uniform_real_distribution<double> unit_interval_{0.0, 1.0};
  std::mt19937 jitter_rng_;
  std::uniform_real_distribution<double> jitter_unit_{-1.0, 1.0};

  JointArray target_position_{};
  bool has_target_{false};

  Stats stats_{};

  void poll_incoming();
  void advance_state_machine();
  void send_monitoring_message();
};

}  // namespace erd_fri_emulator
