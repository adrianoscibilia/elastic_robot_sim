// SPDX-License-Identifier: Apache-2.0
#include <atomic>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "erd_fri_emulator/emulator.hpp"

namespace
{
std::atomic<bool> g_keep_running{true};
void on_signal(int) { g_keep_running.store(false); }
volatile std::sig_atomic_t g_drops_enabled = 1;
void on_drop_signal(int signum) { g_drops_enabled = signum == SIGUSR1 ? 1 : 0; }
}  // namespace

int main(int argc, char ** argv)
{
  erd_fri_emulator::EmulatorOptions options;
  std::string urdf_path;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "erd_fri_emulator: missing value for %s\n", arg.c_str());
        std::exit(2);
      }
      return argv[++i];
    };
    if (arg == "--port") {
      options.port = std::stoi(next());
    } else if (arg == "--bind-address") {
      options.bind_address = next();
    } else if (arg == "--client-address") {
      options.client_address = next();
    } else if (arg == "--client-port") {
      options.client_port = std::stoi(next());
    } else if (arg == "--urdf") {
      urdf_path = next();
    } else if (arg == "--no-auto-activate") {
      options.auto_activate = false;
    } else if (arg == "--settle-cycles") {
      options.settle_cycles = std::stoi(next());
    } else if (arg == "--drop-probability") {
      options.drop_probability = std::stod(next());
    } else if (arg == "--drop-on-signal") {
      g_drops_enabled = 0;
      options.drops_enabled = &g_drops_enabled;
    } else if (arg == "--drop-every") {
      options.drop_every = std::stoi(next());
    } else if (arg == "--jitter-us") {
      options.jitter_us = std::stod(next());
    } else if (arg == "--send-period-s") {
      options.send_period_s = std::stod(next());
    } else if (arg == "--initial-positions") {
      // Comma-separated, one value per joint in A1..A7 order (rad).
      const std::string text = next();
      erd_fri_emulator::JointArray q{};
      size_t start = 0;
      int count = 0;
      while (start <= text.size() && count < erd_fri_emulator::kNumJoints) {
        const size_t comma = text.find(',', start);
        q[count++] = std::stod(text.substr(start, comma - start));
        if (comma == std::string::npos) {
          start = text.size() + 1;
          break;
        }
        start = comma + 1;
      }
      if (count != erd_fri_emulator::kNumJoints || start <= text.size()) {
        std::fprintf(stderr, "erd_fri_emulator: --initial-positions needs exactly %d values\n",
          erd_fri_emulator::kNumJoints);
        return 2;
      }
      options.initial_position = q;
    } else if (arg == "--seed") {
      options.seed = static_cast<unsigned>(std::stoul(next()));
    } else if (arg == "--help") {
      std::printf(
        "erd_fri_emulator --urdf <path> [--port 30200] [--no-auto-activate] "
        "[--settle-cycles 10] [--drop-probability 0.0] [--drop-every 0] [--jitter-us 0.0] "
        "[--send-period-s 0.001] [--seed 0] [--drop-on-signal] "
        "[--initial-positions a1,...,a7]\n");
      return 0;
    } else {
      std::fprintf(stderr, "erd_fri_emulator: unknown argument %s\n", arg.c_str());
      return 2;
    }
  }
  if (urdf_path.empty()) {
    std::fprintf(stderr, "erd_fri_emulator: --urdf <path> is required (RR_01 S3.5: the bare asset URDF)\n");
    return 2;
  }
  options.urdf_path = urdf_path;

  erd_fri_emulator::FriEmulator emulator(options);
  if (!emulator.open()) {
    return 1;
  }
  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);
  std::signal(SIGUSR1, on_drop_signal);
  std::signal(SIGUSR2, on_drop_signal);

  std::printf("erd_fri_emulator: listening on UDP :%d, urdf=%s\n", options.port, urdf_path.c_str());
  emulator.run(g_keep_running);

  const auto & stats = emulator.stats();
  std::printf(
    "erd_fri_emulator: stopped. cycles=%llu received=%llu decode_failures=%llu dropped_injected=%llu\n",
    static_cast<unsigned long long>(stats.cycles), static_cast<unsigned long long>(stats.packets_received),
    static_cast<unsigned long long>(stats.decode_failures),
    static_cast<unsigned long long>(stats.packets_dropped_injected));
  return 0;
}
