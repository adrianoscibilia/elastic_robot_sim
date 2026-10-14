// Copyright 2026 elastic_robot_sim contributors
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace erd_iiwa
{
namespace detail
{

/// Re-times the trajectory controller's position target onto the robot's own
/// sample clock (found on the real iiwa, 2026-10-12, run `iiwa_real_01`).
///
/// The JTC samples its spline at the controller manager's wall-clock time
/// `t_k`, i.e. when our blocking `read()` returned with robot packet `k`.
/// The robot stamps that packet with its own clock `tau_k`, exactly 1 ms
/// apart, but `t_k` arrives with sub-millisecond wake-up jitter. So the JTC
/// targets advance by `v * (t_k - t_{k-1})`, uneven by up to +-40 % per
/// cycle; in FRI POSITION mode the robot follows every one of those steps
/// (audible buzz on A4, KUKA tracking performance ~0.3, measured torque 30x
/// noisier than at rest).
///
/// With `L_k = (t_k - t_0) - (tau_k - tau_0)` the packet's apparent latency
/// and `Lbar` its slow running mean, the command is the JTC's trajectory
/// evaluated at the robot tick plus the MEAN latency instead of the actual one:
///
///     v_k = (p_k - p_{k-1}) / (t_k - t_{k-1})      (exact for a sampled spline)
///     c_k = p_k - v_k * clamp(L_k - Lbar, +-kMaxShift)
///
/// Successive commands then differ by `v * T` (the robot's own period),
/// whatever the wake-up jitter; on a held target (`v = 0`) `c_k = p_k`
/// exactly; the deviation from the JTC target is `v * |L_k - Lbar|` (a few
/// tens of microradians at 0.1 rad/s). `Lbar` follows the slow drift between
/// the two clocks with a ~1 s time constant.
///
/// `reset()` makes the next `step()` pass the target through unchanged: call
/// it whenever the target becomes invalid or the session leaves
/// COMMANDING_ACTIVE, so a new activation starts exactly at the JTC's output.
template <std::size_t N>
class CommandRetimer
{
public:
  /// Per-cycle weight of the running latency mean (~1 s at 1 kHz).
  static constexpr double kLatencyFilter = 1e-3;
  /// Largest latency deviation compensated [s]; beyond it (a stall) the
  /// target is shifted by this much only.
  static constexpr double kMaxShift = 3e-3;

  void reset() {valid_ = false;}
  bool valid() const {return valid_;}

  /// `target`: the JTC position command this cycle; `wall_ns`: steady-clock
  /// time at which this cycle's robot packet was read; `robot_ns`: that
  /// packet's robot timestamp. Returns the command to send.
  const std::array<double, N> & step(
    const std::array<double, N> & target, std::int64_t wall_ns, std::int64_t robot_ns)
  {
    if (!valid_) {
      command_ = target;
      last_target_ = target;
      velocity_.fill(0.0);
      wall0_ = last_wall_ = wall_ns;
      robot0_ = robot_ns;
      mean_latency_ = 0.0;
      valid_ = true;
      return command_;
    }
    const double dt = 1e-9 * static_cast<double>(wall_ns - last_wall_);
    const double latency = 1e-9 * static_cast<double>((wall_ns - wall0_) - (robot_ns - robot0_));
    mean_latency_ += kLatencyFilter * (latency - mean_latency_);
    const double shift = std::clamp(latency - mean_latency_, -kMaxShift, kMaxShift);
    // A back-to-back cycle (queued packets after a stall) gives no usable
    // velocity estimate: keep the previous one instead of dividing by ~0.
    const bool fresh_velocity = dt > 1e-4;
    for (std::size_t i = 0; i < N; ++i) {
      if (fresh_velocity) {
        velocity_[i] = (target[i] - last_target_[i]) / dt;
      }
      command_[i] = target[i] - velocity_[i] * shift;
    }
    last_target_ = target;
    last_wall_ = wall_ns;
    return command_;
  }

private:
  bool valid_{false};
  std::array<double, N> command_{};
  std::array<double, N> last_target_{};
  std::array<double, N> velocity_{};
  std::int64_t wall0_{0};
  std::int64_t last_wall_{0};
  std::int64_t robot0_{0};
  double mean_latency_{0.0};
};

}  // namespace detail
}  // namespace erd_iiwa
