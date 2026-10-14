// Copyright 2026 elastic_robot_sim contributors
// SPDX-License-Identifier: Apache-2.0
// T1.10: interface list; NaN until the first valid packet; the POSITION-only
// guard's client-side precondition; GPIO fields are all exported.
#include <gtest/gtest.h>

#include <cmath>

#include "erd_iiwa/erd_fri_client.hpp"
#include "erd_iiwa/fri_position_system.hpp"
#include "hardware_interface/hardware_info.hpp"

namespace
{

hardware_interface::HardwareInfo make_info(int n_joints)
{
  hardware_interface::HardwareInfo info;
  for (int i = 0; i < n_joints; ++i) {
    hardware_interface::ComponentInfo joint;
    joint.name = "joint_a" + std::to_string(i + 1);
    joint.command_interfaces.push_back({"position", "", ""});
    for (const char * iface : {"position", "velocity", "effort", "commanded_effort",
        "external_effort", "commanded_position", "ipo_position"})
    {
      joint.state_interfaces.push_back({iface, "", ""});
    }
    info.joints.push_back(joint);
  }
  return info;
}

}  // namespace

TEST(FriPositionSystem, OnInitAcceptsSevenJoints)
{
  erd_iiwa::FriPositionSystem system;
  EXPECT_EQ(system.on_init(make_info(7)),
           hardware_interface::SystemInterface::CallbackReturn::SUCCESS);
}

TEST(FriPositionSystem, OnInitRejectsWrongJointCount)
{
  erd_iiwa::FriPositionSystem system;
  EXPECT_EQ(system.on_init(make_info(6)),
           hardware_interface::SystemInterface::CallbackReturn::ERROR);
}

TEST(FriPositionSystem, ExportsEveryStateAndCommandInterface)
{
  erd_iiwa::FriPositionSystem system;
  ASSERT_EQ(system.on_init(make_info(7)),
           hardware_interface::SystemInterface::CallbackReturn::SUCCESS);
  auto states = system.export_state_interfaces();
  auto commands = system.export_command_interfaces();
  // 7 joints x 7 state interfaces + 12 GPIO "fri" interfaces.
  EXPECT_EQ(states.size(), 7u * 7u + 12u);
  EXPECT_EQ(commands.size(), 7u);
}

TEST(FriPositionSystem, StateIsNanBeforeAnyValidPacket)
{
  erd_iiwa::FriPositionSystem system;
  ASSERT_EQ(system.on_init(make_info(7)),
           hardware_interface::SystemInterface::CallbackReturn::SUCCESS);
  auto states = system.export_state_interfaces();
  for (auto & handle : states) {
    if (handle.get_interface_name() == "position") {
      double value = 0.0;
      ASSERT_TRUE(handle.get_value(value));
      EXPECT_TRUE(std::isnan(value));
    }
  }
}

TEST(ErdFriClient, StepFailsWithoutConnect)
{
  erd_iiwa::ErdFriClient client;
  EXPECT_FALSE(client.step());
}

// RR_04 A-4: before the session has ever reached COMMANDING_ACTIVE, a failed
// step() (IDLE/MONITORING_WAIT) is OK+NaN, as today. Once it has been
// COMMANDING_ACTIVE, a failed step() or falling back to any other state is a
// hardware ERROR, so the controller manager deactivates the JTC rather than
// let it hold a stale/NaN setpoint.
TEST(FriReadReturn, StaysOkWhileNeverActive)
{
  bool has_been_active = false;
  EXPECT_EQ(erd_iiwa::detail::fri_read_return(false, KUKA::FRI::IDLE, has_been_active),
           hardware_interface::return_type::OK);
  EXPECT_FALSE(has_been_active);
  EXPECT_EQ(erd_iiwa::detail::fri_read_return(true, KUKA::FRI::MONITORING_WAIT, has_been_active),
           hardware_interface::return_type::OK);
  EXPECT_FALSE(has_been_active);
  EXPECT_EQ(erd_iiwa::detail::fri_read_return(true, KUKA::FRI::COMMANDING_WAIT, has_been_active),
           hardware_interface::return_type::OK);
  EXPECT_FALSE(has_been_active);
}

TEST(FriReadReturn, ReachingActiveLatchesTheFlag)
{
  bool has_been_active = false;
  EXPECT_EQ(erd_iiwa::detail::fri_read_return(true, KUKA::FRI::COMMANDING_ACTIVE, has_been_active),
           hardware_interface::return_type::OK);
  EXPECT_TRUE(has_been_active);
}

TEST(FriReadReturn, StepFailureAfterActiveIsError)
{
  bool has_been_active = true;
  EXPECT_EQ(erd_iiwa::detail::fri_read_return(false, KUKA::FRI::IDLE, has_been_active),
           hardware_interface::return_type::ERROR);
  EXPECT_TRUE(has_been_active);  // the flag itself doesn't reset; on_activate() does
}

TEST(FriReadReturn, LeavingActiveWithoutAStepFailureIsAlsoError)
{
  // A session can drop straight to MONITORING_WAIT on a still-successful
  // step() -- not just via a decode/receive failure.
  bool has_been_active = true;
  EXPECT_EQ(erd_iiwa::detail::fri_read_return(true, KUKA::FRI::MONITORING_WAIT, has_been_active),
           hardware_interface::return_type::ERROR);
}

TEST(FriReadReturn, StayingActiveIsOk)
{
  bool has_been_active = true;
  EXPECT_EQ(erd_iiwa::detail::fri_read_return(true, KUKA::FRI::COMMANDING_ACTIVE, has_been_active),
           hardware_interface::return_type::OK);
  EXPECT_TRUE(has_been_active);
}

int main(int argc, char ** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

// Real iiwa, 2026-10-12 (run iiwa_real_01): the JTC samples its trajectory
// when the packet is read (jittery wall clock); the robot ticks every 1 ms.
#include <algorithm>
#include <cstdint>
#include <random>
#include "erd_iiwa/command_retimer.hpp"

namespace
{
// Robot tick k at k ms; the packet is read 0.4 ms later +- 0.3 ms.
struct JitteredClock
{
  std::mt19937 rng{7};
  std::uniform_real_distribution<double> jitter{-0.3e-3, 0.3e-3};
  std::int64_t robot_ns(int k) const {return 1700000000000000000LL + 1000000LL * k;}
  std::int64_t wall_ns(int k) {return 5000000000LL + 1000000LL * k + 400000LL + std::llround(1e9 * jitter(rng));}
};
}  // namespace

TEST(CommandRetimer, FirstStepAndResetPassTheTargetThrough)
{
  erd_iiwa::detail::CommandRetimer<2> retimer;
  std::array<double, 2> target{0.5, -0.25};
  EXPECT_EQ(retimer.step(target, 1000, 2000), target);
  target = {0.6, -0.35};
  retimer.reset();
  EXPECT_EQ(retimer.step(target, 5000, 9000), target);
}

TEST(CommandRetimer, JitteredRampBecomesEvenSteps)
{
  erd_iiwa::detail::CommandRetimer<1> retimer;
  JitteredClock clock;
  const double velocity = -0.0885;  // rad/s, the A4 approach of iiwa_real_01
  double previous = 0.0, worst_raw = 0.0, worst_step = 0.0, previous_target = 0.0;
  for (int k = 0; k < 20000; ++k) {
    const std::int64_t wall = clock.wall_ns(k);
    std::array<double, 1> target{velocity * 1e-9 * static_cast<double>(wall)};
    const double command = retimer.step(target, wall, clock.robot_ns(k))[0];
    if (k > 0) {
      worst_raw = std::max(worst_raw, std::fabs((target[0] - previous_target) - velocity * 1e-3));
      worst_step = std::max(worst_step, std::fabs((command - previous) - velocity * 1e-3));
      EXPECT_LT(std::fabs(command - target[0]), std::fabs(velocity) * 0.7e-3);  // on the JTC trajectory
    }
    previous = command;
    previous_target = target[0];
  }
  // Raw steps are uneven by up to ~60 %; re-timed steps by well under 1 %.
  EXPECT_GT(worst_raw, 0.4 * std::fabs(velocity) * 1e-3);
  EXPECT_LT(worst_step, 0.01 * std::fabs(velocity) * 1e-3);
}

TEST(CommandRetimer, HeldTargetPassesThroughExactly)
{
  erd_iiwa::detail::CommandRetimer<1> retimer;
  JitteredClock clock;
  std::array<double, 1> target{0.25};
  for (int k = 0; k < 2000; ++k) {
    EXPECT_EQ(retimer.step(target, clock.wall_ns(k), clock.robot_ns(k))[0], 0.25);
  }
}

TEST(CommandRetimer, StallShiftIsBounded)
{
  erd_iiwa::detail::CommandRetimer<1> retimer;
  std::array<double, 1> target{0.0};
  retimer.step(target, 0, 0);
  for (int k = 1; k < 3000; ++k) {  // settle the latency mean at 0
    target[0] = 0.1 * 1e-3 * k;
    retimer.step(target, 1000000LL * k, 1000000LL * k);
  }
  target[0] = 0.1 * 1e-3 * 3050;  // packet read 50 ms late
  const double command = retimer.step(target, 1000000LL * 3050, 1000000LL * 3000)[0];
  EXPECT_LE(std::fabs(command - target[0]), 0.1 * 3e-3 + 1e-9);
}
