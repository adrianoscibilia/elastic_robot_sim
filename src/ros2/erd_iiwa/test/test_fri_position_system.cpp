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
  // 7 joints x 7 state interfaces + 11 GPIO "fri" interfaces.
  EXPECT_EQ(states.size(), 7u * 7u + 11u);
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
