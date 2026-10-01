// SPDX-License-Identifier: Apache-2.0
//
// The KUKA FRI SDK bundled in iiwa_ros2 ships only client-direction message
// helpers: `MonitoringMessageDecoder` (decode robot->client) and
// `CommandMessageEncoder` (encode client->robot) -- both in
// friMonitoringMessageDecoder.{h,cpp}/friCommandMessageEncoder.{h,cpp}. A
// robot-side emulator needs the opposite pair: encode the monitoring message,
// decode the command message. `FRIMonitoringMessage`/`FRICommandMessage`
// (FRIMessages.pb.h) are plain, unrestricted nanopb structs -- unlike
// `LBRState`/`LBRCommand`, nothing here is `friend`-scoped to `IRDFClient` --
// and `pb_frimessages_callbacks.h`'s `map_repeatedDouble`/`map_repeatedInt`
// already take a `FRI_MANAGER_NANOPB_ENCODE`/`_DECODE` direction argument, so
// building the missing pair is exactly mirroring
// `friMonitoringMessageDecoder.cpp`/`friCommandMessageEncoder.cpp` with the
// direction flipped (verified field-by-field against those two files, T1.9).
#pragma once

#include "FRIMessages.pb.h"
#include "pb_frimessages_callbacks.h"

namespace erd_fri_emulator
{

constexpr int kFriMonitorMsgMaxSize = 1500;
constexpr int kFriCommandMsgMaxSize = 1500;

// Same values as KUKA::FRI::LBRState::LBRMONITORMESSAGEID /
// LBRCommand::LBRCOMMANDMESSAGEID (friLBRState.h/friLBRCommand.h) -- plain
// protocol constants, not the friend-scoped classes that declare them, so
// redeclaring them here needs no access to either.
constexpr uint32_t kLbrMonitorMessageId = 0x245142;
constexpr uint32_t kLbrCommandMessageId = 0x34001;

/// Encodes a `FRIMonitoringMessage` (robot -> client), mirroring
/// `friMonitoringMessageDecoder.cpp`'s field wiring with the opposite
/// nanopb callback direction.
class MonitoringMessageEncoder
{
public:
  explicit MonitoringMessageEncoder(FRIMonitoringMessage * message, int num_joints = 7);

  /// Encodes the message as currently filled in `message()`. Returns false
  /// (and prints nanopb's own error) on a buffer overrun or malformed struct.
  bool encode(char * buffer, int & size);

  FRIMonitoringMessage * message() { return message_; }

private:
  struct RepeatedFields
  {
    tRepeatedDoubleArguments measured_position;
    tRepeatedDoubleArguments measured_torque;
    tRepeatedDoubleArguments commanded_position;
    tRepeatedDoubleArguments commanded_torque;
    tRepeatedDoubleArguments external_torque;
    tRepeatedDoubleArguments ipo_position;
    tRepeatedIntArguments drive_state;

    RepeatedFields()
    {
      init_repeatedDouble(&measured_position);
      init_repeatedDouble(&measured_torque);
      init_repeatedDouble(&commanded_position);
      init_repeatedDouble(&commanded_torque);
      init_repeatedDouble(&external_torque);
      init_repeatedDouble(&ipo_position);
      init_repeatedInt(&drive_state);
    }
    ~RepeatedFields()
    {
      free_repeatedDouble(&measured_position);
      free_repeatedDouble(&measured_torque);
      free_repeatedDouble(&commanded_position);
      free_repeatedDouble(&commanded_torque);
      free_repeatedDouble(&external_torque);
      free_repeatedDouble(&ipo_position);
      free_repeatedInt(&drive_state);
    }
  };

  int num_joints_;
  RepeatedFields fields_;
  FRIMonitoringMessage * message_;

  void init_message();
};

/// Decodes a `FRICommandMessage` (client -> robot), mirroring
/// `friCommandMessageEncoder.cpp`'s field wiring with the opposite nanopb
/// callback direction.
class CommandMessageDecoder
{
public:
  explicit CommandMessageDecoder(FRICommandMessage * message, int num_joints = 7);

  bool decode(char * buffer, int size);

  FRICommandMessage * message() { return message_; }

private:
  struct RepeatedFields
  {
    tRepeatedDoubleArguments joint_position;
    tRepeatedDoubleArguments joint_torque;

    RepeatedFields()
    {
      init_repeatedDouble(&joint_position);
      init_repeatedDouble(&joint_torque);
    }
    ~RepeatedFields()
    {
      free_repeatedDouble(&joint_position);
      free_repeatedDouble(&joint_torque);
    }
  };

  int num_joints_;
  RepeatedFields fields_;
  FRICommandMessage * message_;

  void init_message();
};

}  // namespace erd_fri_emulator
