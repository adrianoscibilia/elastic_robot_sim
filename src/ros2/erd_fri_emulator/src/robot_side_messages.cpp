// SPDX-License-Identifier: Apache-2.0
#include "erd_fri_emulator/robot_side_messages.hpp"

#include <cstdio>

#include "pb_decode.h"
#include "pb_encode.h"

namespace erd_fri_emulator
{

MonitoringMessageEncoder::MonitoringMessageEncoder(FRIMonitoringMessage * message, int num_joints)
: num_joints_(num_joints), message_(message)
{
  init_message();
}

void MonitoringMessageEncoder::init_message()
{
  message_->header.messageIdentifier = kLbrMonitorMessageId;
  message_->header.sequenceCounter = 0;
  message_->header.reflectedSequenceCounter = 0;

  message_->has_robotInfo = true;
  message_->robotInfo.has_numberOfJoints = true;
  message_->robotInfo.numberOfJoints = num_joints_;
  message_->robotInfo.has_safetyState = true;
  message_->robotInfo.safetyState = SafetyState_NORMAL_OPERATION;
  message_->robotInfo.has_operationMode = true;
  message_->robotInfo.operationMode = OperationMode_AUTOMATIC_MODE;
  message_->robotInfo.has_controlMode = true;
  message_->robotInfo.controlMode = ControlMode_POSITION_CONTROLMODE;

  message_->has_monitorData = true;
  message_->monitorData.has_measuredJointPosition = true;
  message_->monitorData.has_measuredTorque = true;
  message_->monitorData.has_commandedJointPosition = true;
  message_->monitorData.has_commandedTorque = true;
  message_->monitorData.has_externalTorque = true;
  message_->monitorData.readIORequest_count = 0;
  message_->monitorData.has_timestamp = true;

  message_->has_connectionInfo = true;
  message_->connectionInfo.sessionState = FRISessionState_IDLE;
  message_->connectionInfo.quality = FRIConnectionQuality_EXCELLENT;
  message_->connectionInfo.has_sendPeriod = true;
  message_->connectionInfo.has_receiveMultiplier = true;
  message_->connectionInfo.receiveMultiplier = 1;

  message_->has_ipoData = true;
  message_->ipoData.has_jointPosition = true;
  message_->ipoData.has_clientCommandMode = true;
  message_->ipoData.clientCommandMode = ClientCommandMode_POSITION;
  message_->ipoData.has_overlayType = true;
  message_->ipoData.overlayType = OverlayType_NO_OVERLAY;
  message_->ipoData.has_trackingPerformance = true;
  message_->ipoData.trackingPerformance = 1.0;

  message_->requestedTransformations_count = 0;
  message_->has_endOfMessageData = false;

  map_repeatedDouble(FRI_MANAGER_NANOPB_ENCODE, num_joints_,
                    &message_->monitorData.measuredJointPosition.value, &fields_.measured_position);
  map_repeatedDouble(FRI_MANAGER_NANOPB_ENCODE, num_joints_,
                    &message_->monitorData.measuredTorque.value, &fields_.measured_torque);
  map_repeatedDouble(FRI_MANAGER_NANOPB_ENCODE, num_joints_,
                    &message_->monitorData.commandedJointPosition.value, &fields_.commanded_position);
  map_repeatedDouble(FRI_MANAGER_NANOPB_ENCODE, num_joints_,
                    &message_->monitorData.commandedTorque.value, &fields_.commanded_torque);
  map_repeatedDouble(FRI_MANAGER_NANOPB_ENCODE, num_joints_,
                    &message_->monitorData.externalTorque.value, &fields_.external_torque);
  map_repeatedDouble(FRI_MANAGER_NANOPB_ENCODE, num_joints_,
                    &message_->ipoData.jointPosition.value, &fields_.ipo_position);
  map_repeatedInt(FRI_MANAGER_NANOPB_ENCODE, num_joints_,
                 &message_->robotInfo.driveState, &fields_.drive_state);
}

bool MonitoringMessageEncoder::encode(char * buffer, int & size)
{
  pb_ostream_t stream = pb_ostream_from_buffer(reinterpret_cast<uint8_t *>(buffer), kFriMonitorMsgMaxSize);
  bool status = pb_encode(&stream, FRIMonitoringMessage_fields, message_);
  size = static_cast<int>(stream.bytes_written);
  if (!status) {
    std::fprintf(stderr, "erd_fri_emulator: monitoring encode error: %s\n", PB_GET_ERROR(&stream));
  }
  return status;
}

CommandMessageDecoder::CommandMessageDecoder(FRICommandMessage * message, int num_joints)
: num_joints_(num_joints), message_(message)
{
  init_message();
}

void CommandMessageDecoder::init_message()
{
  message_->header.messageIdentifier = 0;
  message_->header.sequenceCounter = 0;
  message_->header.reflectedSequenceCounter = 0;
  message_->has_commandData = false;
  message_->commandData.has_jointPosition = false;
  message_->commandData.has_jointTorque = false;
  message_->commandData.has_cartesianWrenchFeedForward = false;
  message_->commandData.commandedTransformations_count = 0;
  message_->commandData.writeIORequest_count = 0;
  message_->has_endOfMessageData = false;

  map_repeatedDouble(FRI_MANAGER_NANOPB_DECODE, num_joints_,
                    &message_->commandData.jointPosition.value, &fields_.joint_position);
  map_repeatedDouble(FRI_MANAGER_NANOPB_DECODE, num_joints_,
                    &message_->commandData.jointTorque.value, &fields_.joint_torque);
}

bool CommandMessageDecoder::decode(char * buffer, int size)
{
  pb_istream_t stream = pb_istream_from_buffer(reinterpret_cast<uint8_t *>(buffer), size);
  bool status = pb_decode(&stream, FRICommandMessage_fields, message_);
  if (!status) {
    std::fprintf(stderr, "erd_fri_emulator: command decode error: %s\n", PB_GET_ERROR(&stream));
  }
  return status;
}

}  // namespace erd_fri_emulator
