// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <random>
#include <string>

#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>

namespace erd_fri_emulator
{

constexpr int kNumJoints = 7;
using JointArray = std::array<double, kNumJoints>;

/// RR_01 S3.5's "kinematic" default plant: physically-meaningful dummy
/// signals, not a claim of fidelity to the real robot or KUKA's controller.
///
/// * measured position follows the commanded position through a 2nd-order
///   lag (omega, zeta) plus quantization;
/// * measured torque = RNEA(q, qdot, qddot) on the bare asset URDF + noise;
/// * commanded torque = measured + nominal rotor inertia * qddot + nominal
///   motor friction(qdot);
/// * external torque = noise alone.
///
/// Nominal rotor inertia / friction values are the round-6
/// `kuka_lbr_iiwa_14_r820_table_round6_drive.yaml` priors (S1.4's class-E
/// numbers), hardcoded here since this plant does not read the training
/// config -- good enough for exercising the recording pipeline, not for any
/// physical claim.
class KinematicPlant
{
public:
  struct Sample
  {
    JointArray measured_position{};
    JointArray measured_torque{};
    JointArray commanded_torque{};
    JointArray external_torque{};
  };

  KinematicPlant(const std::string & urdf_path, double dt, unsigned seed = 0);

  /// Advances the plant by one `dt` toward `commanded_position` and returns
  /// the resulting sample.
  Sample step(const JointArray & commanded_position);

  /// Places the plant at rest at `position` (RR_10: the emulator starts
  /// where the operator left the real robot, the lab config's home).
  void reset(const JointArray & position);

  int n_joints() const { return static_cast<int>(model_.nv); }
  const JointArray & current_position() const { return q_; }

private:
  pinocchio::Model model_;
  pinocchio::Data data_;
  double dt_;
  double omega_{60.0};
  double zeta_{0.8};
  double quantization_{5.989e-8};
  JointArray q_{};
  JointArray dq_{};
  JointArray rotor_inertia_{{1.0, 1.0, 0.5, 0.5, 0.25, 0.15, 0.15}};
  JointArray effort_limit_{{320.0, 320.0, 176.0, 176.0, 110.0, 40.0, 40.0}};
  // Read from the bare-asset URDF's own `<limit velocity>` at construction
  // time (RR_04 B-9), not hardcoded like the other nominal priors: the
  // viscous coefficient below needs it and this is the one number the URDF
  // itself already states authoritatively.
  JointArray velocity_limit_{};
  double friction_epsilon_{1.0e-2};
  std::mt19937 rng_;
  std::normal_distribution<double> noise_{0.0, 1.0};
};

}  // namespace erd_fri_emulator
