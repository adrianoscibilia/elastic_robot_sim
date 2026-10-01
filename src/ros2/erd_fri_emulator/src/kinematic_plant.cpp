// SPDX-License-Identifier: Apache-2.0
#include "erd_fri_emulator/kinematic_plant.hpp"

#include <cmath>

#include <pinocchio/parsers/urdf.hpp>

namespace erd_fri_emulator
{

KinematicPlant::KinematicPlant(const std::string & urdf_path, double dt, unsigned seed)
: dt_(dt), rng_(seed)
{
  pinocchio::urdf::buildModel(urdf_path, model_);
  data_ = pinocchio::Data(model_);
  q_.fill(0.0);
  dq_.fill(0.0);
  // model_.velocityLimit is indexed like q/v (model.nv), which for this
  // single-DoF-per-joint arm is the same order as our JointArray (RR_04 B-9:
  // "use the bare-asset URDF limits").
  for (int i = 0; i < n_joints() && i < kNumJoints; ++i) {
    velocity_limit_[i] = model_.velocityLimit(i);
  }
}

KinematicPlant::Sample KinematicPlant::step(const JointArray & commanded_position)
{
  const int n = n_joints();
  Eigen::VectorXd q(n), dq(n), ddq(n);

  for (int i = 0; i < n; ++i) {
    // Critically-damped-ish 2nd order lag toward the commanded position
    // (RR_01 S3.5): ddq = omega^2 (u - q) - 2 zeta omega dq.
    double acceleration = omega_ * omega_ * (commanded_position[i] - q_[i]) - 2.0 * zeta_ * omega_ * dq_[i];
    dq_[i] += acceleration * dt_;
    q_[i] += dq_[i] * dt_;
    q(i) = q_[i];
    dq(i) = dq_[i];
    ddq(i) = acceleration;
  }

  Eigen::VectorXd rnea_torque = pinocchio::rnea(model_, data_, q, dq, ddq);

  Sample sample;
  for (int i = 0; i < n; ++i) {
    double quantized = std::round(q_[i] / quantization_) * quantization_;
    sample.measured_position[i] = quantized;

    double torque_noise = 0.01 * effort_limit_[i] * noise_(rng_);
    sample.measured_torque[i] = rnea_torque(i) + torque_noise;

    // Nominal motor friction (round-6 class E, S1.4): viscous+Coulomb on a
    // tanh-smoothed sign of the (true) joint velocity, plus the nominal
    // reflected rotor inertia's contribution to the motor-side equation.
    // Viscous units are Nm*s/rad (RR_04 B-9): `0.05 * tau_eff` alone is a
    // torque, not a viscous coefficient -- it must be divided by the joint's
    // own v_max (R6 S5.4's motor-friction rule) to have the right units and
    // the right order of magnitude at the excitation's actual speeds.
    double viscous = 0.05 * effort_limit_[i] / std::max(velocity_limit_[i], 1e-6);
    double coulomb = 0.02 * effort_limit_[i];
    double friction = viscous * dq_[i] + coulomb * std::tanh(dq_[i] / friction_epsilon_);
    // An independent noise draw, not `torque_noise` again: at rest
    // (dq=ddq=0) `rotor_inertia*ddq + friction` is exactly zero, so without
    // its own noise `commanded_torque` would equal `measured_torque` bit for
    // bit -- precisely the I-9 defect pattern RR_01's own `tau != ft` check
    // exists to catch (found live testing T1.9's real erd_iiwa integration:
    // the standstill sample showed effort == commanded_effort exactly).
    double command_noise = 0.01 * effort_limit_[i] * noise_(rng_);
    sample.commanded_torque[i] = sample.measured_torque[i] + rotor_inertia_[i] * ddq(i) + friction + command_noise;

    sample.external_torque[i] = torque_noise;
  }
  return sample;
}

}  // namespace erd_fri_emulator
