# Simulation Framework and Synthetic Dataset Generation for an Elastic Cartesian Robot

## Purpose and Intended Use

This document is a technical source for drafting the simulation-framework and synthetic-dataset sections of a scientific paper. It describes a physics-based data-generation workflow for a three-axis Cartesian robot with compliant transmission dynamics. The synthetic trajectories were used to train neural-network dynamic models. The discussion of physical experiments is intentionally limited to the interfaces that make simulator trajectories replayable and calibratable; a dedicated real-robot section should be drafted separately.

The wording below distinguishes between (i) the implemented simulation and dataset-generation procedure, (ii) software-level verification checks, and (iii) experimental claims that require independently reported results. In particular, passing a software test should not be presented as evidence of simulation-to-hardware accuracy.

## System Overview

The simulated platform is a three-axis Cartesian gantry with translational $X$, $Y$, and $Z$ axes. The nominal mechanical structure, kinematic limits, visual geometry, and link inertial properties are defined in a URDF model. The translational workspace is

$$
x \in [-1.8, 1.8]~\mathrm{m}, \qquad
y \in [-1.8, 1.8]~\mathrm{m}, \qquad
z \in [-1.0, 1.0]~\mathrm{m}.
$$

The URDF model represents a fixed supporting structure, a serial Cartesian chain, a flange, an end-effector link, and a yaw joint. The three translational joints have nominal velocity limits of $1~\mathrm{m/s}$ in the URDF. The end-effector carries an optional rigid cubic payload, attached by a fixed joint. Gravity is enabled with acceleration $(0,0,-9.81)~\mathrm{m/s^2}$.

The physical source model includes the following principal moving-link inertial masses: $0.2~\mathrm{kg}$ for the first translating link, $0.6~\mathrm{kg}$ for the intermediate beam, $0.2~\mathrm{kg}$ for the second translating link, $0.97~\mathrm{kg}$ for the vertical link, and $0.01~\mathrm{kg}$ for each of the flange, force/torque, and end-effector links. These values are retained as authoritative inertial properties during model construction; visual geometry is not used to infer mass.

## Elastic-Drive Model

### Motor and Link Coordinates

Each translational direction is represented by a motor-side coordinate and a compliant link-side coordinate. The motor coordinate is position-velocity controlled, while the link coordinate is connected through a passive spring-damper element. This creates a two-degree-of-freedom representation of each translational drive, allowing the recorded state to include both motor motion and elastic displacement.

For axis $i \in \{x,y,z\}$, the motor-side controller uses the fixed gains

$$
k_{p,m} = 30{,}000~\mathrm{N/m}, \qquad
k_{d,m} = 500~\mathrm{N\,s/m}.
$$

The compliant element is parameterized by stiffness $k_i$ and damping ratio $\zeta_i$. The physical damping coefficient is derived rather than sampled directly:

$$
c_i = 2\zeta_i\sqrt{k_i m_{\mathrm{eff},i}}.
$$

The effective masses used in this conversion are $m_{\mathrm{eff},x}=1.2~\mathrm{kg}$, $m_{\mathrm{eff},y}=1.8~\mathrm{kg}$, and $m_{\mathrm{eff},z}=1.0~\mathrm{kg}$. They are explicit modeling constants used to map a dimensionless damping ratio to a damping coefficient. They should be described as effective parameters of the compliant-drive model, rather than as direct measurements of individual link mass.

The simulation is assembled from the URDF and then rebuilt so that the three compliant joints are injected into the kinematic chain. The $Y$ compliance is introduced between the first translating link and its beam. The $X$ and $Z$ compliances use intermediate links with negligible mass ($10^{-4}~\mathrm{kg}$) to separate the motor and elastic degrees of freedom without materially adding inertia. The yaw joint is fixed in the simulated dynamic chain because rotational tool motion is outside the present three-axis dataset scope.

### Tunable Parameters and Domain Randomization

The synthetic dataset varies six elastic-drive parameters: three stiffnesses and three damping ratios. The values sampled for each robot configuration are:

| Axis | Stiffness interval $k_i$ (N/m) | Damping-ratio interval $\zeta_i$ |
| --- | ---: | ---: |
| $X$ | 3,000 to 9,000 | 0.45 to 1.05 |
| $Y$ | 3,000 to 9,000 | 0.45 to 1.05 |
| $Z$ | 2,500 to 7,000 | 0.55 to 1.20 |

Stiffness is sampled log-uniformly:

$$
\log_{10}(k_i) \sim \mathcal{U}\left(\log_{10} k_{i,\min},\log_{10} k_{i,\max}\right),
$$

whereas the damping ratio is sampled uniformly in its stated interval. This design gives coverage across compliant and stiffer drive behavior without concentrating samples at the upper end of a linear stiffness interval. The sampled configuration is fixed for all trajectories of one virtual robot, thereby separating between-robot compliance variation from within-robot trajectory variation.

The implementation also supports a rigid payload parameter in the interval $[0,6]~\mathrm{kg}$. However, in the dataset-generation configuration described here, the number of payload levels is zero, which causes every generated rollout to use a payload of $0.0~\mathrm{kg}$. Consequently, the reported training corpus is not a payload-randomized dataset. Payload variation should only be claimed for a separately executed configuration in which payload levels are enabled and documented.

## Simulation Engine and Numerical Procedure

### Primary Dataset Backend

Synthetic training data are generated with the Newton physics backend, accessed through NVIDIA Warp and run on a CUDA-capable GPU. The batch executable invokes the Newton simulator as a separate headless process for each rollout. The model is initialized from the rest configuration for every rollout, which prevents state carry-over between episodes.

The standard numerical settings are a simulation horizon of $15~\mathrm{s}$ and integration step

$$
\Delta t = 0.01~\mathrm{s}.
$$

Thus, a full untrimmed rollout contains 1,500 integration samples at a nominal sampling frequency of $100~\mathrm{Hz}$. The simulation loop obtains the reference position and reference velocity, sets the three motor targets, advances the physics solver, and records the motor- and link-side state and effort signals. In the batch CSV workflow, the initial transient is retained; the separate calibration workflow can exclude an initial interval during comparison.

The framework includes a MuJoCo backend with the same high-level state schema and a partially developed Isaac Sim entry point. These alternatives are useful for cross-backend studies or future extensions, but the synthetic dataset described here is generated only with the Newton backend. They should not be described as sources of samples in this corpus unless such runs are separately reported.

### Measurement Model

The simulator can inject a simple measurement model into the recorded signals. Position is quantized at a nominal encoder resolution of $1.2\times10^{-8}~\mathrm{m}$ and then perturbed with zero-mean Gaussian noise of standard deviation $5\times10^{-6}~\mathrm{m}$. Velocity receives zero-mean Gaussian noise with standard deviation $10^{-2}~\mathrm{m/s}$. Force/torque-like joint effort signals receive zero-mean Gaussian noise with standard deviation

$$
\sigma_{\tau} = 0.03|\tau| + 0.01.
$$

The standalone Newton batch simulation is configured to use this noisy measurement output. Calibration simulations can disable noise to obtain a smoother objective function. The manuscript should state whether neural networks were trained using the noisy measurements, the ideal state, or a later preprocessed version of the CSV data; the batch executable itself exports the noisy state streams.

## Reference-Trajectory Generation

### Trajectory Families

The batch procedure samples one of two reference-motion families for each rollout:

1. **Sinusoidal trajectories (mode 1).** Each Cartesian axis receives a seeded sinusoid; the $Y$ axis uses a cosine convention to match the controller coordinate convention.
2. **Point-to-point trajectories (mode 2).** Random Cartesian waypoints are joined by cubic blends with zero velocity at waypoint boundaries.

A hold trajectory mode is implemented for static analysis but is not selected by the batch dataset generator. Reference trajectories are deterministic given their seed and configuration. The batch uses simulator seeds from 1000 through 1099, one per rollout.

### Workspace and Velocity Governance

References are generated within the joint limits given above. The configured Cartesian peak-speed cap is

$$
v_{\mathrm{cart,max}} = 0.5~\mathrm{m/s}.
$$

For sinusoidal trajectories, the implementation computes a conservative Euclidean speed bound from the axis amplitudes and frequencies. For point-to-point motion, it uses the analytical maximum speed of cubic interpolation. A single global factor is chosen as the more restrictive of the user-requested speed factor and the factor required by Cartesian or optional axis-wise velocity limits. The factor is then baked into the generated trajectory: sinusoidal frequencies are reduced, or point-to-point segment durations are increased. This preserves the spatial path while preventing speed-limit violation.

The current generator uses seeded draws with amplitudes from 10% to 40% of each axis range and angular frequencies from $0.5$ to $3.0~\mathrm{rad/s}$ for sinusoidal motion. Point-to-point motion uses a nominal $2~\mathrm{s}$ segment duration and requires consecutive sampled waypoints to be at least $0.2~\mathrm{m}$ apart. A startup envelope of the form

$$
r(t) = 1 - \exp(-t/\tau_r), \qquad \tau_r = 1~\mathrm{s},
$$

is applied to the sinusoidal position reference. For rigorous reporting, the reference generator should be treated as the implementation of record. In particular, some trajectory-shape fields exposed in configuration are not yet consumed by the sinusoidal draw logic, and the implemented sinusoidal velocity expression applies the envelope to the sinusoidal derivative without the additional $\dot{r}(t)$ term. Any paper that analyzes early-time reference acceleration or differentiability should account for this detail or use a corrected, versioned generator.

### Reproducible Replay Path

Outside the batch CSV workflow, trajectory definitions can be serialized to JSON. The serialized object stores the mode, seed, limits, generated sinusoidal coefficients or point-to-point waypoints, nominal and executed duration, speed scaling, peak-speed metadata, and controller sampling density. This path supports replay of identical reference motions in the Newton and MuJoCo backends and on the physical robot.

This replay mechanism is distinct from the batch generator. The batch generator stores seed and drive information in the CSV filename, but it does not write one trajectory JSON file per batch rollout and it does not create a batch manifest. Therefore, exact trajectory reconstruction from the synthetic CSV dataset depends on preserving the generator version, the settings template, the seed, and the filename metadata. For a future dataset release, a machine-readable per-rollout manifest containing sampled physical parameters, full trajectory parameters, code revision, solver version, and random seeds would materially improve provenance.

## Batch Dataset-Generation Protocol

### Experimental Design

The batch executable creates 10 virtual robot configurations. For each virtual robot, it runs 10 trajectory trials, giving

$$
N_{\mathrm{rollouts}} = 10 \times 10 = 100.
$$

At the default horizon and time step, this corresponds to approximately 150,000 time-stamped rows before any downstream filtering or file-level failures. The random-number generator used for sampling virtual robots and choosing reference modes is initialized with seed 42. Parameter values are rounded before insertion into the runtime configuration. Each rollout receives a distinct simulator seed, incremented from 1000.

The generator first constructs a template of 10 randomly chosen trajectory modes, then shuffles that template independently for each virtual robot. This produces varied sinusoidal and point-to-point excitation but does not enforce exactly equal counts of the two motion types. The actual mode distribution should therefore be reported from the generated files, rather than assumed to be balanced.

For every trial, the generator makes a deep copy of the initial settings, substitutes the sampled elastic parameters and trial-specific mode and seed, writes the temporary runtime settings file, and starts a headless Newton process. A `finally` operation restores the original settings file even when a batch process fails. The restoration protects the baseline configuration, but the temporary shared settings-file design means concurrent batch invocations should be avoided.

### Output Naming and Parameter Traceability

Each rollout is written to a timestamped CSV file. Its filename encodes the simulation seed, payload, the three stiffness values, the three derived damping coefficients, and the reference mode. The approximate format is

```text
d<time>_s<seed>_p<payload>kg_X<kx>_<cx>_Y<ky>_<cy>_Z<kz>_<cz>_ref<mode>.csv
```

The encoded damping values are physical damping coefficients rounded for the filename; they are not the sampled damping ratios. The damping ratios are present only transiently in the settings used to launch a rollout unless separately archived. Any data-processing pipeline should parse filenames carefully and should retain a mapping from each file to the six sampled parameters, the payload, and the reference-mode label.

## Recorded Dataset Schema

Each CSV has 25 columns sampled on the simulation grid:

| Signal group | Columns | Unit | Interpretation |
| --- | --- | --- | --- |
| Time | `t` | s | Simulation time |
| Reference position | `ref_x`, `ref_y`, `ref_z` | m | Cartesian motor reference |
| Reference velocity | `vel_x`, `vel_y`, `vel_z` | m/s | Reference velocity supplied to the motor controller |
| Motor position | `q_motor_x`, `q_motor_y`, `q_motor_z` | m | Motor-side translational coordinate |
| Link position | `q_link_x`, `q_link_y`, `q_link_z` | m | Compliant link-side coordinate |
| Motor velocity | `dq_motor_x`, `dq_motor_y`, `dq_motor_z` | m/s | Motor-side velocity |
| Link velocity | `dq_link_x`, `dq_link_y`, `dq_link_z` | m/s | Link-side velocity |
| Motor effort | `tau_motor_x`, `tau_motor_y`, `tau_motor_z` | N in the prismatic model | Motor-joint generalized effort |
| Elastic effort | `tau_link_x`, `tau_link_y`, `tau_link_z` | N in the prismatic model | Compliant-joint generalized effort |

Although several variable names use `tau`, the translational joints are prismatic. The corresponding generalized efforts should be described as forces in newtons, not rotary torques in newton-metres.

The state choice exposes elastic deformation directly through differences between motor- and link-side coordinates. This is useful for learning forward dynamics or state-transition models because it retains latent compliant behavior that would be lost if only motor positions were exported. When constructing neural-network inputs and targets, the train/validation/test split should be performed at the virtual-robot configuration level, not only at the row or trajectory level, if the intended evaluation is generalization across unseen compliance parameters. A row-level random split would leak the same sampled mechanical parameters into all partitions.

## Verification Evidence Implemented in Software

The codebase contains focused checks that support the internal consistency of the dataset pipeline:

| Verification target | Implemented check | Scope of supported claim |
| --- | --- | --- |
| Position limits | Generated sinusoidal and point-to-point positions are sampled for multiple seeds and checked against configured limits. | Reference generators remain in the nominal workspace. |
| Peak velocity | Multiple seeded trajectories are checked against the $0.5~\mathrm{m/s}$ Cartesian limit. | The generated reference velocity respects the configured cap. |
| Replay | A serialized trajectory is reloaded and its velocities are compared over sampled times. | The JSON trajectory representation can reproduce the generated reference trajectory. |
| Legacy compatibility | Older trajectory JSON records with speed overrides are reconstructed. | Historical serialized trajectories retain a replay path. |
| Tabular schema | Required columns and optional Parquet round-trip are checked. | The standardized rollout schema is preserved. |
| Settings validation | Invalid limits, velocity limits, frequencies, and amplitudes are rejected. | Basic trajectory configuration consistency is checked. |
| Static vertical compliance | A payload sweep script compares settled vertical deflection with $\Delta z=(m+p)g/k_z$. | The vertical spring model is checked against a static force-balance expectation. |

These checks are software and model-consistency tests. They do not establish numerical-convergence error, contact-model fidelity, parameter identifiability, or agreement with hardware. Those claims require reported experimental protocols, comparison metrics, confidence intervals, and measured results.

## Connection to Calibration and Physical-Robot Work

The framework contains an optional calibration path that is relevant when moving from synthetic data toward the physical platform. It can replay an identical serialized trajectory in simulation and compare it with a recorded physical rollout. The calibratable parameter vector consists of the same six elastic parameters and, optionally, payload mass. Stiffness is represented in log space during normalization, while damping ratios and payload use linear coordinates.

The default calibration objective compares position, velocity, and force streams after resampling onto a common time grid and optionally excluding an initial transient. The normalized objective is

$$
\mathcal{L} = w_q\,\mathrm{NRMSE}_q + w_{\dot q}\,\mathrm{NRMSE}_{\dot q} + w_F\,\mathrm{NRMSE}_F,
$$

with default weights $w_q=1.0$, $w_{\dot q}=0.3$, and $w_F=0.5$. The normalization scale for each signal is the 95th percentile of its absolute magnitude, subject to a small positive floor. CMA-ES is the default optimization method; Bayesian optimization and an experimental reinforcement-learning optimizer are also available.

For physical recordings, a ROS 2 node sends a `FollowJointTrajectory` goal, records joint states and a three-component force/torque stream, and resamples the result to a $100~\mathrm{Hz}$ grid. This makes data layout and trajectory replay compatible with simulated rollouts. It is not evidence that the motor-side and link-side states are equally observable in hardware: the real recorder uses motor position and velocity as proxies for unmeasured elastic-link motion. Likewise, the recorded force/torque signal may contain sensor bias, gravity preload, and coordinate/sign-convention differences from the simulated elastic reaction force. These observability and force-alignment issues should be addressed explicitly before presenting calibrated parameters as physically identified ground truth.

## Recommended Scientific Framing

The simulation section can state that a parameter-randomized, physics-based Newton model of a compliant Cartesian robot was used to generate multi-axis dynamic trajectories for learning. It can describe the three motor-link spring-damper pairs, the sampled stiffness and damping-ratio ranges, the 15-second and 100-Hz rollout settings, the two trajectory families, and the 100-rollout default corpus. The data representation should be described as motor-side and link-side positions, velocities, and generalized forces together with reference states.

The paper should avoid the following unsupported formulations unless additional evidence is provided elsewhere:

- Do not call the default corpus payload-randomized; it uses zero payload.
- Do not call the two motion families balanced; their counts are stochastic in the batch generator.
- Do not say that a JSON trajectory exists for every CSV batch sample; JSON serialization belongs to a separate replay workflow.
- Do not claim real-world fidelity, calibrated-parameter accuracy, or neural-network performance from the software implementation alone.
- Do not label prismatic generalized efforts as torques with units of newton-metres.
- Do not state that all configurable trajectory-shape fields govern the current generator without checking the exact code version used for dataset generation.

## Information to Add Before Final Manuscript Submission

The following run-specific information should be supplied from the actual dataset-generation logs and model-training pipeline before finalizing the paper:

1. The generation date, operating system, GPU model, CUDA/Warp/Newton versions, and exact code revision.
2. The number of successful CSV files, total samples after preprocessing, and observed counts of sinusoidal versus point-to-point trajectories.
3. A manifest of each sampled stiffness, damping ratio, derived damping coefficient, payload, mode, and seed.
4. The neural-network input variables, target variables, normalization procedure, sequence length, sampling or downsampling policy, and treatment of noise.
5. The split strategy, including confirmation that all rollouts from a sampled virtual robot are isolated to a single partition when testing parameter generalization.
6. Solver settings beyond the public time step, together with at least one time-step or solver-sensitivity analysis if numerical accuracy is a paper claim.
7. Quantitative results for learned-model prediction, rollout stability, and comparison against appropriate baseline models.
8. For a later sim-to-real section, the physical sensing arrangement, coordinate transforms, force bias-removal method, synchronization procedure, calibration data split, and held-out hardware metrics.

## Concise Methods Statement

A three-axis Cartesian robot was modeled as a URDF-based rigid-body system augmented with one compliant spring-damper transmission per translational axis. Each motor-side prismatic coordinate was regulated by a fixed position-velocity controller and coupled to a link-side coordinate through a spring whose stiffness and damping ratio were randomized across virtual robot instances. Stiffness values were sampled log-uniformly over axis-specific intervals and damping ratios uniformly over underdamped to slightly overdamped ranges, with damping coefficients computed from an effective-mass relation. The primary Newton/Warp simulation generated 15-second trajectories at 100 Hz under gravity, using either seeded multi-axis sinusoidal references or seeded cubic point-to-point references constrained to the Cartesian workspace and a 0.5 m/s peak-speed limit. The default batch produced 10 sampled robot configurations with 10 rollouts per configuration. Each rollout exported references, motor-side and link-side positions and velocities, and motor and elastic generalized forces, providing state histories suitable for training neural dynamic models of compliant Cartesian motion.