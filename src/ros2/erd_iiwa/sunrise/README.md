# `ErdFri.java` deployment (RR_06 P3-2d)

`src/ErdFri.java` is checked-in **source**, derived from
`iiwa_ros2.java` per `RR_01_ROS2_RECORDING_STACK_PLAN.md` S3.3. Writing it
needed no KUKA tooling; **deploying** it does -- that step needs Sunrise
Workbench and a station project, and is the owner's T2.0 task
(`RR_01_ROS2_RECORDING_STACK_PLAN.md` S7.3), not something this repo's CI or
any `colcon` build can do.

## Prerequisites

* Sunrise Workbench **1.11**, matching the controller's installed
  Sunrise.OS version (RR_04's templates record this per-lab in
  `connection.sunrise_version`; confirm it matches before importing).
* The station project **`iiwa_stack_final`** already exists on this
  Workbench installation (or import/create it first -- out of scope here).
* Network access from the Workbench PC to the KUKA Sunrise Cabinet (the
  usual KONI/smartPAD network, not the FRI client network).

## Steps

1. Open the station project `iiwa_stack_final` in Sunrise Workbench.
2. Under the project's `src/application` package, import
   `ErdFri.java` from this repo (`src/ros2/erd_iiwa/sunrise/src/ErdFri.java`)
   -- copy the file in, or use Workbench's own "Import file" on the
   `application` package.
3. **Process data.** `ErdFri.java` reads the FRI client's IP from a
   station-level process-data entry named `ErdFriClientIp` (falls back to
   `192.170.10.5` if undefined, with a logged warning). In the Workbench
   station editor's "Process data" tab, add a `String` entry named exactly
   `ErdFriClientIp` and set it to this lab's actual FRI client IP (the
   laptop's USB-Ethernet adapter address on the FRI network -- the same
   value this lab's `erd.lab/1` YAML declares at `connection.client_ip`).
   If this step is skipped, the application still runs, against the
   default IP, with a warning in the Sunrise log -- confirm that's actually
   this lab's IP before relying on it.
4. Synchronize the project to the KUKA Sunrise Cabinet (Workbench's
   "Install" / sync action).
5. On the smartHMI, select the `ErdFri` application, confirm the robot is
   in a safe, known pose (the overlay does **not** move the robot to a
   fixed position at start -- RR_01 S3.3's "no automatic `ptp`" -- so
   whatever pose the robot is holding when the user button is pressed is
   where the FRI session starts from), then run it.
6. On the ROS side, bring up `erd_iiwa/launch/iiwa.launch.py
   hardware:=real robot_ip:=<client_ip> lab_config:=<this lab's YAML>` --
   `fri_send_period_ms` in that YAML **must equal** `ErdFri.java`'s
   `SEND_PERIOD_MS` (1 ms); RR_01 S3.3 flags this as a value that must stay
   in sync by hand across the two sides.
7. The smartPAD's "User button" (the Media Flange's `UserButton` input)
   ends the FRI overlay at any time, independent of the ROS side -- this is
   the operator's immediate stop during commissioning, in addition to
   RR_01 S9's e-stop rule.

## What to check after deploying, before trusting it

* The Sunrise log shows `ErdFri: FRI connection established` with a
  plausible jitter figure (not a timeout) -- confirms `ErdFriClientIp`
  actually matches the FRI client's listening address.
* `fri/session_state` on the ROS side reaches `COMMANDING_ACTIVE` (RR_04
  A-4) only once this application is running and past the button-press
  step above; `erd_recording`'s preflight already refuses to activate the
  JTC before that (unchanged by this pass).
