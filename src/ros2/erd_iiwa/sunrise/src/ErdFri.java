// Copyright 2026 elastic_robot_sim contributors
// SPDX-License-Identifier: Apache-2.0
//
// Derived from iiwa_ros2.java (Copyright 2022, ICube Laboratory, University
// of Strasbourg, Apache-2.0) -- RR_01 S3.3's required changes from that
// original are the ones that matter here, not a rewrite from scratch:
//   * setSendPeriodMilliSec(1), receive multiplier 1 (RR_01: "must equal
//     ErdFri.java" -- erd_recording's lab config's own fri_send_period_ms);
//   * POSITION only (PositionControlMode) -- no mode-selection dialog,
//     no TORQUE/MONITORING branch, no JointImpedanceControlMode;
//   * no start-up `ptp`: the overlay starts from whatever pose the robot is
//     already holding (an automatic move to a fixed joint position the
//     instant the station starts is exactly the kind of unannounced motion
//     RR_01 S9/RR_04 A-2's start-state guard exists to refuse on the ROS
//     side -- it should not happen on the Sunrise side either);
//   * the client IP is a process/application parameter, not a compiled-in
//     constant -- `ErdFriClientIp`, configured per-station in the Sunrise
//     Workbench "Process data" editor; falls back to CLIENT_IP_DEFAULT
//     (192.170.10.5) if the station project hasn't defined it yet;
//   * FRI quality/jitter logging is kept, unchanged;
//   * the user button still ends the overlay (unchanged).

package application;

import com.kuka.roboticsAPI.applicationModel.RoboticsAPIApplication;

import com.kuka.roboticsAPI.conditionModel.BooleanIOCondition;
import com.kuka.roboticsAPI.controllerModel.Controller;
import com.kuka.roboticsAPI.deviceModel.LBR;
import com.kuka.roboticsAPI.motionModel.PositionHold;
import com.kuka.roboticsAPI.motionModel.controlModeModel.PositionControlMode;
import com.kuka.connectivity.fastRobotInterface.ClientCommandMode;
import com.kuka.connectivity.fastRobotInterface.FRIChannelInformation;
import com.kuka.connectivity.fastRobotInterface.FRIConfiguration;
import com.kuka.connectivity.fastRobotInterface.FRIJointOverlay;
import com.kuka.connectivity.fastRobotInterface.FRISession;
import com.kuka.connectivity.fastRobotInterface.IFRISessionListener;
import com.kuka.generated.ioAccess.MediaFlangeIOGroup;

import javax.inject.Inject;

import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

/**
 * ERD's Sunrise-side FRI application (RR_01 S3.3).
 * <p>
 * POSITION-only, fixed 1 ms send period, no start-up motion: the overlay
 * holds whatever pose the robot is already in when the user button starts
 * it, and the ROS side (erd_recording's start-state guard, RR_04 A-2) is the
 * only thing that ever commands the robot to move from there. This
 * deliberately does not offer the original iiwa_ros2.java's
 * POSITION/TORQUE/MONITORING dialog -- erd_iiwa's hardware plugin only ever
 * speaks POSITION (RR_01 S3.3's own guard refuses TORQUE sessions).
 * <p>
 * Deployment: the owner imports this file into the station project
 * {@code iiwa_stack_final} (Sunrise.OS 1.11) with Sunrise Workbench and runs
 * it from the smartPAD like any other application -- see {@code README.md}
 * in this folder for the exact steps. Writing this file needs no Workbench
 * (RR_06 P3-2d); only deploying it does.
 */
public class ErdFri extends RoboticsAPIApplication {
	private Controller _lbrController;
	private LBR _lbr;

	@Inject
	private MediaFlangeIOGroup _medflange;

	private FRISession _friSession;
	private FRIJointOverlay _jointOverlay;
	private PositionHold _positionHold;

	//: RR_01 S3.3: 1 ms send period, receive multiplier 1 -- must equal
	//: erd_recording's lab config `connection.fri_send_period_ms`.
	private static final double SEND_PERIOD_MS = 1;
	private static final int RECEIVE_MULTIPLIER = 1;

	//: RR_06 P3-2c: the client IP is a process/application parameter
	//: (Sunrise Workbench "Process data" editor, key below), not a compiled
	//: constant; this is only the fallback for a station project that
	//: hasn't defined it yet.
	private static final String CLIENT_IP_PROCESS_DATA_KEY = "ErdFriClientIp";
	private static final String CLIENT_IP_DEFAULT = "192.170.10.5";

	private final IFRISessionListener _listener = new IFRISessionListener() {
		@Override
		public void onFRIConnectionQualityChanged(FRIChannelInformation friChannelInformation) {
			getLogger().info("ErdFri: quality changed -- quality=" + friChannelInformation.getQuality()
					+ " jitter=" + friChannelInformation.getJitter()
					+ " latency=" + friChannelInformation.getLatency());
		}

		@Override
		public void onFRISessionStateChanged(FRIChannelInformation friChannelInformation) {
			getLogger().info("ErdFri: session state changed -- state="
					+ friChannelInformation.getFRISessionState()
					+ " jitter=" + friChannelInformation.getJitter()
					+ " latency=" + friChannelInformation.getLatency());
		}
	};

	@Override
	public void initialize() {
		_lbrController = (Controller) getContext().getControllers().toArray()[0];
		_lbr = (LBR) _lbrController.getDevices().toArray()[0];
		_lbr.attachTo(_lbr.getFlange());
		// Deliberately no `_lbr.move(ptp(...))` here (RR_01 S3.3: "no
		// automatic ptp at start") -- the overlay below holds the pose the
		// robot is already in.
	}

	private String clientIp() {
		try {
			return getApplicationData().getProcessData(CLIENT_IP_PROCESS_DATA_KEY).getValue().toString();
		} catch (Exception e) {
			getLogger().warn("ErdFri: process data '" + CLIENT_IP_PROCESS_DATA_KEY
					+ "' not configured for this station project, using default " + CLIENT_IP_DEFAULT);
			return CLIENT_IP_DEFAULT;
		}
	}

	@Override
	public void run() {
		_medflange.setLEDRed(true);

		PositionControlMode controlMode = new PositionControlMode();
		_positionHold = new PositionHold(controlMode, -1, TimeUnit.MINUTES);

		FRIConfiguration friConfiguration = FRIConfiguration.createRemoteConfiguration(_lbr, clientIp());
		friConfiguration.setSendPeriodMilliSec(SEND_PERIOD_MS);
		friConfiguration.setReceiveMultiplier(RECEIVE_MULTIPLIER);

		getLogger().info("ErdFri: creating FRI connection to " + friConfiguration.getHostName());
		getLogger().info("ErdFri: sendPeriod=" + friConfiguration.getSendPeriodMilliSec() + " ms"
				+ " receiveMultiplier=" + friConfiguration.getReceiveMultiplier());

		_friSession = new FRISession(friConfiguration);
		_friSession.addFRISessionListener(_listener);

		try {
			_friSession.await(20, TimeUnit.SECONDS);
		} catch (final TimeoutException e) {
			getLogger().error("ErdFri: FRI session did not reach a ready state -- " + e.getLocalizedMessage());
			_friSession.close();
			_medflange.setLEDRed(false);
			return;
		}

		getLogger().info("ErdFri: FRI connection established, jitter="
				+ _friSession.getFRIChannelInformation().getJitter());

		_jointOverlay = new FRIJointOverlay(_friSession, ClientCommandMode.POSITION);

		_medflange.setLEDRed(false);
		_medflange.setLEDGreen(true);
		// The user button still ends the overlay (unchanged from iiwa_ros2.java).
		BooleanIOCondition buttonPressed = new BooleanIOCondition(_medflange.getInput("UserButton"), true);
		_lbr.move(_positionHold.addMotionOverlay(_jointOverlay).breakWhen(buttonPressed));

		_medflange.setLEDGreen(false);
		_medflange.setLEDRed(true);
		_friSession.close();
		getLogger().info("ErdFri: FRI connection closed, application stopped.");
	}

	@Override
	public void dispose() {
		if (_friSession != null) {
			try {
				_friSession.close();
			} catch (Exception e) {
				// already closed from run(); nothing more to do.
			}
		}
		super.dispose();
	}

	public static void main(final String[] args) {
		final ErdFri app = new ErdFri();
		app.runApplication();
	}
}
