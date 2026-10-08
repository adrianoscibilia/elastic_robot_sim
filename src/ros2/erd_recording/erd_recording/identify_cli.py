"""Clean numerical subprocess for recorded E-iiwa-1/2 and E-ur-1/3."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import correlate, correlation_lags, savgol_filter
from .env_guard import assert_environment
assert_environment(require_ros=False, require_pinocchio=True)
from elastic_sim.assets import AssetRegistry
from elastic_sim import identification as idn
from .bagio import (data_topic_losses, segment_skip_verdict, segment_skips, event_delivery, event_window, iiwa_cycle_health, losses_in_window,
                    robot_clock_samples, segment_completeness)
from .config import load_lab_config
from .contract import load_reference, differentiation_from_reference, resolve_reference
from .identification import choose_ur_tau_source, fit_linear_per_joint, identify_motor_side, fit_sweep_friction
from .planning import load_plan
from .validate import require_hardware


def block(frame, prefix, n):
    return frame[[f'{prefix}{i}' for i in range(n)]].to_numpy(dtype=float)


#: RR_12 B-3: E-iiwa-1's lag search half-width.
LAG_SEARCH_S = 0.05
#: RR_12 B-3: a joint's standstill gravity slope is testable only where its
#: gravity range across the poses is this many standstill sigmas ...
GRAVITY_RANGE_SIGMAS = 10.0
#: ... and the per-joint fit reaches this r^2.
GRAVITY_MIN_R_SQUARED = 0.9
#: RR_12 B-3: UR excitation/sweep segments need an exact actual_q anchor at
#: least this often (holds are exempt: constant actual_q has no unique anchor).
MAX_ANCHOR_INTERVAL_S = 0.1
FIT_INPUT_KINDS = ('identify_hold', 'identify_excitation', 'identify_sweep')


def stamp_column(raw):
    """RR_13 B-1: windows use publish-side stamps (``stamp_ns``: controller-
    manager update time / RTDE aligned to it) on the same clock as the event
    headers; ``bag_stamp_ns`` (recorder receive time) only for old raw files."""
    return 'stamp_ns' if 'stamp_ns' in raw else 'bag_stamp_ns'


def windows(raw, events, segments):
    """``{segment_id: frame}`` between each segment's start and end events,
    plus ``{segment_id: (start_ns, end_ns)}``."""
    result, bounds = {}, {}
    stamps = raw[stamp_column(raw)].to_numpy()
    for segment in segments:
        bracket = event_window(events, segment.segment_id)
        if bracket is None or bracket[0]['plan_digest'] != segment.digest:
            raise ValueError(f'missing event or changed plan digest: {segment.segment_id}')
        lo_ns, hi_ns = bracket[0]['stamp_ns'], bracket[1]['stamp_ns']
        lo, hi = np.searchsorted(stamps, [lo_ns, hi_ns])
        frame = raw.iloc[lo:hi].copy().reset_index(drop=True)
        if len(frame) < 2:
            raise ValueError(f'empty identification segment {segment.segment_id}')
        result[segment.segment_id] = frame
        bounds[segment.segment_id] = (int(lo_ns), int(hi_ns))
    return result, bounds


def robot_clock_ratio(config, run_dir):
    """Robot seconds per wall second: 1 except on URSim (RR_13 B-1), where
    the RTDE alignment measures it."""
    path = Path(run_dir) / 'rtde_alignment.json'
    if config.hardware != 'ursim' or not path.is_file():
        return 1.0
    return 1.0 / float(json.loads(path.read_text())['wall_seconds_per_robot_second'])


def window_clock_ratio(frame, run_ratio, *, hardware):
    """RR_15 refinement of D-4 (URSim only; RR_16 Q-3 gates it on
    ``hardware == 'ursim'``, not on the ratio, and every other value gets 1.0): the
    robot clock's rate inside this window -- RTDE robot time elapsed over
    the aligned wall time elapsed. URSim's rate drifts by about +-1 % between
    windows (0.926-0.944 on rr15_ur10_sim_c), enough to fail a 5 s hold
    against the whole-run ratio. Robot time, not a sample count: a window
    with missing robot samples or a hold cut short still fails."""
    if require_hardware(hardware) != 'ursim':
        return 1.0
    if 'timestamp' not in frame or 'stamp_ns' not in frame or len(frame) < 2:
        return run_ratio
    robot = float(frame['timestamp'].iloc[-1] - frame['timestamp'].iloc[0])
    wall = float(frame['stamp_ns'].iloc[-1] - frame['stamp_ns'].iloc[0]) * 1e-9
    return robot / wall if robot > 0 and wall > 0 else run_ratio


def segment_health(segments, frames, bounds, losses, *, hardware, clock_ratio=1.0):
    """RR_12 B-1/B-2 per fit-input segment: robot-clock completeness against
    the plan, losses on the two data topics, and the event-stamped duration."""
    table = {}
    for segment in segments:
        sid = segment.segment_id
        lo, hi = bounds[sid]
        complete = segment_completeness(robot_clock_samples(frames[sid]), len(segment.trajectory.time),
                                        clock_ratio=window_clock_ratio(frames[sid], clock_ratio, hardware=hardware))
        lost = data_topic_losses(losses, lo, hi)
        table[sid] = {**complete, 'kind': segment.kind, 'duration_s': (hi - lo) * 1e-9,
                      'planned_duration_s': float(segment.trajectory.duration),
                      'data_topic_losses': lost, 'data_topic_skips': segment_skips(losses, lo, hi), 'skip_verdict': segment_skip_verdict(losses, lo, hi), 'losses_by_topic': losses_in_window(losses, lo, hi),
                      'ok': complete['ok'] and lost == 0}
    return table


def lag_correlation(a, b, dt, *, max_lag_s=LAG_SEARCH_S):
    """E-iiwa-1 (RR_12 B-3): the lag of the strongest |correlation| within
    +-``max_lag_s``. A maximum on the bound is no estimate: it is reported as
    ``lag_at_bound: true`` with ``lag_s: null``."""
    a = a - a.mean(); b = b - b.mean()
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    bound = int(round(max_lag_s / dt))
    if norm == 0:
        return {'correlation': 0., 'lag_s': None, 'lag_at_bound': False, 'degenerate': True,
                'search_s': max_lag_s}
    cc = correlate(a, b, mode='full', method='fft') / norm
    lags = correlation_lags(len(a), len(b), mode='full')
    allowed = np.abs(lags) <= bound
    best = np.flatnonzero(allowed)[np.argmax(np.abs(cc[allowed]))]
    at_bound = abs(int(lags[best])) >= bound
    return {'correlation': float(cc[best]), 'lag_s': None if at_bound else float(lags[best] * dt),
            'lag_at_bound': bool(at_bound), 'degenerate': False, 'search_s': max_lag_s}


def detrended_sigma(frames, columns):
    """Standstill sigma per column: linear-detrended per hold, averaged over holds."""
    sigmas = []
    for frame in frames:
        values = frame[columns].to_numpy(dtype=float)
        t = np.arange(len(values), dtype=float)
        design = np.column_stack([t, np.ones_like(t)])
        coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
        sigmas.append(np.std(values - design @ coefficients, axis=0))
    return np.mean(sigmas, axis=0)


def gravity_testability(pose_gravity, sigma, slope, r_squared, joint_names):
    """RR_12 B-3: slope reported only where the joint's gravity range across
    the poses is >= 10x its standstill sigma and r^2 >= 0.9."""
    pose_gravity = np.asarray(pose_gravity, dtype=float)
    span = pose_gravity.max(axis=0) - pose_gravity.min(axis=0)
    table = []
    for j, name in enumerate(joint_names):
        range_ok = span[j] >= GRAVITY_RANGE_SIGMAS * sigma[j]
        fit_ok = r_squared[j] >= GRAVITY_MIN_R_SQUARED
        testable = bool(range_ok and fit_ok)
        reason = None if testable else (
            'not testable at these poses: gravity range ' f'{span[j]:.3g} N*m < {GRAVITY_RANGE_SIGMAS:g} sigma '
            f'({GRAVITY_RANGE_SIGMAS * sigma[j]:.3g})' if not range_ok else
            f'not testable at these poses: r^2 {r_squared[j]:.3g} < {GRAVITY_MIN_R_SQUARED}')
        table.append({'joint': name, 'gravity_range_nm': float(span[j]), 'standstill_sigma_nm': float(sigma[j]),
                      'r_squared': float(r_squared[j]), 'testable': testable,
                      'slope': float(slope[j]) if testable else None, 'reason': reason})
    return table


#: A sample is "moving" when any joint's |dq| exceeds this (rad/s).
MOVING_SPEED = 1e-3


def anchor_intervals(anchor_timestamps, segments, frames):
    """RR_12 B-3: per UR segment, the longest stretch of robot time without an
    exact actual_q anchor while the arm moves (the moving span's edges count
    as interval ends). A standing arm repeats actual_q, so the event window's
    still edges -- 0.17 s before the goal starts moving on URSim -- carry no
    anchor by construction."""
    anchors = np.asarray(anchor_timestamps, dtype=float)
    table = {}
    for segment in segments:
        f = frames[segment.segment_id]
        t = f['t'].to_numpy(dtype=float)
        speed_columns = [c for c in f.columns if c.startswith('dq') and c[2:].isdigit()]
        moving = np.abs(f[speed_columns].to_numpy(dtype=float)).max(axis=1) > MOVING_SPEED if speed_columns else None
        if moving is not None and moving.any():
            t0, t1 = float(t[moving][0]), float(t[moving][-1])
        else:
            t0, t1 = float(t[0]), float(t[-1])
        inside = anchors[(anchors >= t0) & (anchors <= t1)]
        worst = float(np.max(np.diff(np.concatenate([[t0], inside, [t1]]))))
        required = segment.kind in ('identify_excitation', 'identify_sweep', 'excitation', 'excitation_commissioning')
        table[segment.segment_id] = {'max_anchor_interval_s': worst, 'anchors': int(len(inside)),
                                     'required': required,
                                     'ok': (worst <= MAX_ANCHOR_INTERVAL_S) if required else True}
    return table


#: E-ur-3's nominal rotor-inertia prior (the `rotor_inertia_nominal` below).
UR_J_M_NOMINAL = (4., 4., 1.7, .5, .5, .5)


def _mock_ur_identification(n):
    """``hardware: mock``: no motor currents, so nothing to fit (E-ur-1/E-ur-3
    and the friction sweeps need them). Convert still needs a model, so the
    file carries labelled placeholders: K_tau 1 (it multiplies zero
    currents), the nominal J_m prior, zero friction."""
    zeros = [0.0] * n
    return {'tau_source': 'target_current', 'k_tau': [1.0] * n, 'epsilon': .01, 'friction_split': 'all_motor',
            'e_ur_1': {'status': 'not_applicable', 'reason': 'mock: no motor currents'},
            'motor_fit': {'j_m': list(UR_J_M_NOMINAL[:n]), 'viscous': zeros, 'coulomb': zeros,
                          'base_parameters': [], 'residual_rms': None, 'holdout_rms': None,
                          'j_m_source': ['prior'] * n, 'j_m_sensitivity': None,
                          'stribeck': zeros, 'stribeck_velocity': [1.0] * n},
            'friction_sweeps': [], 'friction_fit': [], 'holdout_segments': [], 'fit_rows': 0, 'holdout_rows': 0,
            'mock_placeholders': {'k_tau': 'placeholder 1.0: mock has no currents to fit',
                                  'motor_fit': 'nominal J_m prior, zero friction: no fit without currents'},
            'physical_validity': 'none: mock placeholders'}


def _serializable(value):
    if isinstance(value, dict): return {k: _serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_serializable(v) for v in value]
    if isinstance(value, np.ndarray): return _serializable(value.tolist())
    if isinstance(value, (float, np.floating)) and not np.isfinite(value): return None
    if isinstance(value, np.generic): return value.item()
    return value


def identify(config, root, reference):
    bundle = load_plan(root / 'plan')
    recording = root / 'identification'
    raw = pd.read_parquet(recording / 'raw.parquet')
    events = json.loads((recording / 'events.json').read_text())
    selected = [s for s in bundle.identify_segments if s.kind in FIT_INPUT_KINDS]
    frames, bounds = windows(raw, events, selected)
    losses_path = recording / 'recorder_losses.json'
    losses = json.loads(losses_path.read_text()) if losses_path.is_file() else None
    health = segment_health(selected, frames, bounds, losses, hardware=config.hardware,
                                clock_ratio=robot_clock_ratio(config, recording))
    n = len(config.joint_order)
    diff = differentiation_from_reference(reference, rate_real=config.rate_hz, probe_top_real=0., reference_path=reference['_path'])
    sg, poly, dt = diff['sg_window'], diff['sg_poly'], 1 / config.rate_hz
    asset = AssetRegistry.for_repository().load(config.description.sim_asset)
    pin, model, data = idn.build_model(asset)
    hold_frames = [frames[s.segment_id].iloc[round(.5 / dt):] for s in selected if s.kind == 'identify_hold']
    holds = pd.concat(hold_frames, ignore_index=True)
    q_hold = block(holds, 'q', n)
    gravity = np.asarray([pin.computeGeneralizedGravity(model, data, q).copy() for q in q_hold])
    pose_gravity = [pin.computeGeneralizedGravity(model, data, block(f, 'q', n).mean(axis=0)).copy() for f in hold_frames]
    result = {'schema': 'erd.identification/1', 'robot': config.robot, 'synthetic': config.hardware != 'real',
              'config_digest': bundle.config_digest,
              'segments': {s.segment_id: {'rows': len(frames[s.segment_id]), 'digest': s.digest, **health[s.segment_id]}
                           for s in selected},
              'differentiation': diff, 'reference': {'path': diff['derived_from'].split('#')[0],
                                                     'sha256': reference['_sha256'], 'asset': reference.get('_asset'),
                                                     'arm_check': reference.get('_arm_check')},
              'dynamics_model': 'simulation bare asset (RR_08 section 1)',
              'recorded_rows': len(raw), 'fit_rows': 0, 'event_delivery': event_delivery(events),
              'recorder_losses': _loss_summary(losses)}
    excitations = [s for s in selected if s.kind == 'identify_excitation']
    if config.robot == 'kuka_lbr_iiwa_14_r820':
        health = {s.segment_id: iiwa_cycle_health(frames[s.segment_id]) for s in selected}
        result['cycle_health'] = health
        if config.hardware != 'mock' and any(not h['ok'] for h in health.values()):
            raise ValueError(f'identification contains lost iiwa cycles: {health}')
        frame = frames[excitations[0].segment_id]
        q = block(frame, 'q', n)
        sigma = detrended_sigma(hold_frames, [f'ft{j}' for j in range(n)])
        dq = savgol_filter(q, sg, poly, deriv=1, delta=dt, axis=0)
        ddq = savgol_filter(q, sg, poly, deriv=2, delta=dt, axis=0)
        rnea = np.asarray([pin.rnea(model, data, qq, vv, aa).copy() for qq, vv, aa in zip(q, dq, ddq)])
        measured = block(frame, 'ft', n); residual = block(frame, 'tau', n) - measured
        sign = fit_linear_per_joint(block(holds, 'ft', n), gravity)
        table = []
        testability = gravity_testability(pose_gravity, sigma, sign.slope, sign.r_squared, config.joint_order)
        for j, name in enumerate(config.joint_order):
            denom = float(np.sqrt(np.mean(measured[:, j]**2)))
            table.append({'joint': name, 'rms_delta_over_measured': float(np.sqrt(np.mean(residual[:, j]**2))) / max(denom, 1e-15),
                          'measured_rms_nm': denom, 'ddq': lag_correlation(residual[:, j], ddq[:, j], dt),
                          'rnea': lag_correlation(residual[:, j], rnea[:, j], dt),
                          'gravity_slope': testability[j]['slope'], 'gravity_testable': testability[j]['testable'],
                          'gravity_reason': testability[j]['reason'], 'gravity_range_nm': testability[j]['gravity_range_nm'],
                          'standstill_sigma_nm': testability[j]['standstill_sigma_nm'],
                          'offset_nm': float(sign.intercept[j]), 'gravity_r_squared': float(sign.r_squared[j])})
        result.update(fit_rows=len(frame), e_iiwa_1_2=table,
                      verdict='synthetic emulator semantics; physical CR-1 verdict requires T2' if config.hardware != 'real' else 'review per-joint E-iiwa-1/2 table before CR-1')
    else:
        alignment = json.loads((recording / 'rtde_alignment.json').read_text())
        if alignment['reader_missing_cycles'] or alignment['unclassified_cycles'] or (config.hardware == 'real' and alignment['gaps']):
            raise ValueError('identification has disallowed RTDE gaps')
        result['rtde_alignment'] = {k: v for k, v in alignment.items() if k != 'anchor_timestamps'}
        if config.hardware == 'mock':
            result.update(_mock_ur_identification(n))
        else:
            _identify_ur(config, result, bundle, selected, frames, alignment, excitations, holds, hold_frames,
                         gravity, pose_gravity, asset, n, sg, poly, dt)
    invalid = sorted(sid for sid, entry in result['segments'].items() if not entry['ok'])
    result['invalid_segments'] = invalid
    result['ok'] = not invalid
    result = _serializable(result)
    (root / 'identification.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    return result


def _identify_ur(config, result, bundle, selected, frames, alignment, excitations, holds, hold_frames,
                 gravity, pose_gravity, asset, n, sg, poly, dt):
    """E-ur-1/E-ur-3 and the friction sweeps (hardware with motor currents)."""
    intervals = anchor_intervals(alignment.get('anchor_timestamps', []), selected, frames)
    result['anchor_intervals'] = intervals
    for sid, entry in intervals.items():
        result['segments'][sid]['max_anchor_interval_s'] = entry['max_anchor_interval_s']
        if not entry['ok']:
            result['segments'][sid]['ok'] = False
    frame = frames[excitations[1].segment_id]
    choice = choose_ur_tau_source(target_moment=block(frame,'target_moment',n),
                joint_control_output=block(frame,'joint_control_output',n), target_current=block(frame,'target_current',n),
                actual_current=block(frame,'actual_current',n),
                tracking_error=block(frame,'target_q',n)-block(frame,'q',n),
                tracking_error_rate=block(frame,'target_qd',n)-block(frame,'dq',n),
                standstill_actual_current=block(holds,'actual_current',n), standstill_gravity=gravity)
    k_tau = choice['k_tau']; friction = []; sweep_table = []
    for j in range(n):
        speeds, torques = [], []
        for k in range(6):
            values = []
            for direction in ('positive','negative'):
                sid = f'identify_sweep_{j}_{k}_{direction}'
                segment = bundle.segment(sid); f = frames[sid]
                elapsed = f['t'].to_numpy() - f['t'].iloc[0]
                meta = segment.trajectory.metadata
                plateau = f[(elapsed >= meta['plateau_start_s'] + .05) & (elapsed < meta['plateau_end_s'] - .05)]
                if len(plateau) < 5: raise ValueError(f'{sid}: insufficient plateau rows')
                values.append((float(plateau[f'dq{j}'].mean()), float(k_tau[j] * plateau[f'i_act{j}'].mean())))
                sweep_table.append({'segment': sid, 'plateau_rows':len(plateau), 'speed_rad_s':values[-1][0],
                                    'torque_nm': values[-1][1], 'temperature_before_c': float(f[f'joint_temperatures{j}'].iloc[0]),
                                    'temperature_after_c': float(f[f'joint_temperatures{j}'].iloc[-1])})
            speeds.append((values[0][0] - values[1][0]) / 2)
            torques.append((values[0][1] - values[1][1]) / 2)
        friction.append(fit_sweep_friction(speeds, torques))
    parameters = {key: np.asarray([f[key] for f in friction]) for key in ('viscous','coulomb','stribeck','stribeck_velocity')}
    qs, dqs, ddqs, currents, holdouts = [], [], [], [], []
    for i, segment in enumerate(excitations):
        f = frames[segment.segment_id]
        dq = block(f,'dq',n)
        qs.append(block(f,'q',n)); dqs.append(dq)
        ddqs.append(savgol_filter(dq,sg,poly,deriv=1,delta=dt,axis=0))
        currents.append(block(f,'i_act',n)); holdouts.extend([i in (0,len(excitations)-1)] * len(f))
    fit = identify_motor_side(asset, np.vstack(qs), np.vstack(dqs), np.vstack(ddqs), np.vstack(currents), k_tau,
                             holdout_mask=np.asarray(holdouts), rotor_inertia_nominal=np.asarray(UR_J_M_NOMINAL),
                             friction_parameters=parameters)
    sanity = choice['diagnostics'].get('standstill_sanity_check')
    if sanity is not None:
        ft_holds = [f.assign(**{f'kt_i_act{j}': k_tau[j] * f[f'i_act{j}'] for j in range(n)}) for f in hold_frames]
        sigma = detrended_sigma(ft_holds, [f'kt_i_act{j}' for j in range(n)])
        sanity['testability'] = gravity_testability(pose_gravity, sigma, np.asarray(sanity['slope']),
                                                    np.asarray(sanity.get('r_squared', [1.0] * n)), config.joint_order)
    result.update(tau_source=choice['tau_source'], k_tau=k_tau, e_ur_1=choice['diagnostics'], motor_fit=asdict(fit),
                  friction_sweeps=sweep_table, friction_fit=friction, friction_split='all_motor', epsilon=.01,
                  holdout_segments=[excitations[0].segment_id,excitations[-1].segment_id],
                  fit_rows=sum(not x for x in holdouts), holdout_rows=sum(holdouts),
                  physical_validity='synthetic URSim fit' if config.hardware != 'real' else 'requires physical review')


def _loss_summary(losses):
    """Per-topic counts without the stamp lists (those stay in recorder_losses.json)."""
    if not losses:
        return None
    return {'rosbag2_transport_lost': losses.get('rosbag2_transport_lost'),
            'attribution': losses.get('attribution'), 'startup_artefacts': losses.get('startup_artefacts'),
            'transport_events': losses.get('transport_events'), 'publisher_skips': losses.get('publisher_skips'),
            'unattributed': losses.get('unattributed'),
            'missing_by_topic': {t: e['missing'] for t, e in losses.get('topics', {}).items()},
            'source': 'identification/recorder_losses.json'}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--config',required=True); parser.add_argument('--run-dir',required=True)
    parser.add_argument('--reference-contract', default=None); parser.add_argument('--reference-sha256', default=None)
    args = parser.parse_args(); config = load_lab_config(args.config)
    path, sha = resolve_reference(args.config, config.consumer.reference_contract, config.consumer.reference_sha256,
                                  args.reference_contract, args.reference_sha256)
    reference = load_reference(path, robot=config.robot, joint_order=config.joint_order, expected_sha256=sha,
                               hardware=config.hardware, sim_asset=config.description.sim_asset)
    result = identify(config,Path(args.run_dir),reference)
    print(json.dumps({'ok':result['ok'],'recorded_rows':result['recorded_rows'],'fit_rows':result['fit_rows'],
                      'invalid_segments':result['invalid_segments']}))
    # RR_12 B-1: an incomplete or lossy fit-input segment fails the stage;
    # identification.json stays on disk as the evidence.
    return 0 if result['ok'] else 1

if __name__ == '__main__': raise SystemExit(main())
