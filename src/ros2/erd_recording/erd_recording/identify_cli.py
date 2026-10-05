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
from .bagio import iiwa_cycle_health
from .config import load_lab_config
from .contract import load_reference, differentiation_from_reference
from .identification import choose_ur_tau_source, fit_linear_per_joint, identify_motor_side, fit_sweep_friction
from .planning import load_plan


def block(frame, prefix, n):
    return frame[[f'{prefix}{i}' for i in range(n)]].to_numpy(dtype=float)


def windows(raw, events, segments):
    result = {}
    for segment in segments:
        starts = [e for e in events if e['segment_id'] == segment.segment_id and e['kind'].endswith('_start')]
        ends = [e for e in events if e['segment_id'] == segment.segment_id and e['kind'].endswith('_end')]
        if not starts or not ends or starts[0]['plan_digest'] != segment.digest:
            raise ValueError(f'missing event or changed plan digest: {segment.segment_id}')
        stamps = raw['bag_stamp_ns'].to_numpy()
        lo, hi = np.searchsorted(stamps, [starts[0]['stamp_ns'], ends[-1]['stamp_ns']])
        frame = raw.iloc[lo:hi].copy().reset_index(drop=True)
        if len(frame) < 2:
            raise ValueError(f'empty identification segment {segment.segment_id}')
        result[segment.segment_id] = frame
    return result


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
    selected = [s for s in bundle.identify_segments if s.kind in ('identify_hold','identify_excitation','identify_sweep')]
    frames = windows(raw, events, selected)
    n = len(config.joint_order)
    diff = differentiation_from_reference(reference, rate_real=config.rate_hz, probe_top_real=0., reference_path=reference['_path'])
    sg, poly, dt = diff['sg_window'], diff['sg_poly'], 1 / config.rate_hz
    asset = AssetRegistry.for_repository().load(config.description.sim_asset)
    pin, model, data = idn.build_model(asset)
    holds = pd.concat([frames[s.segment_id].iloc[round(.5 / dt):] for s in selected if s.kind == 'identify_hold'], ignore_index=True)
    q_hold = block(holds, 'q', n)
    gravity = np.asarray([pin.computeGeneralizedGravity(model, data, q).copy() for q in q_hold])
    result = {'schema': 'erd.identification/1', 'robot': config.robot, 'synthetic': config.hardware != 'real',
              'config_digest': bundle.config_digest, 'segments': {s.segment_id: {'rows': len(frames[s.segment_id]), 'digest': s.digest} for s in selected},
              'differentiation': diff, 'dynamics_model': 'simulation bare asset (RR_08 section 1)',
              'recorded_rows': len(raw), 'fit_rows': 0}
    excitations = [s for s in selected if s.kind == 'identify_excitation']
    if config.robot == 'kuka_lbr_iiwa_14_r820':
        health = {s.segment_id: iiwa_cycle_health(frames[s.segment_id]) for s in selected}
        result['cycle_health'] = health
        if config.hardware != 'mock' and any(not h['ok'] for h in health.values()):
            raise ValueError(f'identification contains lost iiwa cycles: {health}')
        frame = frames[excitations[0].segment_id]
        q = block(frame, 'q', n)
        dq = savgol_filter(q, sg, poly, deriv=1, delta=dt, axis=0)
        ddq = savgol_filter(q, sg, poly, deriv=2, delta=dt, axis=0)
        rnea = np.asarray([pin.rnea(model, data, qq, vv, aa).copy() for qq, vv, aa in zip(q, dq, ddq)])
        measured = block(frame, 'ft', n); residual = block(frame, 'tau', n) - measured
        sign = fit_linear_per_joint(block(holds, 'ft', n), gravity)
        table = []
        def corr_lag(a, b):
            a = a - a.mean(); b = b - b.mean()
            norm = np.linalg.norm(a) * np.linalg.norm(b)
            if norm == 0: return {'correlation': 0., 'lag_s': 0., 'degenerate': True}
            cc = correlate(a, b, mode='full', method='fft') / norm
            lags = correlation_lags(len(a), len(b), mode='full')
            allowed = np.abs(lags) <= round(.5 / dt)
            best = np.flatnonzero(allowed)[np.argmax(np.abs(cc[allowed]))]
            return {'correlation': float(cc[best]), 'lag_s': float(lags[best] * dt), 'degenerate': False}
        for j, name in enumerate(config.joint_order):
            denom = float(np.sqrt(np.mean(measured[:, j]**2)))
            table.append({'joint': name, 'rms_delta_over_measured': float(np.sqrt(np.mean(residual[:, j]**2))) / max(denom, 1e-15),
                          'measured_rms_nm': denom, 'ddq': corr_lag(residual[:, j], ddq[:, j]),
                          'rnea': corr_lag(residual[:, j], rnea[:, j]), 'gravity_slope': float(sign.slope[j]),
                          'offset_nm': float(sign.intercept[j]), 'gravity_r_squared': float(sign.r_squared[j])})
        result.update(fit_rows=len(frame), e_iiwa_1_2=table,
                      verdict='synthetic emulator semantics; physical CR-1 verdict requires T2' if config.hardware != 'real' else 'review per-joint E-iiwa-1/2 table before CR-1')
    else:
        alignment = json.loads((recording / 'rtde_alignment.json').read_text())
        if alignment['reader_missing_cycles'] or alignment['unclassified_cycles'] or (config.hardware == 'real' and alignment['gaps']):
            raise ValueError('identification has disallowed RTDE gaps')
        result['rtde_alignment'] = alignment
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
                                 holdout_mask=np.asarray(holdouts), rotor_inertia_nominal=np.asarray([4.,4.,1.7,.5,.5,.5]),
                                 friction_parameters=parameters)
        result.update(tau_source=choice['tau_source'], k_tau=k_tau, e_ur_1=choice['diagnostics'], motor_fit=asdict(fit),
                      friction_sweeps=sweep_table, friction_fit=friction, friction_split='all_motor', epsilon=.01,
                      holdout_segments=[excitations[0].segment_id,excitations[-1].segment_id],
                      fit_rows=sum(not x for x in holdouts), holdout_rows=sum(holdouts),
                      physical_validity='synthetic URSim fit' if config.hardware != 'real' else 'requires physical review')
    result = _serializable(result)
    (root / 'identification.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--config',required=True); parser.add_argument('--run-dir',required=True)
    args = parser.parse_args(); config = load_lab_config(args.config)
    path = (Path(args.config).resolve().parent / config.consumer.reference_contract).resolve()
    reference = load_reference(path,robot=config.robot,joint_order=config.joint_order); reference['_path'] = str(path)
    result = identify(config,Path(args.run_dir),reference)
    print(json.dumps({'ok':True,'recorded_rows':result['recorded_rows'],'fit_rows':result['fit_rows']}))

if __name__ == '__main__': main()
