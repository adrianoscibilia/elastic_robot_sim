"""Recorded UR10 RTDE data to the consumer's measured-state /3 contract."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from .bagio import (data_topic_losses, losses_in_window, robot_clock_samples, segment_completeness,
                    segment_skip_verdict, segment_skips)
from .contract import contract_source, differentiation_from_reference, real_baselines, real_contract
from .convert import ur10_dataset_frame, write_real_dataset
from .identification import MotorSideFit
from .planning import load_plan
from .validate import (check_envelopes, check_identification_freshness_ur, check_noise, check_reference_tracking,
                       check_session_health_ur, validate_dataset)


def _motor_fit(identification):
    motor = identification['motor_fit']
    return MotorSideFit(**{k: np.asarray(v, dtype=float) if isinstance(v, list) and k != 'j_m_source' else v
                           for k, v in motor.items()})


def convert_ur(config, root: Path, reference):
    from elastic_sim.assets import AssetRegistry
    from .convert_cli import (_finish_validation, _prefixed, _reference_block, _refine_window,
                              controller_state_window, excitation_windows, load_losses, sign_convention_check,
                              standstill_holds, tracking_for_segment)
    from .identify_cli import MAX_ANCHOR_INTERVAL_S, anchor_intervals, robot_clock_ratio, window_clock_ratio
    reference_path = reference['_path']
    n = len(config.joint_order); dt = 1 / config.rate_hz
    bundle = load_plan(root / 'plan')
    identification = json.loads((root / 'identification.json').read_text())
    if identification['config_digest'] != bundle.config_digest:
        raise ValueError('identification and recording config digests differ')
    if identification.get('ok') is False:
        raise ValueError(f"identification has invalid segments: {identification.get('invalid_segments')}")
    raw = pd.read_parquet(root / 'raw.parquet')
    events = json.loads((root / 'events.json').read_text())
    alignment = json.loads((root / 'rtde_alignment.json').read_text())
    losses = load_losses(root)
    clock_ratio = robot_clock_ratio(config, root)
    diff = differentiation_from_reference(reference, rate_real=config.rate_hz, probe_top_real=0., reference_path=reference_path)
    segments = [s for s in bundle.segments if s.kind == 'excitation']
    bounds = excitation_windows(raw, events, segments)
    stamp_ns = raw['stamp_ns'].to_numpy() if 'stamp_ns' in raw else raw['bag_stamp_ns'].to_numpy()
    checks = {}; rows = []; bag_digests = {}; plan_digests = {}; session = {}; tracking = {}
    for segment in segments:
        sid = segment.segment_id
        plan_digests[sid] = segment.digest
        if sid not in bounds:
            checks[sid] = {'ok': False, 'reason': 'missing bracketing events', 'samples': 0}; continue
        bag_digests[sid] = next(e['plan_digest'] for e in events if e['segment_id'] == sid and e['kind'].endswith('_start'))
        lo, hi, lo_ns, hi_ns = bounds[sid]
        bracketed = raw.iloc[lo:hi]
        complete = segment_completeness(robot_clock_samples(bracketed), len(segment.trajectory.time),
                                        clock_ratio=window_clock_ratio(bracketed, clock_ratio, hardware=config.hardware))
        data_losses = data_topic_losses(losses, lo_ns, hi_ns)
        if config.hardware == 'mock':  # no RTDE: the mock frame is built on the driver's own clock
            anchors = {'ok': True, 'max_anchor_interval_s': None}
        else:
            anchors = anchor_intervals(alignment.get('anchor_timestamps', []), [segment],
                                       {sid: bracketed})[sid]
        lo, hi = _refine_window(raw, (lo, hi), segment.trajectory.position, n, pad_samples=round(.5 / dt))
        frame = raw.iloc[lo:hi].copy()
        gaps = [g for g in alignment['gaps'] if frame['t'].iloc[0] <= g['timestamp'] <= frame['t'].iloc[-1]]
        allowed = all(g['class'] == 'controller_skipped' for g in gaps) and config.hardware == 'ursim'
        valid = (not gaps or allowed) and complete['ok'] and data_losses == 0 and anchors['ok']
        checks[sid] = {'ok': valid, 'samples': len(frame), 'gaps': gaps,
                       'interpolated_controller_cycles': len(gaps) if allowed else 0, 'completeness': complete,
                       'data_topic_losses': data_losses, 'data_topic_skips': segment_skips(losses, lo_ns, hi_ns), 'skip_verdict': segment_skip_verdict(losses, lo_ns, hi_ns), 'losses_by_topic': losses_in_window(losses, lo_ns, hi_ns),
                       'max_anchor_interval_s': anchors['max_anchor_interval_s'],
                       'max_anchor_interval_limit_s': MAX_ANCHOR_INTERVAL_S, 'duration_s': (hi_ns - lo_ns) * 1e-9}
        if not valid: continue
        ticks = np.rint((frame['t'].to_numpy() - frame['t'].iloc[0]) / dt).astype(int)
        if np.any(np.diff(ticks) <= 0):
            checks[sid].update(ok=False, reason='duplicate or reordered RTDE timestamp'); continue
        session[sid] = bracketed
        bracket_stamps = stamp_ns[bounds[sid][0]:bounds[sid][1]]
        tracking[sid] = tracking_for_segment(
            segment, base_ns=lo_ns,
            commanded=None if config.hardware == 'mock' else (bracket_stamps, _prefixed(bracketed, 'commanded_position', n)),
            jtc=controller_state_window(root, lo_ns, hi_ns, n), measured=(bracket_stamps, _prefixed(bracketed, 'q', n)))
        grid = np.arange(ticks[-1] + 1)
        # L2-only controller skips are explicit synthetic interpolation. Reader
        # gaps and all real-hardware gaps always exclude the segment above.
        sampled = pd.DataFrame({c: np.interp(grid, ticks, frame[c]) for c in frame if np.issubdtype(frame[c].dtype, np.number)})
        sampled['t'] = grid * dt; sampled['bag'] = sid; sampled['split'] = 'test'; rows.append(sampled)
    if not rows:
        (root / 'validation.json').write_text(json.dumps({'ok': False, 'segments': checks, 'rtde_alignment': alignment}, indent=2))
        raise ValueError('no valid UR10 excitation segments')
    windowed = pd.concat(rows, ignore_index=True)
    fit = _motor_fit(identification)
    source = identification['tau_source']
    k_tau = np.asarray(identification['k_tau'])

    def dataset_frame(frame):
        return ur10_dataset_frame(frame, n_dof=n, k_tau=k_tau, motor_fit=fit, epsilon=identification['epsilon'],
                                  sg_window=diff['sg_window'], sg_poly=diff['sg_poly'], time_step=dt,
                                  tau_source_column=source)

    frame = dataset_frame(windowed)
    baselines = real_baselines(frame, asset=AssetRegistry.for_repository().load(config.description.sim_asset), n_dof=n,
                               time_step=dt, sg_window=diff['sg_window'], sg_poly=diff['sg_poly'])

    holds = standstill_holds(config, root, bundle)
    sign = sign_convention_check(config, [k_tau[None, :] * _prefixed(h, 'i_act', n) for h in holds], holds)
    noise = check_noise([dataset_frame(h) for h in holds], frame, n_dof=n,
                        source='identification/raw.parquet identify_hold_* (first 0.5 s dropped)')
    temperatures = pd.concat(list(session.values()))[[f'joint_temperatures{j}' for j in range(n)]].mean().to_numpy()
    freshness = check_identification_freshness_ur(identification, run_config_digest=bundle.config_digest,
                                                  run_temperature_c=temperatures)

    manifest = {'schema_version': 2, 'n_dof': n, 'joint_names': list(config.joint_order), 'split': 'test', 'sample_time_step': dt,
                'signals': {'target': 'link_torque', 'target_source': 'estimated', 'target_kind': 'motor_current_proxy'},
                'differentiation': diff, 'identification': '../identification.json', 'tau_source': source,
                'position_side': 'unknown pending E-ur-4', 'source': contract_source(config.hardware)}
    contract = real_contract(manifest, hardware=config.hardware, n_dof=n, target_instrument='ur10_current_inertia_friction_proxy',
            target_semantics='K_tau * actual_current - J_m * SG(actual_qd derivative) - fitted motor friction [Nm]',
            q_side='unknown', dq_side='unknown', dq_source='actual_qd', tau_instrument=source,
            controller={'location': 'drive', 'vendor': 'ur_servoj', 'model_free': False, 'reference': 'jtc_quintic',
                        'servoj_gain': config.connection.servoj_gain}, differentiation=diff, baselines=baselines,
            real_block={'robot_id': config.robot, 'software_version': config.connection.software_version,
                        'safety_vendor_checksum': config.safety.vendor_checksum,
                        'position_side': 'pending E-ur-4', 'identification': identification,
                        'reference': _reference_block(reference),
                        'rate_mismatch': not diff['cutoff_matched']}, friction_split='all_motor')
    contract['reference_synthetic'] = bool(reference.get('synthetic', False))
    output, _, contract_path = write_real_dataset(frame, manifest, contract, root / 'dataset' / f'{config.robot}.parquet')
    arrays = config.limits.as_arrays(config.joint_order)
    validation = validate_dataset(frame, n_dof=n, nominal_dt=dt, position_lower=arrays['position_lower'],
            position_upper=arrays['position_upper'], quantization=1e-7, tau_source=source,
            ft_source='ur10_current_inertia_friction_proxy', effort_limit=arrays['effort'],
            bag_digests={k: v for k, v in bag_digests.items() if k in session}, plan_digests=plan_digests,
            session_health=check_session_health_ur(session), identification_freshness=freshness,
            reference_tracking=check_reference_tracking(tracking, commanded_report_only=True),
            envelopes=check_envelopes(frame, n_dof=n, effort_limit=arrays['effort'], sg_window=diff['sg_window'],
                                      sg_poly=diff['sg_poly'], time_step=dt),
            noise=noise, sign_convention=sign, hardware=config.hardware, robot=config.robot)
    validation.update(segments=checks, rtde_alignment={k: v for k, v in alignment.items() if k != 'anchor_timestamps'})
    result = _finish_validation(config, root, validation, contract, diff, events,
                                {'dataset': str(output), 'contract': str(contract_path), 'samples': len(frame)})
    return result
