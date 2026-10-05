"""Recorded UR10 RTDE data to the consumer's measured-state /3 contract."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from .contract import load_reference, differentiation_from_reference, real_baselines, real_contract
from .convert import ur10_dataset_frame, write_real_dataset
from .identification import MotorSideFit
from .planning import load_plan
from .validate import validate_dataset


def convert_ur(config, root: Path, reference_path):
    from elastic_sim.assets import AssetRegistry
    from .convert_cli import _refine_window
    n = len(config.joint_order); dt = 1 / config.rate_hz
    bundle = load_plan(root / 'plan')
    identification = json.loads((root / 'identification.json').read_text())
    if identification['config_digest'] != bundle.config_digest:
        raise ValueError('identification and recording config digests differ')
    raw = pd.read_parquet(root / 'raw.parquet')
    events = json.loads((root / 'events.json').read_text())
    alignment = json.loads((root / 'rtde_alignment.json').read_text())
    reference = load_reference(reference_path, robot=config.robot, joint_order=config.joint_order)
    diff = differentiation_from_reference(reference,rate_real=config.rate_hz,probe_top_real=0.,reference_path=reference_path)
    checks = {}; rows = []; bag_digests = {}; plan_digests = {}
    for segment in bundle.segments:
        if segment.kind != 'excitation': continue
        sid = segment.segment_id
        starts = [e for e in events if e['segment_id']==sid and e['kind'].endswith('_start')]
        ends = [e for e in events if e['segment_id']==sid and e['kind'].endswith('_end')]
        plan_digests[sid] = segment.digest
        if not starts or not ends:
            checks[sid] = {'ok':False,'reason':'missing bracketing events','samples':0}; continue
        bag_digests[sid] = starts[0]['plan_digest']
        lo, hi = np.searchsorted(raw['bag_stamp_ns'],[starts[0]['stamp_ns'],ends[-1]['stamp_ns']])
        lo, hi = _refine_window(raw,(lo,hi),segment.trajectory.position,n,pad_samples=round(.5/dt))
        frame = raw.iloc[lo:hi].copy()
        gaps = [g for g in alignment['gaps'] if frame['t'].iloc[0] <= g['timestamp'] <= frame['t'].iloc[-1]]
        allowed = all(g['class']=='controller_skipped' for g in gaps) and config.hardware=='ursim'
        valid = not gaps or allowed
        checks[sid] = {'ok':valid,'samples':len(frame),'gaps':gaps,'interpolated_controller_cycles':len(gaps) if allowed else 0}
        if not valid: continue
        ticks = np.rint((frame['t'].to_numpy()-frame['t'].iloc[0])/dt).astype(int)
        if np.any(np.diff(ticks)<=0):
            checks[sid].update(ok=False,reason='duplicate or reordered RTDE timestamp'); continue
        grid = np.arange(ticks[-1]+1)
        # L2-only controller skips are explicit synthetic interpolation. Reader
        # gaps and all real-hardware gaps always exclude the segment above.
        sampled = pd.DataFrame({c:np.interp(grid,ticks,frame[c]) for c in frame if np.issubdtype(frame[c].dtype,np.number)})
        sampled['t'] = grid * dt; sampled['bag'] = sid; sampled['split'] = 'test'; rows.append(sampled)
    if not rows:
        (root/'validation.json').write_text(json.dumps({'ok':False,'segments':checks,'rtde_alignment':alignment},indent=2))
        raise ValueError('no valid UR10 excitation segments')
    windowed = pd.concat(rows,ignore_index=True)
    motor = identification['motor_fit']
    fit = MotorSideFit(**{k:np.asarray(v,dtype=float) if isinstance(v,list) and k!='j_m_source' else v for k,v in motor.items()})
    source = identification['tau_source']
    frame = ur10_dataset_frame(windowed,n_dof=n,k_tau=np.asarray(identification['k_tau']),motor_fit=fit,
                               epsilon=identification['epsilon'],sg_window=diff['sg_window'],sg_poly=diff['sg_poly'],
                               time_step=dt,tau_source_column=source)
    baselines = real_baselines(frame,asset=AssetRegistry.for_repository().load(config.description.sim_asset),n_dof=n,
                               time_step=dt,sg_window=diff['sg_window'],sg_poly=diff['sg_poly'])
    manifest = {'schema_version':2,'n_dof':n,'joint_names':list(config.joint_order),'split':'test','sample_time_step':dt,
                'signals':{'target':'link_torque','target_source':'estimated','target_kind':'motor_current_proxy'},
                'differentiation':diff,'identification':'../identification.json','tau_source':source,
                'position_side':'unknown pending E-ur-4','source':'synthetic_mock' if config.hardware!='real' else 'real'}
    contract = real_contract(manifest,n_dof=n,target_instrument='ur10_current_inertia_friction_proxy',
            target_semantics='K_tau * actual_current - J_m * SG(actual_qd derivative) - fitted motor friction [Nm]',
            q_side='unknown',dq_side='unknown',dq_source='actual_qd',tau_instrument=source,
            controller={'location':'drive','vendor':'ur_servoj','model_free':False,'reference':'jtc_quintic',
                        'servoj_gain':config.connection.servoj_gain}, differentiation=diff,baselines=baselines,
            real_block={'robot_id':config.robot,'software_version':config.connection.software_version,
                        'position_side':'pending E-ur-4','identification':identification},friction_split='all_motor')
    contract['reference_synthetic'] = bool(reference.get('synthetic',False))
    output, _, contract_path = write_real_dataset(frame,manifest,contract,root/'dataset'/f'{config.robot}.parquet')
    arrays = config.limits.as_arrays(config.joint_order)
    validation = validate_dataset(frame,n_dof=n,nominal_dt=dt,position_lower=arrays['position_lower'],
            position_upper=arrays['position_upper'],quantization=1e-7,tau_source=source,
            ft_source='ur10_current_inertia_friction_proxy',effort_limit=arrays['effort'],
            bag_digests=bag_digests,plan_digests=plan_digests)
    validation.update(segments=checks,rtde_alignment=alignment,reference_synthetic=contract['reference_synthetic'])
    validation['status'] = 'synthetic' if contract['reference_synthetic'] or config.hardware!='real' else ('valid' if validation['ok'] else 'invalid')
    (root/'validation.json').write_text(json.dumps(validation,indent=2))
    return {'ok':validation['ok'],'validation_ok':validation['ok'],'status':validation['status'],
            'dataset':str(output),'contract':str(contract_path),'samples':len(frame)}
