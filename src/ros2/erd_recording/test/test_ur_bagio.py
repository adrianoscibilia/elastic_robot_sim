import numpy as np
import pandas as pd
import pytest
from erd_recording.ur_bagio import align_rtde


def fixture():
    t = 10 + np.arange(100) * .008
    q = np.column_stack([np.sin(t * (i+1)) for i in range(6)])
    f = pd.DataFrame({f'actual_q{i}':q[:,i] for i in range(6)})
    f['timestamp'] = t
    return t, q, f


def test_gap_classes_distinguish_reader_from_controller():
    t,q,f = fixture()
    sidecar = f.drop([30,60]).reset_index(drop=True)
    keep = np.arange(100) != 60
    stamps = np.rint((t[keep]+1000)*1e9).astype(np.int64)
    _,report = align_rtde(sidecar,q[keep],stamps)
    assert report['exact_match_fraction'] == 1
    assert report['reader_missing_cycles'] == 1
    assert report['controller_missing_cycles'] == 1
    assert [g['class'] for g in report['gaps']] == ['sidecar_only','controller_skipped']


def test_clock_uncertainty_cannot_certify_controller_skips():
    t,q,f = fixture()
    stamps = np.rint((t+1000+np.where(np.arange(100)%2,0,.006))*1e9).astype(np.int64)
    # 6 ms offsets spread by median would be only +/-3 ms: use 12 ms.
    stamps[::2] += 6_000_000
    with pytest.raises(ValueError,match='clock uncertainty'):
        align_rtde(f,q,stamps)


def test_a_standstill_value_cut_by_the_bag_end_is_not_an_anchor():
    """RR_19, rr19_ur10_sim: after the last segment the position flickers
    between two exact values; the bag stops first, so its last value occurs
    once in the driver but repeats in the sidecar."""
    t, q, f = fixture()
    a, b = q[-1].copy(), q[-1] + 1e-9
    tail = np.asarray([a, b, a, b, a, b, a, b])
    q_side = np.vstack([q, tail])
    side = pd.DataFrame({f'actual_q{i}': q_side[:, i] for i in range(6)})
    side['timestamp'] = 10 + np.arange(len(q_side)) * .008
    driver = np.vstack([q, tail[:2]])          # the bag stopped after one a, one b
    stamps = np.rint((10 + np.arange(len(driver)) * .008 + 1000) * 1e9).astype(np.int64)
    aligned, report = align_rtde(side, driver, stamps)
    assert report['exact_match_fraction'] > 0.9 and np.all(np.diff(aligned) > 0)
