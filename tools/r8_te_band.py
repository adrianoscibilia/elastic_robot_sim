#!/usr/bin/env python3
"""identified_filter/04 A2: is the simulated transmission error (TE) realistic?  Real vs simulated ripple amplitude.

The TE sits at the phase ``order x theta(t)`` (``02`` Sec 6: iiwa 320 = 2 x 160 per rad, UR10 200 = 2 x 100).
Order tracking follows it: each signal is demodulated by that phase and low-passed over ``WINDOW_S``, so
``2 |z(t)|`` is the amplitude of the ripple at the instantaneous ``f_TE = order |dq| / 2 pi``.  The same analysis
runs on the real files and on the probe-off and real-like test bags of the production file (no training, no filter):

* in ``ft`` [N·m] and in the SG-differentiated motor-side ``ddq`` [rad/s^2];
* per speed bin, on the rows where ``f_TE`` lies in ``[F_LOW, 0.4 x sample rate]`` and the speed is nearly constant
  over the window, the two conditions under which the tracked line is resolved;
* with a floor: the same demodulation at off-orders (``OFF_ORDERS`` x order).  Broadband content that is not TE
  reads there as well, so the TE amplitude is ``sqrt(max(a^2 - floor^2, 0))``, and a line at the floor is
  an upper bound, reported as such.

The ratio is real / simulated, median over the speed bins both sides populate.  The UR10 row is report-only: its
target is a current proxy (``04`` A2).

**This calibrates a simulator parameter from real data, as the sim2real track does.**  It never touches a model's
training or selection.

    uv run python tools/r8_te_band.py
    uv run python tools/r8_te_band.py --platform iiwa_drive --no-report   # look, write nothing
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_REPO / "src")]

import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import uniform_filter1d

from elastic_sim.diagnostics import savitzky_golay_acceleration

RESULTS = _REPO / "REFACTOR_SPECS" / "identified_filter" / "results"
CAMPAIGN = _REPO / "config" / "campaigns" / "r8_campaign.yaml"
REAL = {"iiwa_drive": {"iiwa_real_all_01": "data/real_robot/kuka_lbr_iiwa_14_r820/20261014/iiwa_real_all_01/dataset/"
                                           "kuka_lbr_iiwa_14_r820.parquet",
                       "iiwa_fast_all_01": "data/real_robot/kuka_lbr_iiwa_14_r820/20261014/iiwa_fast_all_01/dataset/"
                                           "kuka_lbr_iiwa_14_r820.parquet"},
        "ur10": {"ur_A_all_02": "data/real_robot/ur10_cb3/20261021/ur_A_all_02/dataset/ur10_cb3.parquet",
                 "ur_B_all_02": "data/real_robot/ur10_cb3/20261021/ur_B_all_02/dataset/ur10_cb3.parquet"}}
ORDER = {"iiwa_drive": 320.0, "ur10": 200.0}
N_DOF = {"iiwa_drive": 7, "ur10": 6}
MAIN = {"iiwa_drive": {0: "A1", 1: "A2", 3: "A4"}, "ur10": {0: "pan", 1: "lift", 2: "elbow"}}
REPORT_ONLY = {"ur10"}
SIM_REGIMES = ("off", "real_like")
WINDOW_S = 0.5
F_LOW = 8.0
OFF_ORDERS = (0.7, 0.85, 1.15, 1.3)
SPEED_BINS = np.array([0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.2])
#: The speed may vary by this share over the window: otherwise the line smears out of the tracked band.
SPEED_STEADINESS = 0.25
MIN_ROWS_S = 1.0
SCAN_ORDERS = (60.0, 720.0, 10.0)


def tracked_amplitude(x: np.ndarray, phase: np.ndarray, window: int) -> np.ndarray:
    """``2 |LP[x exp(-i phase)]|``: the amplitude of the line at ``phase`` over a moving window."""
    centred = x - uniform_filter1d(x, window, mode="nearest")
    carrier = centred * np.exp(-1j * phase)
    return 2.0 * np.abs(uniform_filter1d(carrier.real, window, mode="nearest")
                        + 1j * uniform_filter1d(carrier.imag, window, mode="nearest"))


def bag_rows(q, dq, ddq, ft, step: float, order: float, source: str, joints: dict) -> list[dict]:
    """One row per joint, signal and speed bin of one continuous recording."""
    window = max(3, int(round(WINDOW_S / step)))
    rate = 1.0 / step
    rows = []
    for j, name in joints.items():
        speed = np.abs(dq[:, j])
        mean_speed = uniform_filter1d(speed, window, mode="nearest")
        steady = np.abs(speed - mean_speed) <= SPEED_STEADINESS * np.maximum(mean_speed, 1e-9)
        f_te = order * mean_speed / (2.0 * np.pi)
        valid = steady & (f_te >= F_LOW) & (f_te <= 0.4 * rate)
        bins = np.digitize(mean_speed, SPEED_BINS)
        for signal, x in (("ft", ft[:, j]), ("ddq", ddq[:, j])):
            line = tracked_amplitude(x, order * q[:, j], window)
            floor = np.sqrt(np.mean([tracked_amplitude(x, factor * order * q[:, j], window) ** 2
                                     for factor in OFF_ORDERS], axis=0))
            for b in range(1, len(SPEED_BINS)):
                chosen = valid & (bins == b)
                if chosen.sum():
                    rows.append({"source": source, "joint": name, "signal": signal, "bin": b, "rows": int(chosen.sum()),
                                 "seconds": float(chosen.sum() * step), "power": float(np.sum(line[chosen] ** 2)),
                                 "floor_power": float(np.sum(floor[chosen] ** 2))})
    return rows


def real_rows(platform: str, sg_window: int, sg_poly: int) -> pd.DataFrame:
    n = N_DOF[platform]
    columns = ["t", "bag"] + [f"{p}{i}" for p in ("q", "dq", "ft") for i in range(n)]
    rows = []
    for file_id, path in REAL[platform].items():
        for _, group in pd.read_parquet(_REPO / path, columns=columns).groupby("bag", sort=False):
            step = float(np.median(np.diff(group["t"])))
            q, dq, ft = (group[[f"{p}{i}" for i in range(n)]].to_numpy(float) for p in ("q", "dq", "ft"))
            ddq = savitzky_golay_acceleration(dq, step, window=sg_window, order=sg_poly)
            rows += bag_rows(q, dq, ddq, ft, step, ORDER[platform], f"real:{file_id}", MAIN[platform])
    return pd.DataFrame(rows)


def sim_rows(platform: str, stem: Path, sg_window: int, sg_poly: int) -> pd.DataFrame:
    n = N_DOF[platform]
    columns = ["t", "bag", "split", "probe_regime"] + [f"{p}{i}" for p in ("q", "dq", "ft") for i in range(n)]
    frame = pd.read_parquet(stem.with_suffix(".parquet"), columns=columns, filters=[("split", "==", "test")])
    rows = []
    for _, group in frame.groupby("bag", sort=False):
        regime = str(group["probe_regime"].iloc[0])
        if regime not in SIM_REGIMES:
            continue
        step = float(np.median(np.diff(group["t"])))
        q, dq, ft = (group[[f"{p}{i}" for i in range(n)]].to_numpy(float) for p in ("q", "dq", "ft"))
        ddq = savitzky_golay_acceleration(dq, step, window=sg_window, order=sg_poly)
        rows += bag_rows(q, dq, ddq, ft, step, ORDER[platform], f"sim:{regime}", MAIN[platform])
    return pd.DataFrame(rows)


def te_amplitude(table: pd.DataFrame) -> pd.DataFrame:
    """Per source, joint, signal and speed bin: line RMS amplitude, the floor, and the floor-corrected TE amplitude."""
    keys = ["source", "joint", "signal", "bin"]
    out = table.groupby(keys, sort=False)[["rows", "seconds", "power", "floor_power"]].sum().reset_index()
    out["line"] = np.sqrt(out["power"] / out["rows"])
    out["floor"] = np.sqrt(out["floor_power"] / out["rows"])
    out["te"] = np.sqrt(np.maximum(out["line"] ** 2 - out["floor"] ** 2, 0.0))
    out["above_floor"] = out["line"] > 1.1 * out["floor"]
    return out.drop(columns=["power", "floor_power"])


def ratios(amplitudes: pd.DataFrame, real_prefix: str = "real:", sim_source: str = "sim:off") -> pd.DataFrame:
    """Real / simulated TE amplitude per file, joint and signal, over the speed bins with enough rows on both sides."""
    rows = []
    sim = amplitudes[amplitudes["source"] == sim_source].set_index(["joint", "signal", "bin"])
    for (source, joint, signal), part in amplitudes[amplitudes["source"].str.startswith(real_prefix)].groupby(
            ["source", "joint", "signal"], sort=False):
        common = [row for row in part.itertuples() if (joint, signal, row.bin) in sim.index
                  and row.seconds >= MIN_ROWS_S and sim.loc[(joint, signal, row.bin), "seconds"] >= MIN_ROWS_S]
        if not common:
            rows.append({"source": source, "joint": joint, "signal": signal, "bins": 0})
            continue
        real, simulated, bound = [], [], []
        for row in common:
            other = sim.loc[(joint, signal, row.bin)]
            # A line at the floor is an upper bound on the TE: take the floor as its value and say so.
            real.append(row.te if row.above_floor else row.floor)
            simulated.append(other["te"] if other["above_floor"] else other["floor"])
            bound.append(not row.above_floor or not other["above_floor"])
        rows.append({"source": source, "joint": joint, "signal": signal, "bins": len(common),
                     "real": float(np.median(real)), "sim": float(np.median(simulated)),
                     "ratio": float(np.median(np.asarray(real) / np.maximum(simulated, 1e-12))),
                     "ratio_low": float(np.min(np.asarray(real) / np.maximum(simulated, 1e-12))),
                     "ratio_high": float(np.max(np.asarray(real) / np.maximum(simulated, 1e-12))),
                     "bins_at_floor": int(np.sum(bound))})
    return pd.DataFrame(rows)


def order_scan(platform: str) -> pd.DataFrame:
    """Per real file and main joint, ``ft``: the tracked amplitude over orders, a guard on the ratio's premise.

    The ratio only reads the assumed order.  A real TE at another order would be invisible there and the ratio would
    say "smaller" when the truth is "elsewhere", so this reports the strongest order next to the assumed one.
    """
    n = N_DOF[platform]
    columns = ["t", "bag"] + [f"{p}{i}" for p in ("q", "dq", "ft") for i in range(n)]
    orders = np.arange(*SCAN_ORDERS)
    rows = []
    for file_id, path in REAL[platform].items():
        for j, joint in MAIN[platform].items():
            power, count = np.zeros(len(orders)), 0
            for _, group in pd.read_parquet(_REPO / path, columns=columns).groupby("bag", sort=False):
                step = float(np.median(np.diff(group["t"])))
                window = max(3, int(round(WINDOW_S / step)))
                dq = group[f"dq{j}"].to_numpy(float)
                mean_speed = uniform_filter1d(np.abs(dq), window, mode="nearest")
                steady = np.abs(np.abs(dq) - mean_speed) <= SPEED_STEADINESS * np.maximum(mean_speed, 1e-9)
                q, ft = group[f"q{j}"].to_numpy(float), group[f"ft{j}"].to_numpy(float)
                for i, order in enumerate(orders):
                    f = order * mean_speed / (2.0 * np.pi)
                    chosen = steady & (f >= F_LOW) & (f <= 0.4 / step)
                    power[i] += float(np.sum(tracked_amplitude(ft, order * q, window)[chosen] ** 2)) / max(chosen.sum(), 1)
                count += 1
            amplitude = np.sqrt(power / max(count, 1))
            assumed = ORDER[platform]
            rows.append({"source": file_id, "joint": joint, "peak_order": float(orders[int(np.argmax(amplitude))]),
                         "peak_amplitude": float(np.max(amplitude)), "assumed_order": assumed,
                         "assumed_amplitude": float(np.interp(assumed, orders, amplitude)),
                         "median_amplitude": float(np.median(amplitude))})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--platform", nargs="+", default=None, help="default: both")
    parser.add_argument("--no-report", action="store_true", help="print only; write nothing")
    args = parser.parse_args(argv)
    settings = yaml.safe_load(CAMPAIGN.read_text(encoding="utf-8"))
    tables, ratio_tables, scans = [], [], []
    for platform in args.platform or REAL:
        stem = _REPO / settings["data_root"] / platform / settings["platforms"][platform]["stem"]
        differentiation = json.loads(stem.with_suffix(".contract.json").read_text(encoding="utf-8"))["differentiation"]
        window, poly = int(differentiation["sg_window"]), int(differentiation["sg_poly"])
        amplitudes = te_amplitude(pd.concat([real_rows(platform, window, poly), sim_rows(platform, stem, window, poly)]))
        ratio = ratios(amplitudes).assign(platform=platform, report_only=platform in REPORT_ONLY)
        print(ratio.round(4).to_string(index=False), flush=True)
        scan = order_scan(platform).assign(platform=platform)
        print(scan.round(3).to_string(index=False), flush=True)
        scans.append(scan)
        tables.append(amplitudes.assign(platform=platform))
        ratio_tables.append(ratio)
    if not args.no_report:
        pd.concat(tables).to_csv(RESULTS / "04_te_band.csv", index=False)
        pd.concat(ratio_tables).to_csv(RESULTS / "04_te_band_ratio.csv", index=False)
        pd.concat(scans).to_csv(RESULTS / "04_te_order_scan.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
