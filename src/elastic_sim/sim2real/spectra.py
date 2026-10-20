"""Residual spectra and coherence: the S2R_06 F-9 diagnostic (P-4b's first).

F-9 asks, before the > 2 Hz part of the R0 residual is read as plant
structure, what of it is (1) above the **floor** -- the twin's residual at
its own theta: injected noise plus the SG differentiation -- and (2)
**coherent with a physical signal**:

* the transmission-ripple order: ``n = 320`` cycles per rad of joint angle
  (51 Hz per rad/s); a time-varying frequency, so it is tested by a
  phase-free regression of the residual on ``cos(n q), sin(n q)``
  (:func:`order_fraction`), not by a stationary coherence;
* ``q''`` (the replay's SG acceleration) and the tracking error
  ``cmd - q`` (the FRI command echo): magnitude-squared coherence, reported
  per band as the **fraction of the residual power it explains linearly**,
  ``sum_f C(f) P_rr(f) / sum_f P_rr(f)`` (:func:`explained_fraction`).

Spectra are Welch (Hann, ``nperseg`` 2048 at 1 kHz), summed over bags
before the ratios are formed.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.signal import csd, welch

FS = 1000.0
NPERSEG = 2048
#: F-9 bands, Hz (A-2 (d)'s three, the > 2 Hz one split).
BANDS = ((0.0, 0.5), (0.5, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, 20.0), (20.0, 50.0), (50.0, 100.0),
         (100.0, 200.0), (200.0, 500.0))
RIPPLE_ORDER = 320.0


def band_label(band: tuple[float, float]) -> str:
    return f"{band[0]:g}-{band[1]:g} Hz"


def psd_sum(signals: Iterable[np.ndarray], *, fs: float = FS, nperseg: int = NPERSEG) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD per column, summed over the signals (bags) weighted by length."""
    total, f = None, None
    for x in signals:
        f, p = welch(np.asarray(x, dtype=float), fs=fs, nperseg=min(nperseg, len(x)), axis=0)
        p = p * len(x)
        total = p if total is None else total + _match(p, total)
    return f, total


def _match(p: np.ndarray, ref: np.ndarray) -> np.ndarray:
    if p.shape != ref.shape:
        raise ValueError("bags shorter than nperseg give a different frequency grid; use equal segments")
    return p


def band_powers(f: np.ndarray, psd: np.ndarray, bands: Sequence[tuple[float, float]] = BANDS) -> np.ndarray:
    """(bands, joints): the integral of the PSD over each band (power, units^2)."""
    df = f[1] - f[0]
    return np.vstack([np.sum(psd[(f >= lo) & (f < hi)], axis=0) * df for lo, hi in bands])


def explained_fraction(residuals: Sequence[np.ndarray], signals: Sequence[np.ndarray], *, fs: float = FS,
                       nperseg: int = NPERSEG, bands: Sequence[tuple[float, float]] = BANDS) -> np.ndarray:
    """(bands, joints): the share of the residual's band power that ``signal``
    explains linearly, joint by joint (``C(f) P_rr(f)`` summed over the band
    over ``P_rr`` summed), spectra summed over bags."""
    prr = pss = prs = None
    for r, s in zip(residuals, signals):
        n = min(nperseg, len(r))
        f, a = welch(r, fs=fs, nperseg=n, axis=0)
        _, b = welch(s, fs=fs, nperseg=n, axis=0)
        _, c = csd(r, s, fs=fs, nperseg=n, axis=0)
        w = len(r)
        prr = a * w if prr is None else prr + a * w
        pss = b * w if pss is None else pss + b * w
        prs = c * w if prs is None else prs + c * w
    coherence = np.abs(prs) ** 2 / np.maximum(prr * pss, 1e-300)
    out = []
    for lo, hi in bands:
        mask = (f >= lo) & (f < hi)
        out.append(np.sum(coherence[mask] * prr[mask], axis=0) / np.maximum(np.sum(prr[mask], axis=0), 1e-300))
    return np.vstack(out)


def order_fraction(residuals: Sequence[np.ndarray], angles: Sequence[np.ndarray], *, order: float = RIPPLE_ORDER,
                   highpass_hz: float = 2.0, fs: float = FS) -> np.ndarray:
    """(joints,): the share of the residual's > ``highpass_hz`` power that a
    phase-free sinusoid in ``order * q_j`` explains (least squares on
    ``cos, sin`` per bag, pooled)."""
    from scipy.signal import butter, sosfiltfilt

    sos = butter(4, highpass_hz, btype="highpass", fs=fs, output="sos")
    n = residuals[0].shape[1]
    explained = np.zeros(n)
    total = np.zeros(n)
    for r, q in zip(residuals, angles):
        r = sosfiltfilt(sos, np.asarray(r, dtype=float), axis=0)
        for j in range(n):
            basis = np.column_stack([np.cos(order * q[:, j]), np.sin(order * q[:, j])])
            basis = sosfiltfilt(sos, basis, axis=0)
            coef, *_ = np.linalg.lstsq(basis, r[:, j], rcond=None)
            explained[j] += float(np.sum((basis @ coef) ** 2))
            total[j] += float(np.sum(r[:, j] ** 2))
    return explained / np.maximum(total, 1e-300)


def table(values: np.ndarray, joints: Sequence[str], *, rows: Sequence[str], fmt: str = ".3f") -> list[str]:
    lines = ["| band | " + " | ".join(joints) + " |", "|---|" + "---|" * len(joints)]
    for label, row in zip(rows, values):
        lines.append(f"| {label} | " + " | ".join(format(float(v), fmt) for v in row) + " |")
    return lines


def summary_rows(blocks: Mapping[str, np.ndarray]) -> dict[str, list]:
    return {k: np.asarray(v).tolist() for k, v in blocks.items()}


__all__ = ["BANDS", "FS", "NPERSEG", "RIPPLE_ORDER", "band_label", "band_powers", "explained_fraction",
           "order_fraction", "psd_sum", "table"]
