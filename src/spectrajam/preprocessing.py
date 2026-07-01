"""Parity-critical MPC value transforms from pinned TESSERA preprocessing."""

from __future__ import annotations

from datetime import date, datetime

S2_BASELINE_CUTOFF = date(2022, 1, 25)
S2_BASELINE_OFFSET = 1000
SCL_INVALID_CLASSES = frozenset({0, 1, 2, 3, 8, 9})


def _numpy():
    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover - exercised without the data extra
        raise RuntimeError("install spectrajam[data] for MPC preprocessing") from error
    return np


def valid_scl(values):
    """Return TESSERA's S2 validity mask for MPC SCL values."""
    np = _numpy()
    array = np.asarray(values)
    finite = np.isfinite(array)
    return finite & ~np.isin(np.nan_to_num(array, nan=0), tuple(SCL_INVALID_CLASSES))


def harmonize_mpc_s2(values, acquired: date | datetime):
    """Remove the MPC PB-04.00 +1000 offset and return canonical uint16 values."""
    np = _numpy()
    acquired_date = acquired.date() if isinstance(acquired, datetime) else acquired
    array = np.asarray(values)
    work = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float64, copy=True
    )
    if acquired_date > S2_BASELINE_CUTOFF:
        subtract = np.isfinite(array) & (array >= S2_BASELINE_OFFSET)
        work[subtract] -= S2_BASELINE_OFFSET
    return np.clip(work, 0, np.iinfo(np.uint16).max).astype(np.uint16)


def amplitude_to_mpc_s1(values):
    """Convert MPC S1 amplitude to TESSERA's shifted/scaled int16 dB units."""
    np = _numpy()
    amplitude = np.asarray(values)
    output = np.zeros(amplitude.shape, dtype=np.int16)
    valid = np.isfinite(amplitude) & (amplitude > 0)
    if np.any(valid):
        scaled = (20.0 * np.log10(amplitude[valid]) + 50.0) * 200.0
        output[valid] = np.clip(scaled, 0, 32767).astype(np.int16)
    return output
