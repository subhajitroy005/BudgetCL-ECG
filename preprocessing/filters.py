"""Signal filters for the ECG front end.

All three filters here are NON-CAUSAL. They are kept because they match the
source model's published front end, and changing them would confound the
adaptation comparison with a preprocessing change. The consequence is stated
plainly wherever it matters rather than hidden: see
:mod:`preprocessing.influence_analysis` for the measured backward reach and
:mod:`preprocessing.split_first_pipeline` for the sensitivity rerun.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, medfilt

__all__ = [
    "BASELINE_FIRST_MS",
    "BASELINE_SECOND_MS",
    "LOWPASS_CUTOFF_HZ",
    "LOWPASS_ORDER",
    "odd_window",
    "remove_baseline_wander",
    "lowpass_filter",
    "denoise",
]

BASELINE_FIRST_MS = 200
BASELINE_SECOND_MS = 600
LOWPASS_CUTOFF_HZ = 35.0
LOWPASS_ORDER = 4


def odd_window(seconds: float, fs: int) -> int:
    """Median-filter kernel length in samples, rounded up to odd.

    ``scipy.signal.medfilt`` requires an odd kernel.
    """
    n = int(round(seconds * fs))
    return n if n % 2 == 1 else n + 1


def remove_baseline_wander(sig: np.ndarray, fs: int) -> np.ndarray:
    """Two-stage median-filter baseline removal (non-causal).

    A 200 ms kernel strips the QRS and P complexes leaving baseline plus
    T-wave; a 600 ms kernel on that result strips the T-wave; the remaining
    estimate is subtracted. Standard technique (de Chazal et al. 2004).

    Both kernels are centred, so each output sample depends on samples on BOTH
    sides -- roughly 144 samples of one-sided support at 360 Hz.
    """
    w1 = odd_window(BASELINE_FIRST_MS / 1000.0, fs)
    w2 = odd_window(BASELINE_SECOND_MS / 1000.0, fs)
    baseline = medfilt(medfilt(sig, kernel_size=w1), kernel_size=w2)
    return sig - baseline


def lowpass_filter(
    sig: np.ndarray,
    fs: int,
    cutoff_hz: float = LOWPASS_CUTOFF_HZ,
    order: int = LOWPASS_ORDER,
) -> np.ndarray:
    """Zero-phase Butterworth low-pass (non-causal).

    ``filtfilt`` runs the filter forward and backward to cancel phase
    distortion, which necessarily makes each output sample depend on FUTURE
    input samples. It could not run unmodified in a streaming device.
    """
    nyquist = fs / 2.0
    b, a = butter(order, cutoff_hz / nyquist, btype="low")
    return filtfilt(b, a, sig)


def denoise(sig: np.ndarray, fs: int) -> np.ndarray:
    """Full front end: baseline-wander removal then low-pass."""
    return lowpass_filter(remove_baseline_wander(sig, fs), fs).astype(np.float32)
