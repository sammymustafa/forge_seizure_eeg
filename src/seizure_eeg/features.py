"""Window-level feature extraction.

All montages are scored on the *same* windows of the *same* recording, so
features are computed once for the union of channels any montage needs and the
montage-specific design matrices are column slices of that array.

``window_features`` is vectorised over arbitrary leading dimensions, so a whole
recording can be filtered once and then reshaped to
``(n_channels, n_windows, n_samples)`` and featurised in a single call.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch

from .config import BANDPASS, BANDS, FEATURES_PER_CHANNEL

_SOS_CACHE: dict[float, np.ndarray] = {}
_EPS = 1e-20


def _sos(sfreq: float) -> np.ndarray:
    if sfreq not in _SOS_CACHE:
        _SOS_CACHE[sfreq] = butter(
            4, list(BANDPASS), btype="bandpass", fs=sfreq, output="sos"
        )
    return _SOS_CACHE[sfreq]


def bandpass(data: np.ndarray, sfreq: float) -> np.ndarray:
    """Zero-phase 0.5-40 Hz band-pass applied along the last axis."""
    return sosfiltfilt(_sos(sfreq), data, axis=-1)


def window_features(x: np.ndarray, sfreq: float) -> np.ndarray:
    """Features for band-passed windows.

    Parameters
    ----------
    x : array with shape ``(..., n_samples)``, already band-passed, in volts.
    sfreq : sampling rate (Hz).

    Returns
    -------
    array of shape ``(..., len(FEATURES_PER_CHANNEL))``.
    """
    nperseg = int(min(2 * sfreq, x.shape[-1]))
    freqs, pxx = welch(x, fs=sfreq, nperseg=nperseg, noverlap=nperseg // 2,
                       detrend="constant", axis=-1)
    df = float(freqs[1] - freqs[0])

    lo, hi = BANDPASS
    inband = (freqs >= lo) & (freqs <= hi)
    total = pxx[..., inband].sum(axis=-1) * df

    abs_pow = np.stack(
        [pxx[..., (freqs >= f_lo) & (freqs < f_hi)].sum(axis=-1) * df
         for f_lo, f_hi in BANDS.values()],
        axis=-1,
    )

    log_abs = np.log10(abs_pow + _EPS)
    log_rel = np.log10(abs_pow / (total[..., None] + _EPS) + _EPS)

    rms = np.sqrt(np.mean(x ** 2, axis=-1))
    line_len = np.mean(np.abs(np.diff(x, axis=-1)), axis=-1)

    p = pxx[..., inband]
    p = p / (p.sum(axis=-1, keepdims=True) + _EPS)
    ent = -np.sum(p * np.log(p + _EPS), axis=-1) / np.log(p.shape[-1])

    names = list(BANDS)
    slow = abs_pow[..., names.index("delta")] + abs_pow[..., names.index("theta")]
    fast = abs_pow[..., names.index("beta")] + abs_pow[..., names.index("low_gamma")]
    ratio = np.log10((fast + _EPS) / (slow + _EPS))

    feats = np.concatenate(
        [log_abs, log_rel,
         np.log10(rms + _EPS)[..., None],
         np.log10(line_len + _EPS)[..., None],
         ent[..., None], ratio[..., None]],
        axis=-1,
    )
    assert feats.shape[-1] == len(FEATURES_PER_CHANNEL), feats.shape
    return feats


def window_quality(x: np.ndarray) -> np.ndarray:
    """Per-window, per-channel usability: finite and not flat-lined."""
    finite = np.isfinite(x).all(axis=-1)
    return finite & (np.ptp(x, axis=-1) > 1e-9)
