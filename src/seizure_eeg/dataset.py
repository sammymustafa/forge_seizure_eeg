"""Turn EDF recordings into window-level feature tensors.

Design note
-----------
Every montage in :data:`seizure_eeg.config.MONTAGES` is a subset of one union
channel set. We extract features for that union **once per recording** and keep
a per-window / per-channel quality mask. A montage's design matrix is then a
column slice, which guarantees that all montages are compared on exactly the
same windows of exactly the same recordings.
"""
from __future__ import annotations

import re
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from .config import (BANDS, FEATURES_PER_CHANNEL, MONTAGES, WINDOW_S,
                     patient_id)
from .features import bandpass, window_features, window_quality

mne.set_log_level("ERROR")

#: Canonical union of every bipolar derivation any montage needs.
UNION_CHANNELS: list[str] = sorted({ch for chans in MONTAGES.values() for ch in chans})
CH_INDEX = {ch: i for i, ch in enumerate(UNION_CHANNELS)}

_DUP_SUFFIX = re.compile(r"-\d+$")
#: Common-reference labels seen in CHB-MIT (and in TUH-style referential files).
_REF_LABEL = re.compile(r"^(?P<electrode>[A-Z0-9]+)-(?:CS2|REF|LE|AVG)$")


def canonical_channel_map(ch_names: list[str]) -> dict[str, str]:
    """Map canonical CHB-MIT derivation names to the EDF's own channel labels.

    MNE disambiguates CHB-MIT's repeated channels by appending ``-0``/``-1``;
    placeholder channels named ``-`` are ignored. The first occurrence wins.
    """
    out: dict[str, str] = {}
    for name in ch_names:
        canon = _DUP_SUFFIX.sub("", name.strip().upper())
        # CHB-MIT pads its montage with placeholder channels named "-" / "--0";
        # a real derivation is "A-B" with non-empty electrode names on both sides.
        a, sep, b = canon.partition("-")
        if not sep or not a or not b:
            continue
        out.setdefault(canon, name)
    return out


#: Electrodes any montage could need, used to recognise a bare-label
#: referential recording (channels named "F7", "T7", ... with no reference).
_ELECTRODES = {e for chans in MONTAGES.values() for ch in chans
               for e in ch.split("-")}
#: Known label typos in the CHB-MIT EDF headers.
_LABEL_FIXES = {"01": "O1", "02": "O2"}


def referential_map(ch_names: list[str]) -> dict[str, str]:
    """Electrode -> EDF label for a common-reference recording.

    A few CHB-MIT recordings were stored referentially rather than as bipolar
    pairs: ``chb12_27`` against an explicit ``CS2`` reference, ``chb12_28/29``
    with bare electrode labels. Any bipolar derivation ``A-B`` can be
    reconstructed exactly as ``(A-ref) - (B-ref)`` — whatever the reference is,
    it cancels — so those recordings need not be discarded. The same applies to
    referential files in other corpora (e.g. TUH).

    Returns an empty mapping for an ordinary bipolar recording.
    """
    out: dict[str, str] = {}
    for name in ch_names:
        label = _DUP_SUFFIX.sub("", name.strip().upper())
        m = _REF_LABEL.match(label)
        if m:
            electrode = m.group("electrode")
        else:
            electrode = _LABEL_FIXES.get(label, label)
            if electrode not in _ELECTRODES:
                continue
        out.setdefault(electrode, name)
    return out


def _overlap_seconds(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def extract_recording(edf_path: str | Path,
                      intervals: list[tuple[int, int]],
                      window_s: float = WINDOW_S):
    """Featurise one EDF.

    Returns
    -------
    X : float32 (n_windows, n_union_channels, n_features), NaN where a channel
        is absent from this recording.
    Q : bool (n_windows, n_union_channels) usability mask.
    meta : DataFrame, one row per window.
    """
    edf_path = Path(edf_path)
    raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose="ERROR")
    sfreq = float(raw.info["sfreq"])
    n_samp_win = int(round(window_s * sfreq))
    n_win = int(raw.n_times // n_samp_win)
    if n_win == 0:
        raw.close()
        raise ValueError(f"{edf_path.name}: shorter than one window")

    present = canonical_channel_map(raw.ch_names)
    direct = [ch for ch in UNION_CHANNELS if ch in present]

    # Reconstruct any missing derivation from a common-reference recording.
    ref = referential_map(raw.ch_names)
    derived: list[tuple[str, str, str]] = []
    if ref:
        for ch in UNION_CHANNELS:
            if ch in present:
                continue
            a, _, b = ch.partition("-")
            if a in ref and b in ref:
                derived.append((ch, a, b))

    if not direct and not derived:
        raw.close()
        raise ValueError(f"{edf_path.name}: no usable derivations")

    ref_used = sorted({e for _, a, b in derived for e in (a, b)})
    labels = [present[ch] for ch in direct] + [ref[e] for e in ref_used]
    picks = [raw.ch_names.index(lbl) for lbl in labels]
    block = raw.get_data(picks=picks, start=0, stop=n_win * n_samp_win)
    raw.close()

    ref_pos = {e: len(direct) + k for k, e in enumerate(ref_used)}
    if derived:
        extra = np.stack([block[ref_pos[a]] - block[ref_pos[b]]
                          for _, a, b in derived])
        data = np.concatenate([block[:len(direct)], extra], axis=0)
    else:
        data = block[:len(direct)]
    available = direct + [ch for ch, _, _ in derived]
    del block

    bad_samples = ~np.isfinite(data)
    windows_raw = data.reshape(len(available), n_win, n_samp_win)
    q_avail = window_quality(windows_raw)
    if bad_samples.any():
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    filt = bandpass(data, sfreq)
    windows = filt.reshape(len(available), n_win, n_samp_win)
    feats = window_features(windows, sfreq)          # (n_avail, n_win, n_feat)

    n_feat = len(FEATURES_PER_CHANNEL)
    X = np.full((n_win, len(UNION_CHANNELS), n_feat), np.nan, dtype=np.float32)
    Q = np.zeros((n_win, len(UNION_CHANNELS)), dtype=bool)
    for k, ch in enumerate(available):
        j = CH_INDEX[ch]
        X[:, j, :] = feats[k].astype(np.float32)
        Q[:, j] = q_avail[k]
    X[~np.isfinite(X)] = np.nan

    starts = np.arange(n_win) * window_s
    stops = starts + window_s
    overlap = np.zeros(n_win)
    seiz_idx = np.full(n_win, -1, dtype=np.int16)
    for k, (a, b) in enumerate(intervals):
        ov = np.array([_overlap_seconds(s, e, a, b) for s, e in zip(starts, stops)])
        hit = ov > 0
        seiz_idx[hit & (seiz_idx < 0)] = k
        overlap = np.maximum(overlap, ov)

    case = edf_path.parent.name
    meta = pd.DataFrame({
        "case_id": case,
        "patient_id": patient_id(case),
        "edf_file": edf_path.name,
        "window_idx": np.arange(n_win),
        "start_s": starts,
        "stop_s": stops,
        "sfreq": sfreq,
        "ictal_overlap_s": overlap,
        "seizure_idx": seiz_idx,
        "label": (overlap > 0).astype(np.int8),
        "n_channels_ok": Q.sum(axis=1).astype(np.int16),
    })
    return X, Q, meta


def montage_matrix(X: np.ndarray, Q: np.ndarray, montage: str):
    """Slice the union tensor down to one montage.

    Returns ``(design_matrix, usable_window_mask, column_names)``.
    """
    chans = MONTAGES[montage]
    cols = [CH_INDEX[c] for c in chans]
    sub = X[:, cols, :]
    usable = Q[:, cols].all(axis=1) & np.isfinite(sub).all(axis=(1, 2))
    design = sub.reshape(sub.shape[0], -1)
    names = [f"{ch}:{f}" for ch in chans for f in FEATURES_PER_CHANNEL]
    return design, usable, names


def available_montages(Q: np.ndarray) -> list[str]:
    """Montages whose channels exist (usable in at least one window) in a file."""
    any_ok = Q.any(axis=0)
    return [m for m, chans in MONTAGES.items()
            if all(any_ok[CH_INDEX[c]] for c in chans)]
