"""Event-level seizure-detection scoring.

Window-level F1 is not a clinically meaningful endpoint. These helpers turn a
stream of per-window scores into *alarms*, then score alarms against annotated
seizure events: sensitivity, false alarms per hour, and detection latency.

Conventions (following the spirit of SzCORE, Dan et al. 2024):

* an alarm requires ``min_consecutive`` consecutive windows above threshold;
* the alarm is raised at the END of the last of those windows, because a causal
  detector cannot report a window before it has been observed;
* a continuous run above threshold is re-issued every ``max_alarm_s`` seconds.
  Without this, a detector stuck permanently in alarm would score one false
  alarm per recording and near-perfect sensitivity;
* a seizure counts as detected if an alarm overlaps
  ``[onset - PRE_TOL, offset + POST_TOL]``;
* latency is measured from the annotated onset to the moment the alarm became
  informative, clipped at ``-PRE_TOL`` so an alarm that was already running
  cannot earn arbitrarily large negative credit;
* any alarm overlapping no seizure tolerance window is one false alarm.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PRE_TOL = 30.0      # s before annotated onset that still counts as a detection
POST_TOL = 60.0     # s after annotated offset that still counts as a detection
MAX_ALARM_S = 60.0  # a sustained alarm is re-issued at this period


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Index ranges [i, j) of consecutive True values."""
    if mask.size == 0:
        return []
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    change = np.diff(padded.astype(np.int8))
    return list(zip(np.flatnonzero(change == 1).tolist(),
                    np.flatnonzero(change == -1).tolist()))


def alarms_for_file(start_s: np.ndarray, stop_s: np.ndarray, above: np.ndarray,
                    min_consecutive: int = 1,
                    max_alarm_s: float = MAX_ALARM_S) -> list[tuple[float, float]]:
    """Alarm intervals ``(raised_at, ends_at)`` for one recording.

    ``start_s``/``stop_s`` must be sorted and describe contiguous windows.
    """
    out: list[tuple[float, float]] = []
    for i, j in _runs(above):
        if (j - i) < min_consecutive:
            continue
        raised_at = float(stop_s[i + min_consecutive - 1])
        run_end = float(stop_s[j - 1])
        t = raised_at
        while True:
            end = min(t + max_alarm_s, run_end)
            out.append((t, end))
            if end >= run_end:
                break
            t = end
    return out


def score_events(meta: pd.DataFrame, scores: np.ndarray, threshold: float,
                 seizure_table: pd.DataFrame, min_consecutive: int = 1,
                 max_alarm_s: float = MAX_ALARM_S) -> dict:
    """Event-level metrics over a set of recordings.

    Parameters
    ----------
    meta : window metadata with columns ``edf_file, start_s, stop_s``.
    scores : detector score per row of ``meta``.
    seizure_table : ``edf_file, onset_s, offset_s`` for every annotated seizure
        in the recordings covered by ``meta``.
    """
    meta = meta.reset_index(drop=True)
    above = np.asarray(scores) >= threshold

    detected: list[bool] = []
    latencies: list[float] = []
    n_false = 0
    total_s = ictal_s = 0.0
    n_above = int(above.sum())

    by_file_seiz = {f: g.reset_index(drop=True)
                    for f, g in seizure_table.groupby("edf_file")}

    for fname, idx in meta.groupby("edf_file").groups.items():
        idx = np.asarray(idx)
        idx = idx[np.argsort(meta.loc[idx, "start_s"].to_numpy())]
        st = meta.loc[idx, "start_s"].to_numpy()
        sp = meta.loc[idx, "stop_s"].to_numpy()
        total_s += float((sp - st).sum())

        seiz = by_file_seiz.get(fname)
        tol_windows: list[tuple[float, float]] = []
        onsets: list[float] = []
        if seiz is not None:
            for r in seiz.itertuples():
                tol_windows.append((r.onset_s - PRE_TOL, r.offset_s + POST_TOL))
                onsets.append(float(r.onset_s))
                ictal_s += float(r.offset_s - r.onset_s)

        alarms = alarms_for_file(st, sp, above[idx], min_consecutive, max_alarm_s)
        hit = [False] * len(tol_windows)
        for a0, a1 in alarms:
            matched = False
            for k, (w0, w1) in enumerate(tol_windows):
                if a0 <= w1 and a1 >= w0:
                    matched = True
                    if not hit[k]:
                        hit[k] = True
                        # clip: credit starts no earlier than the tolerance edge
                        latencies.append(float(max(a0, w0) - onsets[k]))
            if not matched:
                n_false += 1
        detected.extend(hit)

    n_events = len(detected)
    interictal_h = max(total_s - ictal_s, 1.0) / 3600.0
    lat = np.asarray(latencies, dtype=float)
    return {
        "n_seizure_events": n_events,
        "n_detected": int(np.sum(detected)),
        "event_sensitivity": float(np.mean(detected)) if n_events else np.nan,
        "n_false_alarms": int(n_false),
        "false_alarms_per_hour": n_false / interictal_h,
        "false_alarms_per_24h": 24.0 * n_false / interictal_h,
        "alarm_duty_cycle": n_above / max(len(meta), 1),
        "recorded_hours": total_s / 3600.0,
        "median_latency_s": float(np.median(lat)) if lat.size else np.nan,
        "mean_latency_s": float(np.mean(lat)) if lat.size else np.nan,
    }


def threshold_for_target_fa(meta: pd.DataFrame, scores: np.ndarray,
                            seizure_table: pd.DataFrame, target_fa_per_h: float,
                            min_consecutive: int = 1, n_grid: int = 40,
                            max_alarm_s: float = MAX_ALARM_S) -> float:
    """Smallest threshold whose false-alarm rate is <= the target.

    Intended to be run on *training* data only; the chosen value is then applied
    unchanged to held-out patients.
    """
    finite = np.asarray(scores)[np.isfinite(scores)]
    if finite.size == 0:
        return np.inf
    qs = np.unique(np.quantile(finite, np.linspace(0.50, 0.9999, n_grid)))
    for thr in qs:                      # increasing threshold -> fewer alarms
        res = score_events(meta, scores, float(thr), seizure_table,
                           min_consecutive, max_alarm_s)
        if res["false_alarms_per_hour"] <= target_fa_per_h:
            return float(thr)
    return float(qs[-1])
