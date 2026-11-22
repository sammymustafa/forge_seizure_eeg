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

:class:`EventScorer` precomputes the per-recording layout once so that many
thresholds can be evaluated cheaply (threshold selection does a bisection over
a quantile grid).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PRE_TOL = 30.0      # s before annotated onset that still counts as a detection
POST_TOL = 60.0     # s after annotated offset that still counts as a detection
MAX_ALARM_S = 60.0  # a sustained alarm is re-issued at this period


def _runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Index ranges [i, j) of consecutive True values."""
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    change = np.diff(padded.astype(np.int8))
    return np.flatnonzero(change == 1), np.flatnonzero(change == -1)


def alarms_for_file(start_s: np.ndarray, stop_s: np.ndarray, above: np.ndarray,
                    min_consecutive: int = 1,
                    max_alarm_s: float = MAX_ALARM_S) -> list[tuple[float, float]]:
    """Alarm intervals ``(raised_at, ends_at)`` for one recording."""
    out: list[tuple[float, float]] = []
    starts, stops = _runs(above)
    for i, j in zip(starts, stops):
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


class EventScorer:
    """Score many thresholds against a fixed set of recordings."""

    def __init__(self, meta: pd.DataFrame, seizure_table: pd.DataFrame,
                 min_consecutive: int = 1, max_alarm_s: float = MAX_ALARM_S):
        self.min_consecutive = int(min_consecutive)
        self.max_alarm_s = float(max_alarm_s)
        meta = meta.reset_index(drop=True)
        self.n_rows = len(meta)

        by_file = {f: g.reset_index(drop=True)
                   for f, g in seizure_table.groupby("edf_file")}
        self.files: list[dict] = []
        self.total_s = 0.0
        self.ictal_s = 0.0
        self.n_events = 0

        for fname, idx in meta.groupby("edf_file").groups.items():
            idx = np.asarray(idx)
            idx = idx[np.argsort(meta.loc[idx, "start_s"].to_numpy())]
            st = meta.loc[idx, "start_s"].to_numpy(dtype=float)
            sp = meta.loc[idx, "stop_s"].to_numpy(dtype=float)
            self.total_s += float((sp - st).sum())

            seiz = by_file.get(fname)
            tol, onsets, ids = [], [], []
            if seiz is not None:
                for r in seiz.itertuples():
                    tol.append((r.onset_s - PRE_TOL, r.offset_s + POST_TOL))
                    onsets.append(float(r.onset_s))
                    ids.append(getattr(r, "event_uid", r.Index))
                    self.ictal_s += float(r.offset_s - r.onset_s)
            self.n_events += len(tol)
            self.files.append({"idx": idx, "stop": sp, "tol": tol,
                               "onsets": onsets, "ids": ids, "file": fname})
        self.interictal_h = max(self.total_s - self.ictal_s, 1.0) / 3600.0

    # ------------------------------------------------------------------
    def score(self, scores: np.ndarray, threshold: float) -> dict:
        scores = np.asarray(scores)
        above_all = scores >= threshold
        detected: list[bool] = []
        latencies: list[float] = []
        n_false = 0

        for f in self.files:
            sp = f["stop"]
            above = above_all[f["idx"]]
            tol = f["tol"]
            hit = [False] * len(tol)
            starts, stops = _runs(above)
            for i, j in zip(starts, stops):
                if (j - i) < self.min_consecutive:
                    continue
                raised = float(sp[i + self.min_consecutive - 1])
                run_end = float(sp[j - 1])
                t = raised
                while True:
                    end = min(t + self.max_alarm_s, run_end)
                    matched = False
                    for k, (w0, w1) in enumerate(tol):
                        if t <= w1 and end >= w0:
                            matched = True
                            if not hit[k]:
                                hit[k] = True
                                latencies.append(float(max(t, w0) - f["onsets"][k]))
                    if not matched:
                        n_false += 1
                    if end >= run_end:
                        break
                    t = end
            detected.extend(hit)

        lat = np.asarray(latencies, dtype=float)
        n_events = len(detected)
        return {
            "n_seizure_events": n_events,
            "n_detected": int(np.sum(detected)),
            "event_sensitivity": float(np.mean(detected)) if n_events else np.nan,
            "n_false_alarms": int(n_false),
            "false_alarms_per_hour": n_false / self.interictal_h,
            "false_alarms_per_24h": 24.0 * n_false / self.interictal_h,
            "alarm_duty_cycle": float(above_all.mean()) if above_all.size else np.nan,
            "recorded_hours": self.total_s / 3600.0,
            "median_latency_s": float(np.median(lat)) if lat.size else np.nan,
            "mean_latency_s": float(np.mean(lat)) if lat.size else np.nan,
        }

    # ------------------------------------------------------------------
    def event_table(self, scores: np.ndarray, threshold: float) -> pd.DataFrame:
        """One row per annotated seizure: detected or not, and the latency.

        This is what subgroup analyses need — pooled sensitivity hides which
        seizures were missed, and seizure type / onset lobe are properties of
        individual events.
        """
        scores = np.asarray(scores)
        above_all = scores >= threshold
        rows = []
        for f in self.files:
            sp, tol = f["stop"], f["tol"]
            if not tol:
                continue
            above = above_all[f["idx"]]
            hit = [False] * len(tol)
            lat = [np.nan] * len(tol)
            starts, stops = _runs(above)
            for i, j in zip(starts, stops):
                if (j - i) < self.min_consecutive:
                    continue
                raised = float(sp[i + self.min_consecutive - 1])
                run_end = float(sp[j - 1])
                t = raised
                while True:
                    end = min(t + self.max_alarm_s, run_end)
                    for k, (w0, w1) in enumerate(tol):
                        if t <= w1 and end >= w0 and not hit[k]:
                            hit[k] = True
                            lat[k] = float(max(t, w0) - f["onsets"][k])
                    if end >= run_end:
                        break
                    t = end
            for k in range(len(tol)):
                rows.append({"recording": f["file"], "event_uid": f["ids"][k],
                             "onset_s": f["onsets"][k], "detected": hit[k],
                             "latency_s": lat[k], "threshold": threshold})
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    def threshold_for_target_fa(self, scores: np.ndarray, target_fa_per_h: float,
                                n_grid: int = 64, refine: int = 5) -> float:
        """Smallest grid threshold whose false-alarm rate is <= the target.

        The false-alarm rate falls as the threshold rises, but **not strictly**:
        raising the threshold can punch a hole in one sustained run and split it
        into two alarms. So this bisects for speed and then rescans a window of
        ``refine`` grid points either side of the crossing. Whatever it returns
        satisfies the budget on the data it was given (or is the largest grid
        value, if nothing does).

        Intended for *training* data only; the chosen value is then applied
        unchanged to held-out patients.
        """
        s = np.asarray(scores)
        finite = s[np.isfinite(s)]
        if finite.size == 0:
            return np.inf
        # dense near the upper tail, where a rare-event threshold lives
        qs = 1.0 - np.logspace(np.log10(0.5), np.log10(1e-6), n_grid)
        grid = np.unique(np.quantile(finite, qs))

        def ok(i: int) -> bool:
            return self.score(s, float(grid[i]))["false_alarms_per_hour"] <= \
                target_fa_per_h

        lo, hi = 0, len(grid) - 1
        crossing = hi
        while lo <= hi:
            mid = (lo + hi) // 2
            if ok(mid):
                crossing = mid
                hi = mid - 1
            else:
                lo = mid + 1

        best = crossing
        found = ok(crossing)
        for i in range(max(0, crossing - refine), min(len(grid), crossing + refine + 1)):
            if i == crossing:
                continue
            if ok(i) and (not found or i < best):
                best, found = i, True
        if not found:
            return float(grid[-1])
        return float(grid[best])


# ---------------------------------------------------------------- shims
def score_events(meta: pd.DataFrame, scores: np.ndarray, threshold: float,
                 seizure_table: pd.DataFrame, min_consecutive: int = 1,
                 max_alarm_s: float = MAX_ALARM_S) -> dict:
    return EventScorer(meta, seizure_table, min_consecutive, max_alarm_s).score(
        scores, threshold)


def threshold_for_target_fa(meta: pd.DataFrame, scores: np.ndarray,
                            seizure_table: pd.DataFrame, target_fa_per_h: float,
                            min_consecutive: int = 1, n_grid: int = 64,
                            max_alarm_s: float = MAX_ALARM_S) -> float:
    return EventScorer(meta, seizure_table, min_consecutive,
                       max_alarm_s).threshold_for_target_fa(
        scores, target_fa_per_h, n_grid)
