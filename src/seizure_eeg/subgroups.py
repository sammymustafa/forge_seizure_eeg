"""Subgroup sensitivity / latency with patient-level uncertainty.

Subgroup analyses are reported on **events**, but uncertainty is resampled over
**patients**: seizures from one person are not independent observations, and a
single patient contributing 28 seizures would otherwise dominate a subgroup.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _resample_indices(df: pd.DataFrame, n_boot: int, rng: np.random.Generator):
    """Yield row-index arrays for ``n_boot`` patient-level resamples.

    Patients are resampled with replacement and all of a patient's events come
    along, so a person contributing many seizures is correctly treated as one
    observation rather than many.
    """
    patients = df.patient_id.to_numpy()
    uniq = np.unique(patients)
    if uniq.size < 2:
        return uniq, None
    per_patient = [np.flatnonzero(patients == p) for p in uniq]
    picks = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    return uniq, (per_patient, picks)


def _patient_bootstrap(df: pd.DataFrame, stat, n_boot: int = 2000,
                       alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile CI for ``stat`` over patient-level resamples."""
    rng = np.random.default_rng(seed)
    uniq, plan = _resample_indices(df, n_boot, rng)
    if plan is None:
        return (np.nan, np.nan)
    per_patient, picks = plan
    detected = df.detected.to_numpy().astype(bool)
    latency = df.latency_s.to_numpy(dtype=float)
    vals = []
    for row in picks:
        idx = np.concatenate([per_patient[i] for i in row])
        v = stat(detected[idx], latency[idx])
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return (np.nan, np.nan)
    return (float(np.quantile(vals, alpha / 2)),
            float(np.quantile(vals, 1 - alpha / 2)))


def _sens_arr(detected: np.ndarray, latency: np.ndarray) -> float:
    return float(detected.mean()) if detected.size else np.nan


def _lat_arr(detected: np.ndarray, latency: np.ndarray) -> float:
    d = latency[detected]
    d = d[np.isfinite(d)]
    return float(np.median(d)) if d.size else np.nan


def _sensitivity(df: pd.DataFrame) -> float:
    return float(df.detected.mean()) if len(df) else np.nan


def _median_latency(df: pd.DataFrame) -> float:
    d = df.loc[df.detected, "latency_s"].dropna()
    return float(d.median()) if len(d) else np.nan


def subgroup_sensitivity(events: pd.DataFrame, by: str, *, min_events: int = 1,
                         n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Per-subgroup event sensitivity and latency, with patient bootstrap CIs.

    ``events`` needs columns ``patient_id``, ``detected``, ``latency_s`` and the
    grouping column. Subgroups are reported with their patient and seizure
    counts so that a sensitivity computed from three seizures in one patient is
    visibly not the same evidence as one from eighty seizures in forty.
    """
    rows = []
    for key, g in events.groupby(by, dropna=False):
        if len(g) < min_events:
            continue
        s_lo, s_hi = _patient_bootstrap(g, _sens_arr, n_boot, seed=seed)
        l_lo, l_hi = _patient_bootstrap(g, _lat_arr, n_boot, seed=seed)
        det = g[g.detected]
        rows.append({
            by: key,
            "n_seizures": int(len(g)),
            "n_patients": int(g.patient_id.nunique()),
            "n_detected": int(g.detected.sum()),
            "sensitivity": _sensitivity(g),
            "sensitivity_lo": s_lo, "sensitivity_hi": s_hi,
            "median_latency_s": _median_latency(g),
            "latency_lo": l_lo, "latency_hi": l_hi,
            "median_duration_s": float(g.duration_s.median())
            if "duration_s" in g else np.nan,
            "n_detected_with_latency": int(det.latency_s.notna().sum()),
        })
    out = pd.DataFrame(rows)
    return out.sort_values("n_seizures", ascending=False) if len(out) else out


def paired_subgroup_delta(events: pd.DataFrame, by: str, a: str, b: str,
                          n_boot: int = 2000, seed: int = 0) -> dict:
    """Difference in sensitivity between two subgroups, resampling patients.

    Patients are resampled once and both subgroup sensitivities recomputed from
    the same resample, so the CI respects the fact that one patient can
    contribute seizures to both subgroups.
    """
    sub = events[events[by].isin([a, b])].reset_index(drop=True)
    if sub.empty:
        return {}
    rng = np.random.default_rng(seed)
    uniq, plan = _resample_indices(sub, n_boot, rng)
    deltas: list[float] = []
    if plan is not None:
        per_patient, picks = plan
        detected = sub.detected.to_numpy().astype(bool)
        is_a = (sub[by] == a).to_numpy()
        is_b = (sub[by] == b).to_numpy()
        for row in picks:
            idx = np.concatenate([per_patient[i] for i in row])
            da, db = detected[idx][is_a[idx]], detected[idx][is_b[idx]]
            if da.size and db.size:
                deltas.append(float(da.mean() - db.mean()))
    ga, gb = sub[sub[by] == a], sub[sub[by] == b]
    return {
        "comparison": f"{a} - {b}",
        "n_seizures_a": int(len(ga)), "n_patients_a": int(ga.patient_id.nunique()),
        "n_seizures_b": int(len(gb)), "n_patients_b": int(gb.patient_id.nunique()),
        "sensitivity_a": _sensitivity(ga), "sensitivity_b": _sensitivity(gb),
        "delta": _sensitivity(ga) - _sensitivity(gb),
        "ci_lo": float(np.quantile(deltas, 0.025)) if deltas else np.nan,
        "ci_hi": float(np.quantile(deltas, 0.975)) if deltas else np.nan,
    }
