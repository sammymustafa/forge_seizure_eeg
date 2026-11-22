"""Patient-disjoint benchmark comparing electrode montages on identical windows."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             f1_score, roc_auc_score)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import FEATURES_DIR, FEATURES_PER_CHANNEL, MONTAGES
from .dataset import CH_INDEX, UNION_CHANNELS
from .evaluation import EventScorer
from .summary import parse_summary

_EPS = 1e-9


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
@dataclass
class WindowStore:
    """All windows from all cases: union-channel features + metadata."""
    X: np.ndarray                      # (n_windows, n_union_channels, n_features)
    Q: np.ndarray                      # (n_windows, n_union_channels) usable
    meta: pd.DataFrame
    seizures: pd.DataFrame
    channels: list[str] = field(default_factory=lambda: list(UNION_CHANNELS))

    def __len__(self) -> int:
        return self.X.shape[0]


def load_store(cases: list[str] | None = None,
               features_dir: Path = FEATURES_DIR,
               data_root: Path | None = None) -> WindowStore:
    files = sorted(features_dir.glob("chb*.npz"))
    if cases:
        files = [f for f in files if f.stem in set(cases)]
    if not files:
        raise FileNotFoundError(f"no feature files in {features_dir}")

    Xs, Qs, metas = [], [], []
    for f in files:
        with np.load(f, allow_pickle=False) as z:
            Xs.append(z["X"])
            Qs.append(z["Q"])
        metas.append(pd.read_csv(features_dir / f"{f.stem}_meta.csv.gz"))
    X = np.concatenate(Xs, axis=0)
    Q = np.concatenate(Qs, axis=0)
    meta = pd.concat(metas, ignore_index=True)

    # Rows of X/Q line up with the unsorted concatenation; sort everything
    # together so each recording's windows stay contiguous and time-ordered,
    # which the event-level scorer relies on.
    order = np.lexsort((meta.start_s.to_numpy(),
                        meta.edf_file.to_numpy(),
                        meta.case_id.to_numpy()))
    X, Q = X[order], Q[order]
    meta = meta.iloc[order].reset_index(drop=True)

    root = data_root or (features_dir.parents[1] / "data" / "chbmit")
    seizures = seizure_table(sorted(meta.case_id.unique()), root)
    return WindowStore(X=X, Q=Q, meta=meta, seizures=seizures)


def seizure_table(cases: list[str], data_root: Path) -> pd.DataFrame:
    rows = []
    for case in cases:
        spath = data_root / case / f"{case}-summary.txt"
        if not spath.exists():
            continue
        for fname, intervals in parse_summary(spath).items():
            for k, (a, b) in enumerate(intervals):
                rows.append({"case_id": case, "edf_file": fname, "seizure_idx": k,
                             "onset_s": float(a), "offset_s": float(b)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# design matrices
# --------------------------------------------------------------------------
def montage_design(store: WindowStore, montage: str) -> tuple[np.ndarray, np.ndarray,
                                                              list[str]]:
    chans = MONTAGES[montage]
    cols = [CH_INDEX[c] for c in chans]
    sub = store.X[:, cols, :]
    usable = store.Q[:, cols].all(axis=1) & np.isfinite(sub).all(axis=(1, 2))
    names = [f"{ch}:{f}" for ch in chans for f in FEATURES_PER_CHANNEL]
    return sub.reshape(sub.shape[0], -1).astype(np.float64), usable, names


def paired_mask(store: WindowStore, montages: list[str]) -> np.ndarray:
    """Windows usable by *every* montage under comparison (paired fairness)."""
    mask = np.ones(len(store), dtype=bool)
    for m in montages:
        _, usable, _ = montage_design(store, m)
        mask &= usable
    return mask


def normalise_per_file(X: np.ndarray, meta: pd.DataFrame) -> np.ndarray:
    """Robust (median / IQR) z-score of every feature within each recording.

    Label-free calibration that removes per-recording amplitude and impedance
    offsets. It is non-causal (it uses the whole recording), so it models an
    offline analysis or a device calibrated over a long baseline, not a
    strictly online detector.
    """
    out = np.empty_like(X)
    codes = meta.groupby(["case_id", "edf_file"], sort=False).ngroup().to_numpy()
    for g in np.unique(codes):
        sel = codes == g
        block = X[sel]
        med = np.median(block, axis=0)
        q75, q25 = np.percentile(block, [75, 25], axis=0)
        scale = (q75 - q25) / 1.349
        scale[scale < _EPS] = 1.0
        out[sel] = (block - med) / scale
    return out


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
def make_model(kind: str = "lda"):
    if kind == "lda":
        return make_pipeline(
            StandardScaler(),
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        )
    if kind == "logreg":
        from sklearn.linear_model import LogisticRegression
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0),
        )
    if kind == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        # Tuned for cost: at ~350k windows and 252 features the default 200
        # iterations take tens of minutes per fold, and this is a sanity check
        # on the model class, not a bid for best accuracy.
        return HistGradientBoostingClassifier(
            max_iter=60, learning_rate=0.15, max_leaf_nodes=15,
            early_stopping=True, validation_fraction=0.1,
            class_weight="balanced", random_state=0)
    raise ValueError(f"unknown model kind {kind!r}")


# --------------------------------------------------------------------------
# cross-validation
# --------------------------------------------------------------------------
def patient_folds(groups: np.ndarray, n_splits: int = 5):
    n_groups = len(np.unique(groups))
    if n_groups < n_splits:
        raise ValueError(f"{n_groups} patients < {n_splits} folds")
    dummy = np.zeros(len(groups))
    return list(GroupKFold(n_splits=n_splits).split(dummy, groups=groups))


def run_montage_cv(store: WindowStore, montage: str, mask: np.ndarray, *,
                   model_kind: str = "lda", normalise: str = "perfile",
                   n_splits: int = 5, target_fa_per_h: tuple[float, ...] = (6.0,),
                   min_consecutive: int = 2, inner_splits: int = 3,
                   feature_matrix: np.ndarray | None = None,
                   per_patient: bool = True):
    """Patient-disjoint CV for one montage, restricted to ``mask`` windows.

    Returns ``(fold_rows, patient_rows)`` as DataFrames. Decision thresholds are
    chosen from out-of-fold scores on the TRAINING patients only and then applied
    unchanged to the held-out patients.
    """
    X = feature_matrix
    if X is None:
        X, _, _ = montage_design(store, montage)
        X = X[mask]
        if normalise == "perfile":
            X = normalise_per_file(X, store.meta.loc[mask])
    meta = store.meta.loc[mask].reset_index(drop=True)
    y = meta.label.to_numpy()
    groups = meta.patient_id.to_numpy()
    seiz = store.seizures

    rows, prows = [], []
    for fold, (tr, te) in enumerate(patient_folds(groups, n_splits), 1):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        model = make_model(model_kind)
        model.fit(X[tr], y[tr])
        s_te = model.predict_proba(X[te])[:, 1]

        meta_te = meta.iloc[te].reset_index(drop=True)
        seiz_te = seiz[seiz.edf_file.isin(set(meta_te.edf_file))]

        if not target_fa_per_h:
            # Threshold-free run (used by the placement control): no inner CV,
            # no event scoring, just the ranking metrics.
            rows.append({
                "montage": montage, "fold": fold, "model": model_kind,
                "normalise": normalise, "n_features": int(X.shape[1]),
                "test_patients": ",".join(sorted(set(groups[te]))),
                "n_test_windows": int(len(te)),
                "n_test_ictal_windows": int(y[te].sum()),
                "auprc": float(average_precision_score(y[te], s_te)),
                "auroc": float(roc_auc_score(y[te], s_te)),
                "prevalence": float(y[te].mean()),
            })
            continue

        scorer_te = EventScorer(meta_te, seiz_te, min_consecutive)

        # out-of-fold scores on the TRAINING patients -> thresholds
        s_tr = np.empty(len(tr))
        g_tr = groups[tr]
        n_train_patients = len(np.unique(g_tr))
        if n_train_patients < 2:
            # Degenerate (smoke-test) case: no inner patient split is possible,
            # so the threshold is picked from in-sample training scores.
            s_tr = model.predict_proba(X[tr])[:, 1]
        else:
            inner = min(inner_splits, n_train_patients)
            for itr, ite in patient_folds(g_tr, inner):
                m2 = make_model(model_kind)
                m2.fit(X[tr][itr], y[tr][itr])
                s_tr[ite] = m2.predict_proba(X[tr][ite])[:, 1]
        meta_tr = meta.iloc[tr].reset_index(drop=True)
        seiz_tr = seiz[seiz.edf_file.isin(set(meta_tr.edf_file))]
        scorer_tr = EventScorer(meta_tr, seiz_tr, min_consecutive)

        patient_scorers = {}
        if per_patient:
            for pid in sorted(set(groups[te])):
                sel = meta_te.patient_id.to_numpy() == pid
                m_p = meta_te[sel].reset_index(drop=True)
                sz_p = seiz_te[seiz_te.edf_file.isin(set(m_p.edf_file))]
                patient_scorers[pid] = (sel, EventScorer(m_p, sz_p,
                                                         min_consecutive))

        base = {
            "montage": montage, "fold": fold, "model": model_kind,
            "normalise": normalise,
            "test_patients": ",".join(sorted(set(groups[te]))),
            "n_test_windows": int(len(te)),
            "n_test_ictal_windows": int(y[te].sum()),
            "n_features": int(X.shape[1]),
            "auprc": float(average_precision_score(y[te], s_te)),
            "auroc": float(roc_auc_score(y[te], s_te)),
            "prevalence": float(y[te].mean()),
        }
        pred05 = (s_te >= 0.5).astype(int)
        base["balanced_accuracy_at_0.5"] = float(balanced_accuracy_score(y[te], pred05))
        base["macro_f1_at_0.5"] = float(f1_score(y[te], pred05, average="macro",
                                                 zero_division=0))

        for target in target_fa_per_h:
            thr = scorer_tr.threshold_for_target_fa(s_tr, target)
            ev = scorer_te.score(s_te, thr)
            pred = (s_te >= thr).astype(int)
            row = dict(base)
            row.update({
                "target_fa_per_h": target,
                "threshold": thr,
                "min_consecutive": min_consecutive,
                "balanced_accuracy": float(balanced_accuracy_score(y[te], pred)),
                "window_sensitivity": float(
                    np.mean(pred[y[te] == 1]) if (y[te] == 1).any() else np.nan),
                "window_specificity": float(
                    np.mean(1 - pred[y[te] == 0]) if (y[te] == 0).any() else np.nan),
                **ev,
            })
            rows.append(row)

            for pid, (sel, scorer_p) in patient_scorers.items():
                s_p, y_p = s_te[sel], y[te][sel]
                prows.append({
                    "montage": montage, "fold": fold, "patient_id": pid,
                    "model": model_kind, "normalise": normalise,
                    "target_fa_per_h": target, "threshold": thr,
                    "n_windows": int(sel.sum()),
                    "n_ictal_windows": int(y_p.sum()),
                    "auprc": (float(average_precision_score(y_p, s_p))
                              if len(np.unique(y_p)) > 1 else np.nan),
                    "auroc": (float(roc_auc_score(y_p, s_p))
                              if len(np.unique(y_p)) > 1 else np.nan),
                    **scorer_p.score(s_p, thr),
                })
    return pd.DataFrame(rows), pd.DataFrame(prows)


def bootstrap_ci(values: np.ndarray, n_boot: int = 5000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI over patients (resample patients, not windows)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)),
            float(np.quantile(means, 1 - alpha / 2)))
