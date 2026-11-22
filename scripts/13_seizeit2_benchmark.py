"""Patient-disjoint seizure detection on real behind-the-ear wearable EEG,
with exploratory subgroup analysis by seizure type, onset lobe and hemisphere.

Same preprocessing, features, model and evaluation rules as the CHB-MIT
benchmark. This is **external validation on real wearable hardware**, not a
full-versus-wearable comparison: the public SeizeIT2 release contains no paired
full-scalp recording.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from seizure_eeg.benchmark import bootstrap_ci, make_model, patient_folds
from seizure_eeg.config import FEATURES_PER_CHANNEL, REPO_ROOT, TABLES_DIR
from seizure_eeg.datasets.seizeit2 import placement_confound_note
from seizure_eeg.evaluation import EventScorer
from seizure_eeg.subgroups import paired_subgroup_delta, subgroup_sensitivity
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score

FEAT = REPO_ROOT / "results" / "features_seizeit2"
TARGETS = (1.0, 6.0, 24.0)
SEED = 0


# ------------------------------------------------------------------ loading
def canonical_slots(channels: list[str]) -> tuple[list[int], str, str]:
    """Order channels as [behind-the-ear, second] and name the device side.

    Recordings carry ``BTEleft``/``BTEright`` plus ``CROSStop``, and a handful
    carry both BTE channels. Putting the behind-the-ear derivation in slot 0
    regardless of side makes the feature vector mean the same thing in every
    recording; the side itself is kept separately for the
    hemisphere-versus-placement analysis.
    """
    idx_l = next((i for i, c in enumerate(channels) if "BTELEFT" in c.upper()), None)
    idx_r = next((i for i, c in enumerate(channels) if "BTERIGHT" in c.upper()), None)
    idx_c = next((i for i, c in enumerate(channels) if "CROSS" in c.upper()), None)

    if idx_l is not None and idx_r is not None:
        return [idx_l, idx_r], "bilateral", "BTEleft+BTEright"
    bte, side = (idx_l, "left") if idx_l is not None else (idx_r, "right")
    if bte is None:
        return list(range(len(channels))), "unknown", "other"
    second = idx_c if idx_c is not None else (1 - bte)
    return [bte, second], side, "BTE+CROSStop"


def load_features() -> tuple[np.ndarray, pd.DataFrame]:
    metas, mats = [], []
    for npz in sorted(FEAT.glob("*.npz")):
        rec = npz.stem
        meta_path = FEAT / f"{rec}_meta.csv.gz"
        if not meta_path.exists():
            continue
        with np.load(npz, allow_pickle=False) as z:
            X, Q, chans = z["X"], z["Q"], [str(c) for c in z["channels"]]
        slots, side, config = canonical_slots(chans)
        if len(slots) < 2:
            continue
        m = pd.read_csv(meta_path)
        usable = Q[:, slots].all(axis=1) & np.isfinite(X[:, slots, :]).all(axis=(1, 2))
        if usable.sum() < 2:
            continue
        m = m.loc[usable].copy()
        m["device_side"] = side
        m["channel_config"] = config
        mats.append(X[np.asarray(usable)][:, slots, :].reshape(int(usable.sum()), -1))
        metas.append(m)
    if not mats:
        raise SystemExit(f"no features in {FEAT}; run scripts/12_seizeit2_features.py")
    Xall = np.concatenate(mats).astype(np.float64)
    meta = pd.concat(metas, ignore_index=True)
    return Xall, meta


def normalise_per_recording(X: np.ndarray, meta: pd.DataFrame) -> np.ndarray:
    out = np.empty_like(X)
    codes = meta.groupby("recording_id", sort=False).ngroup().to_numpy()
    for g in np.unique(codes):
        sel = codes == g
        blk = X[sel]
        med = np.median(blk, axis=0)
        q75, q25 = np.percentile(blk, [75, 25], axis=0)
        scale = (q75 - q25) / 1.349
        scale[scale < 1e-9] = 1.0
        out[sel] = (blk - med) / scale
    return out


# --------------------------------------------------------------- evaluation
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--model", default="lda")
    ap.add_argument("--min-consecutive", type=int, default=2)
    args = ap.parse_args()

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    X, meta = load_features()
    meta = meta.rename(columns={"recording_id": "edf_file"})
    meta["start_s"] = meta.start_s.astype(float)
    meta["stop_s"] = meta.stop_s.astype(float)
    X = normalise_per_recording(X, meta.rename(columns={"edf_file": "recording_id"}))

    y = meta.label.to_numpy()
    groups = meta.patient_id.to_numpy()
    print(f"{len(meta)} windows ({len(meta)*10/3600:.0f} h), "
          f"{meta.patient_id.nunique()} patients, "
          f"{meta.edf_file.nunique()} recordings, "
          f"{int(y.sum())} ictal windows ({100*y.mean():.3f}%)", flush=True)
    print("device sides:", meta.device_side.value_counts().to_dict())
    print("channel configs:", meta.channel_config.value_counts().to_dict())

    ann = pd.read_csv(TABLES_DIR / "seizeit2_events_harmonised.csv")
    ann["event_uid"] = ann.recording_id + "#" + ann.event_index.astype(str)
    seiz = ann.rename(columns={"recording_id": "edf_file"})[
        ["edf_file", "onset_s", "offset_s", "event_uid"]]
    seiz = seiz[seiz.edf_file.isin(set(meta.edf_file))]
    print(f"annotated seizures in these recordings: {len(seiz)}", flush=True)

    fold_rows, event_rows = [], []
    for fold, (tr, te) in enumerate(patient_folds(groups, args.folds), 1):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        model = make_model(args.model)
        model.fit(X[tr], y[tr])
        s_te = model.predict_proba(X[te])[:, 1]

        meta_te = meta.iloc[te].reset_index(drop=True)
        seiz_te = seiz[seiz.edf_file.isin(set(meta_te.edf_file))]
        scorer_te = EventScorer(meta_te, seiz_te, args.min_consecutive)

        s_tr = np.empty(len(tr))
        g_tr = groups[tr]
        for itr, ite in patient_folds(g_tr, min(3, len(np.unique(g_tr)))):
            m2 = make_model(args.model)
            m2.fit(X[tr][itr], y[tr][itr])
            s_tr[ite] = m2.predict_proba(X[tr][ite])[:, 1]
        meta_tr = meta.iloc[tr].reset_index(drop=True)
        seiz_tr = seiz[seiz.edf_file.isin(set(meta_tr.edf_file))]
        scorer_tr = EventScorer(meta_tr, seiz_tr, args.min_consecutive)

        for target in TARGETS:
            thr = scorer_tr.threshold_for_target_fa(s_tr, target)
            ev = scorer_te.score(s_te, thr)
            pred = (s_te >= thr).astype(int)
            fold_rows.append({
                "dataset": "seizeit2", "fold": fold, "model": args.model,
                "target_fa_per_h": target, "threshold": thr,
                "n_test_windows": int(len(te)),
                "n_test_patients": int(len(np.unique(groups[te]))),
                "auprc": float(average_precision_score(y[te], s_te)),
                "auroc": float(roc_auc_score(y[te], s_te)),
                "balanced_accuracy": float(balanced_accuracy_score(y[te], pred)),
                "prevalence": float(y[te].mean()),
                **ev,
            })
            tbl = scorer_te.event_table(s_te, thr)
            tbl["fold"] = fold
            tbl["target_fa_per_h"] = target
            event_rows.append(tbl)
        print(f"  fold {fold}: {len(np.unique(groups[te]))} test patients, "
              f"AUPRC {fold_rows[-1]['auprc']:.3f}, "
              f"sens@6 {[r for r in fold_rows if r['fold']==fold and r['target_fa_per_h']==6.0][0]['event_sensitivity']:.3f}",
              flush=True)

    folds_df = pd.DataFrame(fold_rows)
    folds_df.to_csv(TABLES_DIR / "seizeit2_cv_folds.csv", index=False)

    events = pd.concat(event_rows, ignore_index=True)
    events = events.merge(
        ann[["event_uid", "patient_id", "seizure_type", "onset_lobe",
             "onset_lobes_all", "involves_temporal", "involves_frontal",
             "onset_hemisphere", "vigilance", "duration_s",
             "seizure_type_raw", "onset_lobe_raw"]],
        on="event_uid", how="left")
    side = (meta.groupby("edf_file").device_side.first()
            .rename("device_side").reset_index()
            .rename(columns={"edf_file": "recording"}))
    events = events.merge(side, on="recording", how="left")
    events["onset_vs_device"] = np.where(
        events.onset_hemisphere.isin(["left", "right"])
        & events.device_side.isin(["left", "right"]),
        np.where(events.onset_hemisphere == events.device_side,
                 "ipsilateral", "contralateral"),
        "undetermined")
    events.to_csv(TABLES_DIR / "seizeit2_event_outcomes.csv", index=False)

    main6 = events[events.target_fa_per_h == 6.0]
    print(f"\nOverall at the 6 FA/h target: "
          f"{int(main6.detected.sum())}/{len(main6)} seizures detected "
          f"({main6.detected.mean():.3f}), "
          f"median latency {main6.loc[main6.detected,'latency_s'].median():.0f} s")

    for col in ["seizure_type", "onset_lobe", "onset_hemisphere", "vigilance",
                "onset_vs_device", "device_side"]:
        tab = subgroup_sensitivity(main6, col, seed=SEED)
        tab.insert(0, "dataset", "seizeit2")
        tab.insert(1, "target_fa_per_h", 6.0)
        tab.to_csv(TABLES_DIR / f"seizeit2_subgroup_{col}.csv", index=False)
        print(f"\n--- sensitivity by {col} (6 FA/h) ---")
        print(tab[[col, "n_seizures", "n_patients", "sensitivity",
                   "sensitivity_lo", "sensitivity_hi", "median_latency_s"]]
              .round(3).to_string(index=False))

    contrasts = []
    for col, a, b in [("onset_lobe", "temporal", "frontal"),
                      ("involves_temporal", True, False),
                      ("onset_vs_device", "ipsilateral", "contralateral"),
                      ("vigilance", "asleep", "awake"),
                      ("seizure_type", "focal_to_bilateral_tonic_clonic",
                       "focal_aware")]:
        d = paired_subgroup_delta(main6, col, a, b, seed=SEED)
        if d:
            d["grouping"] = col
            contrasts.append(d)
    cdf = pd.DataFrame(contrasts)
    cdf.to_csv(TABLES_DIR / "seizeit2_subgroup_contrasts.csv", index=False)
    print("\n--- paired contrasts (patient bootstrap) ---")
    print(cdf[["grouping", "comparison", "n_seizures_a", "n_seizures_b",
               "sensitivity_a", "sensitivity_b", "delta", "ci_lo", "ci_hi"]]
          .round(3).to_string(index=False))

    print("\nCAVEAT:", placement_confound_note())
    print("CAVEAT: no paired full-scalp EEG exists in this release, so these "
          "numbers must never be plotted against the CHB-MIT scalp proxies as "
          "interchangeable measurements.")


if __name__ == "__main__":
    main()
