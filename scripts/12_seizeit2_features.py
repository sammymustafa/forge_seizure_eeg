"""Featurise the downloaded SeizeIT2 behind-the-ear recordings.

Uses the same preprocessing and feature set as the CHB-MIT benchmark
(:mod:`seizure_eeg.features`): 0.5-40 Hz band-pass, non-overlapping 10 s
windows, band power / RMS / line length / spectral entropy per channel.

Each recording carries two channels — one behind-the-ear derivation
(``BTEleft``/``BTEright``) and one cross-head derivation (``CROSStop``) — so
features are stored per recording with the channel names kept.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import mne
import numpy as np
import pandas as pd

from seizure_eeg.config import FEATURES_PER_CHANNEL, REPO_ROOT, WINDOW_S
from seizure_eeg.datasets.seizeit2 import SeizeIT2Adapter
from seizure_eeg.features import bandpass, window_features, window_quality

mne.set_log_level("ERROR")

ROOT = REPO_ROOT / "data" / "seizeit2"
OUT = REPO_ROOT / "results" / "features_seizeit2"


def featurise(path: str, events: list[tuple[float, float, int]]):
    raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
    sfreq = float(raw.info["sfreq"])
    n_samp = int(round(WINDOW_S * sfreq))
    n_win = int(raw.n_times // n_samp)
    if n_win < 2:
        raw.close()
        raise ValueError("shorter than two windows")
    names = list(raw.ch_names)
    data = raw.get_data(start=0, stop=n_win * n_samp)
    raw.close()

    windows_raw = data.reshape(len(names), n_win, n_samp)
    q = window_quality(windows_raw)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    filt = bandpass(data, sfreq)
    feats = window_features(filt.reshape(len(names), n_win, n_samp), sfreq)

    X = np.transpose(feats, (1, 0, 2)).astype(np.float32)     # (win, ch, feat)
    Q = q.T                                                    # (win, ch)

    starts = np.arange(n_win) * WINDOW_S
    stops = starts + WINDOW_S
    overlap = np.zeros(n_win)
    ev_idx = np.full(n_win, -1, dtype=np.int32)
    for onset, offset, uid in events:
        ov = np.minimum(stops, offset) - np.maximum(starts, onset)
        ov = np.clip(ov, 0, None)
        hit = ov > 0
        ev_idx[hit & (ev_idx < 0)] = uid
        overlap = np.maximum(overlap, ov)

    meta = pd.DataFrame({
        "start_s": starts, "stop_s": stops, "sfreq": sfreq,
        "ictal_overlap_s": overlap, "event_uid": ev_idx,
        "label": (overlap > 0).astype(np.int8),
        "n_channels_ok": Q.sum(axis=1).astype(np.int16),
    })
    return X, Q, meta, names


def process_recording(rec_id: str) -> dict:
    ad = SeizeIT2Adapter(ROOT)
    recs = {r.recording_id: r for r in ad.recordings()}
    rec = recs.get(rec_id)
    if rec is None or not Path(rec.path).exists():
        return {"recording_id": rec_id, "status": "missing-edf"}
    evs = ad.events(rec)
    triples = [(e.onset_s, e.offset_s, i) for i, e in enumerate(evs)]
    try:
        X, Q, meta, names = featurise(rec.path, triples)
    except Exception as exc:  # noqa: BLE001
        return {"recording_id": rec_id, "status": f"failed: {exc!r}"}

    meta.insert(0, "recording_id", rec_id)
    meta.insert(0, "session_id", rec.session_id)
    meta.insert(0, "patient_id", rec.patient_id)
    meta["montage_name"] = rec.montage_name
    meta["event_uid"] = meta.event_uid.map(
        lambda i: f"{rec_id}#{i}" if i >= 0 else "")

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{rec_id}.npz", X=X, Q=Q,
                        channels=np.array(names),
                        feature_names=np.array(FEATURES_PER_CHANNEL))
    meta.to_csv(OUT / f"{rec_id}_meta.csv.gz", index=False)
    return {"recording_id": rec_id, "status": "ok", "n_windows": len(meta),
            "n_ictal": int(meta.label.sum()), "channels": ";".join(names),
            "patient_id": rec.patient_id, "montage": rec.montage_name}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    ad = SeizeIT2Adapter(ROOT)
    recs = [r for r in ad.recordings() if Path(r.path).exists()]
    ids = [r.recording_id for r in recs]
    if args.limit:
        ids = ids[:args.limit]
    print(f"featurising {len(ids)} recordings with {args.workers} workers",
          flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(process_recording, i): i for i in ids}
        for n, fut in enumerate(as_completed(futs), 1):
            try:
                res = fut.result()
            except Exception:  # noqa: BLE001
                traceback.print_exc()
                res = {"recording_id": futs[fut], "status": "crashed"}
            rows.append(res)
            if n % 25 == 0 or res["status"] != "ok":
                print(f"  {n}/{len(ids)} {res['recording_id']}: "
                      f"{res['status']}", flush=True)

    rep = pd.DataFrame(rows)
    rep.to_csv(OUT / "extraction_report.csv", index=False)
    ok = rep[rep.status == "ok"]
    print(f"\nok={len(ok)} failed={len(rep)-len(ok)}")
    if len(ok):
        print(f"windows={int(ok.n_windows.sum())} "
              f"({ok.n_windows.sum()*10/3600:.1f} h) "
              f"ictal={int(ok.n_ictal.sum())}")
        print("channel sets:", ok.channels.value_counts().to_dict())


if __name__ == "__main__":
    main()
