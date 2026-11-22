"""Featurise every downloaded EDF into per-case feature tensors.

Writes ``results/features/<case>.npz`` (X, Q, channels, feature_names) and
``results/features/<case>_meta.csv.gz`` (one row per 10 s window).
"""
from __future__ import annotations

import argparse
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from seizure_eeg.config import DATA_ROOT, FEATURES_DIR, FEATURES_PER_CHANNEL
from seizure_eeg.dataset import UNION_CHANNELS, extract_recording
from seizure_eeg.summary import parse_summary


def process_case(case: str) -> dict:
    case_dir = DATA_ROOT / case
    summary = parse_summary(case_dir / f"{case}-summary.txt")
    edfs = sorted(p for p in case_dir.glob("*.edf") if p.name in summary)
    if not edfs:
        return {"case": case, "n_files": 0, "status": "no-edf"}

    Xs, Qs, metas, skipped = [], [], [], []
    for edf in edfs:
        try:
            X, Q, meta = extract_recording(edf, summary[edf.name])
        except Exception as exc:  # noqa: BLE001
            skipped.append(f"{edf.name}: {exc!r}")
            continue
        Xs.append(X)
        Qs.append(Q)
        metas.append(meta)

    if not Xs:
        return {"case": case, "n_files": 0, "status": "all-failed",
                "skipped": skipped}

    X = np.concatenate(Xs, axis=0)
    Q = np.concatenate(Qs, axis=0)
    meta = pd.concat(metas, ignore_index=True)

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        FEATURES_DIR / f"{case}.npz",
        X=X, Q=Q,
        channels=np.array(UNION_CHANNELS),
        feature_names=np.array(FEATURES_PER_CHANNEL),
    )
    meta.to_csv(FEATURES_DIR / f"{case}_meta.csv.gz", index=False)
    return {
        "case": case,
        "n_files": len(Xs),
        "n_windows": len(meta),
        "n_ictal_windows": int(meta.label.sum()),
        "hours": float(len(meta) * 10 / 3600),
        "status": "ok",
        "skipped": skipped,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    cases = (args.cases.split(",") if args.cases
             else sorted(p.name for p in DATA_ROOT.iterdir() if p.is_dir()))
    print(f"featurising {len(cases)} cases with {args.workers} workers", flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(process_case, c): c for c in cases}
        for fut in as_completed(futs):
            case = futs[fut]
            try:
                res = fut.result()
            except Exception:  # noqa: BLE001
                traceback.print_exc()
                res = {"case": case, "status": "crashed"}
            rows.append(res)
            print(f"  {case}: {res.get('status')} files={res.get('n_files')} "
                  f"windows={res.get('n_windows')} ictal={res.get('n_ictal_windows')}",
                  flush=True)
            for s in res.get("skipped", []):
                print(f"     skipped {s}", flush=True)

    df = pd.DataFrame(rows).sort_values("case")
    df.drop(columns=["skipped"], errors="ignore").to_csv(
        FEATURES_DIR / "extraction_report.csv", index=False)
    print(df.drop(columns=["skipped"], errors="ignore").to_string(index=False))
    tot = df.get("n_windows", pd.Series(dtype=float)).sum()
    print(f"\nTOTAL windows={tot:.0f} ({tot*10/3600:.1f} h), "
          f"ictal={df.get('n_ictal_windows', pd.Series(dtype=float)).sum():.0f}")


if __name__ == "__main__":
    main()
