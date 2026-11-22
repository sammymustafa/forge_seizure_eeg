"""Download CHB-MIT EDF recordings listed in the subject summaries.

Usage:
    python scripts/02_download.py [--cases chb01,chb02] [--seizure-only] [--workers 6]
"""
from __future__ import annotations

import argparse
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from seizure_eeg.config import DATA_ROOT, TABLES_DIR
from seizure_eeg.download import fetch

MIN_FREE_GB = 6.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="", help="comma-separated case ids; default all")
    ap.add_argument("--seizure-only", action="store_true",
                    help="only files that contain at least one annotated seizure")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    inv = pd.read_csv(TABLES_DIR / "file_inventory.csv")
    if args.cases:
        wanted = set(args.cases.split(","))
        inv = inv[inv.case.isin(wanted)]
    if args.seizure_only:
        inv = inv[inv.n_seizures > 0]
    targets = [f"{r.case}/{r.edf_file}" for r in inv.itertuples()]
    print(f"{len(targets)} EDF files queued", flush=True)

    done = fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch, t): t for t in targets}
        for fut in as_completed(futures):
            rel = futures[fut]
            try:
                fut.result()
                done += 1
            except Exception as exc:  # noqa: BLE001
                fail += 1
                print(f"FAILED {rel}: {exc!r}", flush=True)
            if done % 25 == 0 or fail:
                free = shutil.disk_usage(DATA_ROOT).free / 1e9
                print(f"  {done}/{len(targets)} ok, {fail} failed, "
                      f"{free:.1f} GB free", flush=True)
                if free < MIN_FREE_GB:
                    print("Stopping: low disk", flush=True)
                    for f in futures:
                        f.cancel()
                    break
    total = sum(p.stat().st_size for p in DATA_ROOT.rglob("*.edf"))
    print(f"done={done} failed={fail} on-disk={total/1e9:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
