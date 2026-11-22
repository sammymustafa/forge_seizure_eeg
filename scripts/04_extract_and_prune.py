"""Featurise each case as soon as its EDFs finish downloading, then free the raw
EDFs again so the full 24-subject database fits in the session's disk budget.

``--keep chb01`` retains one subject's raw recordings for waveform figures.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

from seizure_eeg.config import DATA_ROOT, FEATURES_DIR, TABLES_DIR

extract_mod = import_module("03_extract_features")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", default="chb01", help="comma-separated cases to keep raw")
    ap.add_argument("--poll", type=int, default=20)
    ap.add_argument("--timeout-min", type=int, default=90)
    args = ap.parse_args()
    keep = set(args.keep.split(",")) if args.keep else set()

    inv = pd.read_csv(TABLES_DIR / "file_inventory.csv")
    expected = inv.groupby("case").edf_file.apply(set).to_dict()
    todo = sorted(expected)
    done: list[str] = []
    reports = []
    deadline = time.time() + args.timeout_min * 60

    while todo and time.time() < deadline:
        progressed = False
        for case in list(todo):
            case_dir = DATA_ROOT / case
            have = {p.name for p in case_dir.glob("*.edf")} if case_dir.exists() else set()
            if not expected[case] <= have:
                continue
            t0 = time.time()
            res = extract_mod.process_case(case)
            reports.append(res)
            print(f"[{len(done)+1}/{len(expected)}] {case}: {res['status']} "
                  f"files={res.get('n_files')} windows={res.get('n_windows')} "
                  f"ictal={res.get('n_ictal_windows')} ({time.time()-t0:.0f}s)",
                  flush=True)
            for s in res.get("skipped", []):
                print(f"    skipped {s}", flush=True)
            if res["status"] == "ok" and case not in keep:
                freed = sum(p.stat().st_size for p in case_dir.glob("*.edf"))
                for p in case_dir.glob("*.edf"):
                    p.unlink()
                print(f"    pruned {freed/1e9:.1f} GB of raw EDF for {case}; "
                      f"{shutil.disk_usage(DATA_ROOT).free/1e9:.1f} GB free",
                      flush=True)
            todo.remove(case)
            done.append(case)
            progressed = True
        if todo and not progressed:
            time.sleep(args.poll)

    pd.DataFrame(reports).drop(columns=["skipped"], errors="ignore").to_csv(
        FEATURES_DIR / "extraction_report.csv", index=False)
    print(f"\nfinished={len(done)} pending={todo}", flush=True)


if __name__ == "__main__":
    main()
