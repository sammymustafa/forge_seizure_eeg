"""Fetch SeizeIT2 (OpenNeuro ds005873) from the public S3 bucket.

Metadata first, signals only on request — the full dataset is ~126 GB.

    python scripts/10_seizeit2_fetch.py metadata            # ~5700 tiny files
    python scripts/10_seizeit2_fetch.py eeg --subjects sub-001,sub-002
    python scripts/10_seizeit2_fetch.py eeg --with-seizures --max-gb 20
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from seizure_eeg.config import REPO_ROOT
from seizure_eeg.datasets.openneuro import fetch_key, list_keys

DATASET = "ds005873"
PREFIX = f"{DATASET}/"
ROOT = REPO_ROOT / "data" / "seizeit2"
KEYS_CACHE = ROOT / "_s3_keys.json"
MIN_FREE_GB = 5.0


def all_keys(refresh: bool = False) -> list[tuple[str, int]]:
    if KEYS_CACHE.exists() and not refresh:
        return [tuple(x) for x in json.loads(KEYS_CACHE.read_text())]
    ROOT.mkdir(parents=True, exist_ok=True)
    keys = list_keys(PREFIX)
    KEYS_CACHE.write_text(json.dumps(keys))
    return keys


def _download(keys: list[tuple[str, int]], workers: int, label: str) -> int:
    done = failed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(fetch_key, k, ROOT, strip_prefix=PREFIX,
                            expected_size=s): k for k, s in keys}
        for fut in as_completed(futs):
            try:
                fut.result()
                done += 1
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"  FAILED {futs[fut]}: {exc!r}", flush=True)
            if done % 250 == 0:
                free = shutil.disk_usage(ROOT).free / 1e9
                print(f"  {label}: {done}/{len(keys)} ({free:.1f} GB free)",
                      flush=True)
                if free < MIN_FREE_GB:
                    print("  stopping: low disk", flush=True)
                    for f in futs:
                        f.cancel()
                    break
    print(f"{label}: {done} downloaded, {failed} failed", flush=True)
    return done


def cmd_metadata(args) -> None:
    keys = all_keys(args.refresh)
    wanted = [(k, s) for k, s in keys
              if k.endswith(("_events.tsv", "_eeg.json"))
              or k.count("/") == 1]       # root-level dataset files
    print(f"{len(wanted)} metadata objects "
          f"({sum(s for _, s in wanted)/1e6:.1f} MB)", flush=True)
    _download(wanted, args.workers, "metadata")


def cmd_eeg(args) -> None:
    keys = all_keys(args.refresh)
    edf = [(k, s) for k, s in keys if k.endswith("_eeg.edf")]

    if args.subjects:
        want = set(args.subjects.split(","))
        edf = [(k, s) for k, s in edf if k.split("/")[1] in want]

    if args.with_seizures:
        inv = ROOT / "_events_inventory.csv"
        if not inv.exists():
            raise SystemExit("run scripts/11_seizeit2_inventory.py first")
        ev = pd.read_csv(inv)
        with_sz = set(ev[ev.n_seizures > 0].recording_id)
        edf = [(k, s) for k, s in edf
               if Path(k).name.replace("_eeg.edf", "") in with_sz]

    edf.sort(key=lambda kv: kv[1])
    if args.max_gb:
        budget, keep = args.max_gb * 1e9, []
        for k, s in edf:
            if budget - s < 0:
                continue
            budget -= s
            keep.append((k, s))
        edf = keep
    print(f"{len(edf)} EEG recordings ({sum(s for _, s in edf)/1e9:.1f} GB)",
          flush=True)
    if args.dry_run:
        return
    _download(edf, args.workers, "eeg")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    m = sub.add_parser("metadata")
    m.set_defaults(func=cmd_metadata)
    e = sub.add_parser("eeg")
    e.add_argument("--subjects", default="")
    e.add_argument("--with-seizures", action="store_true")
    e.add_argument("--max-gb", type=float, default=0.0)
    e.add_argument("--dry-run", action="store_true")
    e.set_defaults(func=cmd_eeg)
    for p in (m, e):
        p.add_argument("--workers", type=int, default=12)
        p.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    args.func(args)


if __name__ == "__main__":
    main()
