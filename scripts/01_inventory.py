"""Download every CHB-MIT subject summary and inventory seizures + montages."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from seizure_eeg.config import DATA_ROOT, MONTAGES, TABLES_DIR
from seizure_eeg.download import fetch_summary
from seizure_eeg.summary import parse_summary, summary_channels

CASES = [f"chb{i:02d}" for i in range(1, 25)]


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    rows, file_rows = [], []
    for case in CASES:
        path = fetch_summary(case)
        anns = parse_summary(path)
        chan_blocks = summary_channels(path)
        declared = {c for blk in chan_blocks for c in blk}
        n_seiz = sum(len(v) for v in anns.values())
        dur = sum(b - a for v in anns.values() for a, b in v)
        montage_ok = {
            name: all(ch in declared for ch in chans)
            for name, chans in MONTAGES.items()
        }
        rows.append({
            "case": case,
            "n_files_in_summary": len(anns),
            "n_seizure_files": sum(1 for v in anns.values() if v),
            "n_seizures": n_seiz,
            "ictal_seconds": dur,
            "n_channel_blocks": len(chan_blocks),
            **{f"has_{k}": v for k, v in montage_ok.items()},
        })
        for fname, seiz in anns.items():
            file_rows.append({
                "case": case, "edf_file": fname, "n_seizures": len(seiz),
                "ictal_seconds": sum(b - a for a, b in seiz),
                "intervals": ";".join(f"{a}-{b}" for a, b in seiz),
            })
        print(f"{case}: {len(anns):3d} files, {n_seiz:2d} seizures, "
              f"{dur:5d} ictal s, {len(chan_blocks)} channel block(s)")

    inv = pd.DataFrame(rows)
    files = pd.DataFrame(file_rows)
    inv.to_csv(TABLES_DIR / "subject_inventory.csv", index=False)
    files.to_csv(TABLES_DIR / "file_inventory.csv", index=False)
    print("\nTotals: {} files, {} seizures, {:.1f} ictal hours".format(
        len(files), int(files.n_seizures.sum()), files.ictal_seconds.sum() / 3600))
    print("\nMontage availability (by declared channels):")
    for name in MONTAGES:
        col = f"has_{name}"
        ok = inv.loc[inv[col], "case"].tolist()
        print(f"  {name:32s} {len(ok):2d}/24  missing: "
              f"{sorted(set(inv.case) - set(ok))}")
    print(f"\nWrote {TABLES_DIR/'subject_inventory.csv'} and "
          f"{TABLES_DIR/'file_inventory.csv'}")
    print(f"Data root: {DATA_ROOT}")


if __name__ == "__main__":
    main()
