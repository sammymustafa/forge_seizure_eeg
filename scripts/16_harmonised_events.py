"""Write the harmonised event-level CSV for every dataset present locally.

One schema (:mod:`seizure_eeg.schema`) across CHB-MIT, SeizeIT2 and, once it is
implemented, TUH — so a subgroup analysis is written once and runs on all of
them. The combined file carries a ``dataset`` column and the per-field
``*_provenance`` columns, so a reader can always tell a value that is unknown
from a field the dataset never had.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from seizure_eeg.config import REPO_ROOT, TABLES_DIR
from seizure_eeg.datasets.chbmit import CHBMITAdapter
from seizure_eeg.datasets.seizeit2 import SeizeIT2Adapter
from seizure_eeg.datasets.tuh import TUSZAdapter
from seizure_eeg.schema import (EVENT_COLUMNS, events_to_frame,
                                recordings_to_frame, validate_events)


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    adapters = {
        "chbmit": CHBMITAdapter(REPO_ROOT / "data" / "chbmit"),
        "seizeit2": SeizeIT2Adapter(REPO_ROOT / "data" / "seizeit2"),
    }
    frames, avail = [], {}
    for name, ad in adapters.items():
        try:
            ev = events_to_frame(ad.events())
        except Exception as exc:  # noqa: BLE001
            print(f"{name}: skipped ({exc!r})")
            continue
        if ev.empty:
            print(f"{name}: no events found locally")
            continue
        validate_events(ev)
        ev.to_csv(TABLES_DIR / f"{name}_events_harmonised.csv", index=False)
        recordings_to_frame(ad.recordings()).to_csv(
            TABLES_DIR / f"{name}_recordings.csv", index=False)
        frames.append(ev)
        avail[name] = ad.label_availability()
        print(f"{name}: {len(ev)} events, {ev.patient_id.nunique()} patients "
              f"-> {name}_events_harmonised.csv")

    tusz = TUSZAdapter(REPO_ROOT / "data" / "tusz")
    avail["tusz (planned)"] = tusz.label_availability()
    print(f"tusz: adapter not implemented (IMPLEMENTED="
          f"{TUSZAdapter.IMPLEMENTED}); no events written")

    if frames:
        combined = pd.concat(frames, ignore_index=True)[EVENT_COLUMNS]
        validate_events(combined)
        combined.to_csv(TABLES_DIR / "all_events_harmonised.csv", index=False)
        print(f"\ncombined: {len(combined)} events across "
              f"{combined.dataset.nunique()} datasets -> "
              f"all_events_harmonised.csv")
        print(combined.groupby("dataset")
              .agg(events=("event_index", "size"),
                   patients=("patient_id", "nunique"),
                   ictal_hours=("duration_s", lambda s: round(s.sum()/3600, 1)))
              .to_string())

    am = pd.DataFrame(avail)
    am.index.name = "field"
    am.to_csv(TABLES_DIR / "label_availability_matrix.csv")
    print("\nlabel availability by dataset:")
    print(am.to_string())


if __name__ == "__main__":
    main()
