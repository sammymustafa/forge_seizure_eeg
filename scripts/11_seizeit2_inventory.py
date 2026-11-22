"""Inventory the SeizeIT2 annotations and write the harmonised event table."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from seizure_eeg.config import REPO_ROOT, TABLES_DIR
from seizure_eeg.datasets.seizeit2 import SeizeIT2Adapter, placement_confound_note
from seizure_eeg.schema import events_to_frame, recordings_to_frame, validate_events

ROOT = REPO_ROOT / "data" / "seizeit2"


def subgroup_table(ev: pd.DataFrame, col: str) -> pd.DataFrame:
    g = ev.groupby(col)
    out = pd.DataFrame({
        "n_seizures": g.size(),
        "n_patients": g.patient_id.nunique(),
        "median_duration_s": g.duration_s.median().round(1),
    }).sort_values("n_seizures", ascending=False)
    out["pct_of_seizures"] = (100 * out.n_seizures / len(ev)).round(1)
    return out


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    ad = SeizeIT2Adapter(ROOT)

    recs = ad.recordings()
    rec_df = recordings_to_frame(recs)
    rec_df.to_csv(TABLES_DIR / "seizeit2_recordings.csv", index=False)

    ev = events_to_frame(ad.events())
    validate_events(ev)
    ev.to_csv(TABLES_DIR / "seizeit2_events_harmonised.csv", index=False)

    print(f"recordings with a sidecar: {len(rec_df)}")
    print(f"patients: {rec_df.patient_id.nunique()}  "
          f"sessions: {rec_df.session_id.nunique()}")
    if len(rec_df):
        print("montages:",
              rec_df.montage_name.value_counts().to_dict())
        print("EEG channel counts:",
              rec_df.n_eeg_channels.value_counts().to_dict())
        print("sampling rates:", sorted(rec_df.sfreq.dropna().unique()))
        print("recording types:", rec_df.notes.value_counts().to_dict())
    print(f"\nseizure events: {len(ev)} in "
          f"{ev.patient_id.nunique()} patients, "
          f"{ev.recording_id.nunique()} recordings")
    print(f"total ictal time: {ev.duration_s.sum()/3600:.1f} h "
          f"(median seizure {ev.duration_s.median():.0f} s)")

    for col in ["seizure_type", "onset_lobe", "onset_hemisphere", "vigilance"]:
        print(f"\n--- by {col} ---")
        print(subgroup_table(ev, col).to_string())

    print("\n--- raw eventType codes (top 15) ---")
    print(ev.seizure_type_raw.value_counts().head(15).to_string())
    print("\n--- raw localization codes ---")
    print(ev.onset_lobe_raw.value_counts().to_string())
    print(f"\ninvolves temporal lobe: {int(ev.involves_temporal.sum())} seizures "
          f"in {ev[ev.involves_temporal].patient_id.nunique()} patients")
    print(f"involves frontal lobe: {int(ev.involves_frontal.sum())} seizures "
          f"in {ev[ev.involves_frontal].patient_id.nunique()} patients")

    # per-recording seizure counts (drives the selective EEG download)
    per_rec = (ev.groupby(["patient_id", "session_id", "recording_id"])
               .size().rename("n_seizures").reset_index())
    all_recs = rec_df[["patient_id", "session_id", "recording_id",
                       "duration_s", "montage_name"]]
    inv = all_recs.merge(per_rec, how="left",
                         on=["patient_id", "session_id", "recording_id"])
    inv["n_seizures"] = inv.n_seizures.fillna(0).astype(int)
    inv.to_csv(ROOT / "_events_inventory.csv", index=False)
    inv.to_csv(TABLES_DIR / "seizeit2_recording_inventory.csv", index=False)
    print(f"\nrecordings containing >=1 seizure: {(inv.n_seizures > 0).sum()}"
          f" of {len(inv)}")

    print("\nCAVEAT:", placement_confound_note())
    print("CAVEAT: the public release has no paired full-scalp EEG, so no "
          "full-versus-wearable comparison is possible within this dataset.")
    print("CAVEAT: all participants have focal epilepsy.")


if __name__ == "__main__":
    main()
