"""CHB-MIT adapter — scalp EEG used as a *proxy* for wearable placements.

CHB-MIT's released annotations contain seizure onset and offset times and
nothing else. There are **no validated seizure-type or onset-lobe labels**, so
every clinical field this adapter reports is ``not_annotated``.

Anatomical origin must not be inferred from which channels show the most power,
the largest amplitude, or the highest classifier weight. That would manufacture
a clinical label out of a signal statistic, and the subgroup analyses in this
project would then be circular. If verified patient-level annotations are
published elsewhere, they can be supplied through
:func:`load_external_annotations`, which requires an explicit citation for every
row and never invents one.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import MONTAGES, patient_id
from ..dataset import extract_recording  # noqa: F401  (re-exported for callers)
from ..schema import Recording, SeizureEvent
from ..summary import parse_summary
from .base import DatasetAdapter

#: Columns an external anatomical annotation table must provide. Provenance is
#: mandatory: a row without a citation is not usable as a clinical label.
EXTERNAL_ANNOTATION_COLUMNS = [
    "case_id",            # e.g. "chb01"
    "edf_file",           # "" to apply to every recording of that case
    "seizure_index",      # -1 to apply to every seizure in the file
    "seizure_type",       # a seizure_eeg.schema.SEIZURE_TYPES value
    "onset_lobe",         # a seizure_eeg.schema.ONSET_LOBES value
    "onset_hemisphere",   # left / right / bilateral / unknown
    "source_citation",    # REQUIRED: DOI or full reference
    "source_table",       # REQUIRED: where in that source the value appears
    "notes",
]


def load_external_annotations(path: str | Path) -> pd.DataFrame:
    """Load an optional external anatomical annotation table.

    The file is user-supplied and must come from a published, verifiable source;
    this project ships none, because no validated per-seizure anatomical
    annotation of CHB-MIT is bundled with the dataset.
    """
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=EXTERNAL_ANNOTATION_COLUMNS)
    df = pd.read_csv(path, dtype=str).fillna("")
    missing = [c for c in EXTERNAL_ANNOTATION_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"external annotation table missing columns: {missing}")
    blank = df[(df.source_citation.str.strip() == "")
               | (df.source_table.str.strip() == "")]
    if len(blank):
        raise ValueError(
            f"{len(blank)} external annotation row(s) have no provenance; "
            "every row needs source_citation and source_table")
    return df


class CHBMITAdapter(DatasetAdapter):
    """Reads ``data/chbmit/chbXX/`` as produced by ``scripts/02_download.py``."""

    name = "chbmit"
    is_wearable = False     # scalp positions standing in for a wearable

    def __init__(self, root: str | Path, external_annotations: str | Path | None = None):
        super().__init__(root)
        self.external = load_external_annotations(external_annotations) \
            if external_annotations else pd.DataFrame(columns=EXTERNAL_ANNOTATION_COLUMNS)

    # ------------------------------------------------------------ discovery
    def cases(self) -> list[str]:
        return sorted(p.name for p in self.root.glob("chb*") if p.is_dir())

    def recordings(self) -> list[Recording]:
        out = []
        for case in self.cases():
            summary = self.root / case / f"{case}-summary.txt"
            if not summary.exists():
                continue
            for edf_name in parse_summary(summary):
                path = self.root / case / edf_name
                out.append(Recording(
                    dataset=self.name,
                    patient_id=patient_id(case),
                    session_id=case,
                    recording_id=edf_name.replace(".edf", ""),
                    path=str(path),
                    sfreq=256.0,
                    duration_s=float("nan"),
                    n_eeg_channels=len(MONTAGES["full_18_bipolar"]),
                    channels=list(MONTAGES["full_18_bipolar"]),
                    montage_name="bipolar 18-channel scalp",
                    placement_scheme="10-20 scalp",
                    is_wearable=False,
                    notes="EDF may be absent locally; features are cached",
                ))
        return out

    # --------------------------------------------------------------- events
    def _external_match(self, case: str, edf: str, idx: int) -> dict:
        if self.external.empty:
            return {}
        m = self.external[
            (self.external.case_id == case)
            & (self.external.edf_file.isin(["", edf]))
            & (self.external.seizure_index.isin(["-1", str(idx)]))]
        return {} if m.empty else m.iloc[-1].to_dict()

    def events(self, recording: Recording | None = None) -> list[SeizureEvent]:
        out: list[SeizureEvent] = []
        cases = [recording.session_id] if recording else self.cases()
        for case in cases:
            summary = self.root / case / f"{case}-summary.txt"
            if not summary.exists():
                continue
            for edf_name, intervals in parse_summary(summary).items():
                rec_id = edf_name.replace(".edf", "")
                if recording and rec_id != recording.recording_id:
                    continue
                for i, (a, b) in enumerate(intervals):
                    ext = self._external_match(case, edf_name, i)
                    has_ext = bool(ext)
                    out.append(SeizureEvent(
                        dataset=self.name,
                        patient_id=patient_id(case),
                        session_id=case,
                        recording_id=rec_id,
                        event_index=i,
                        onset_s=float(a),
                        offset_s=float(b),
                        seizure_type=ext.get("seizure_type") or "unclassified",
                        seizure_type_raw=ext.get("seizure_type", ""),
                        seizure_type_provenance="annotated" if has_ext else "not_annotated",
                        onset_lobe=ext.get("onset_lobe") or "unknown",
                        onset_lobe_raw=ext.get("onset_lobe", ""),
                        onset_lobe_provenance="annotated" if has_ext else "not_annotated",
                        onset_hemisphere=ext.get("onset_hemisphere") or "unknown",
                        onset_hemisphere_raw=ext.get("onset_hemisphere", ""),
                        onset_hemisphere_provenance="annotated" if has_ext else "not_annotated",
                        vigilance="unknown",
                        vigilance_provenance="not_annotated",
                        notes=ext.get("source_citation", ""),
                    ))
        return out

    # -------------------------------------------------------------- signals
    def load_signals(self, recording: Recording, channels: list[str] | None = None):
        import mne
        from ..dataset import canonical_channel_map, referential_map
        mne.set_log_level("ERROR")
        raw = mne.io.read_raw_edf(recording.path, preload=True, verbose="ERROR")
        present = canonical_channel_map(raw.ch_names)
        ref = referential_map(raw.ch_names)
        want = channels or MONTAGES["full_18_bipolar"]
        rows, names = [], []
        for ch in want:
            if ch in present:
                rows.append(raw.get_data(picks=[raw.ch_names.index(present[ch])])[0])
                names.append(ch)
                continue
            a, _, b = ch.partition("-")
            if a in ref and b in ref:
                rows.append(raw.get_data(picks=[raw.ch_names.index(ref[a])])[0]
                            - raw.get_data(picks=[raw.ch_names.index(ref[b])])[0])
                names.append(ch)
        sfreq = float(raw.info["sfreq"])
        raw.close()
        return (np.asarray(rows) if rows else np.empty((0, 0))), sfreq, names

    # ----------------------------------------------------------- dataset info
    def montages(self) -> dict[str, list[str]]:
        return dict(MONTAGES)

    def label_availability(self) -> dict[str, str]:
        has_ext = not self.external.empty
        clinical = "annotated" if has_ext else "not_annotated"
        return {
            "seizure_type": clinical,
            "onset_lobe": clinical,
            "onset_hemisphere": clinical,
            "vigilance": "not_annotated",
            "channel_level_annotation": "not_annotated",
            "paired_full_scalp_eeg": "annotated",   # this IS the full scalp EEG
        }
