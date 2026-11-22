"""Harmonised, dataset-agnostic schema for seizure events and recordings.

One CSV layout is used for CHB-MIT, SeizeIT2 and (later) TUH so that subgroup
analyses can be written once. Two rules shape it:

1. **Raw labels are never destroyed.** Every harmonised field is accompanied by
   the source dataset's own string in a ``*_raw`` column.
2. **Missing is not a category.** ``unknown`` means the dataset says it is
   unknown; ``not_annotated`` means the dataset has no such field at all. They
   are different facts and must not be pooled, so ``*_provenance`` records which
   applies.

Multilobar onsets are kept as ``multilobar`` rather than being collapsed into
whichever lobe is listed first; ``involves_temporal`` / ``involves_frontal``
exist for the "is the temporal lobe implicated at all?" question.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Literal

import pandas as pd

Provenance = Literal[
    "annotated",        # the dataset supplies a validated value
    "unknown",          # the dataset supplies an explicit "unknown" value
    "not_annotated",    # the dataset has no such field
    "derived",          # harmonised from another annotated field
]

SEIZURE_TYPES = [
    "focal_aware",
    "focal_impaired_awareness",
    "focal_unknown_awareness",
    "focal_to_bilateral_tonic_clonic",
    "focal_unspecified",
    "generalized",
    "unknown_onset",
    "unclassified",
]

ONSET_LOBES = [
    "temporal", "frontal", "central", "parietal", "occipital", "insular",
    "multilobar", "unknown",
]

HEMISPHERES = ["left", "right", "bilateral", "unknown"]

VIGILANCE = ["awake", "asleep", "unknown"]


@dataclass
class SeizureEvent:
    """One annotated seizure, harmonised across datasets."""

    dataset: str
    patient_id: str                  # the PERSON (CV grouping key)
    session_id: str
    recording_id: str
    event_index: int

    onset_s: float
    offset_s: float

    # --- seizure type
    seizure_type: str = "unclassified"
    seizure_type_raw: str = ""
    seizure_type_provenance: Provenance = "not_annotated"

    # --- anatomical onset
    onset_lobe: str = "unknown"
    onset_lobe_raw: str = ""
    onset_lobe_provenance: Provenance = "not_annotated"
    onset_lobes_all: str = ""        # ";"-joined, for multilobar onsets
    involves_temporal: bool = False
    involves_frontal: bool = False

    # --- hemisphere
    onset_hemisphere: str = "unknown"
    onset_hemisphere_raw: str = ""
    onset_hemisphere_provenance: Provenance = "not_annotated"

    # --- state
    vigilance: str = "unknown"
    vigilance_raw: str = ""
    vigilance_provenance: Provenance = "not_annotated"

    # --- other
    channels_raw: str = ""           # channel-level annotation, if the dataset has one
    confidence: float = float("nan")
    notes: str = ""

    @property
    def duration_s(self) -> float:
        return self.offset_s - self.onset_s


@dataclass
class Recording:
    """One continuous(ly indexed) recording file."""

    dataset: str
    patient_id: str
    session_id: str
    recording_id: str
    path: str
    sfreq: float
    duration_s: float
    n_eeg_channels: int
    channels: list[str] = field(default_factory=list)
    montage_name: str = ""           # e.g. "behind-the-ear left", "bipolar 18ch"
    placement_scheme: str = ""
    is_wearable: bool = False        # real wearable hardware, not a scalp proxy
    notes: str = ""


EVENT_COLUMNS = [f.name for f in fields(SeizureEvent)] + ["duration_s"]
RECORDING_COLUMNS = [f.name for f in fields(Recording)]


def events_to_frame(events: list[SeizureEvent]) -> pd.DataFrame:
    rows = []
    for e in events:
        d = asdict(e)
        d["duration_s"] = e.duration_s
        rows.append(d)
    df = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    return df


def recordings_to_frame(recs: list[Recording]) -> pd.DataFrame:
    rows = []
    for r in recs:
        d = asdict(r)
        d["channels"] = ";".join(r.channels)
        rows.append(d)
    return pd.DataFrame(rows, columns=RECORDING_COLUMNS)


def validate_events(df: pd.DataFrame) -> None:
    """Fail loudly on anything that would corrupt a subgroup analysis."""
    missing = [c for c in EVENT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"event table missing columns: {missing}")
    bad = set(df.seizure_type) - set(SEIZURE_TYPES)
    if bad:
        raise ValueError(f"unknown seizure_type values: {sorted(bad)}")
    bad = set(df.onset_lobe) - set(ONSET_LOBES)
    if bad:
        raise ValueError(f"unknown onset_lobe values: {sorted(bad)}")
    bad = set(df.onset_hemisphere) - set(HEMISPHERES)
    if bad:
        raise ValueError(f"unknown onset_hemisphere values: {sorted(bad)}")
    bad = set(df.vigilance) - set(VIGILANCE)
    if bad:
        raise ValueError(f"unknown vigilance values: {sorted(bad)}")
    if (df.offset_s <= df.onset_s).any():
        raise ValueError("non-positive seizure duration present")
