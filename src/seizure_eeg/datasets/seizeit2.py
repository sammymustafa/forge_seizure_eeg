"""SeizeIT2 (OpenNeuro ds005873) — real behind-the-ear wearable EEG.

What this dataset is, and is not
--------------------------------
* The public release contains the **wearable** signals (two-channel
  behind-the-ear EEG per montage, plus ECG/EMG/movement) and the clinical
  seizure annotations. It does **not** contain the paired full-scalp video-EEG
  that was recorded simultaneously to establish those annotations. So this
  dataset supports *external validation of a real wearable*, never a
  same-recording full-versus-wearable ablation.
* Every participant has **focal** epilepsy. It cannot support focal-versus-
  generalised comparisons.
* Behind-the-ear sensors were positioned with reference to the suspected onset
  zone, so a difference between anatomical subgroups is confounded with
  electrode placement. :func:`placement_confound_note` returns that caveat so
  analyses can carry it.

Annotation harmonisation follows the ILAE-2017 style ``eventType`` codes in the
dataset's own ``events.json``; see :mod:`seizure_eeg.schema` for the rules.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..schema import Recording, SeizureEvent
from .base import DatasetAdapter

S3_BASE = "https://s3.amazonaws.com/openneuro.org/ds005873/"
DATASET_ID = "ds005873"

NON_SEIZURE_EVENTS = {"bckg", "impd"}
MISSING = {"n/a", "", "nan", "None", None}

# ---------------------------------------------------------------- type map
#: ``eventType`` prefix -> harmonised seizure type. Longest prefix wins, so the
#: focal-to-bilateral code is matched before the generic focal ones.
_TYPE_PREFIXES = [
    ("sz_foc_f2b", "focal_to_bilateral_tonic_clonic"),
    ("sz_foc_a", "focal_aware"),
    ("sz_foc_ia", "focal_impaired_awareness"),
    ("sz_foc_ua", "focal_unknown_awareness"),
    ("sz_foc", "focal_unspecified"),
    ("sz_gen", "generalized"),
    ("sz_uo", "unknown_onset"),
    ("sz", "unclassified"),
]


def harmonise_seizure_type(raw: str) -> str:
    """Map a SeizeIT2 ``eventType`` code to a harmonised type.

    ``sz_foc_a`` and ``sz_foc_ia`` share a prefix, so matching is on complete
    underscore-separated segments, not on raw string prefixes.
    """
    token = (raw or "").strip()
    parts = token.split("_")
    best = ("unclassified", -1)
    for prefix, label in _TYPE_PREFIXES:
        p = prefix.split("_")
        if parts[:len(p)] == p and len(p) > best[1]:
            best = (label, len(p))
    return best[0]


# ---------------------------------------------------------------- lobe map
_LOBE_TOKENS = {
    "front": "frontal",
    "temp": "temporal",
    "cen": "central",
    "par": "parietal",
    "occ": "occipital",
    "ins": "insular",
}


def harmonise_localization(raw: str) -> tuple[str, list[str]]:
    """Return ``(onset_lobe, all_lobes)``.

    A multilobar code such as ``front_cen_temp`` becomes ``("multilobar",
    ["frontal", "central", "temporal"])`` — it is never collapsed to whichever
    lobe happens to be listed first.
    """
    token = (raw or "").strip().lower()
    if token in MISSING or token == "un":
        return "unknown", []
    lobes: list[str] = []
    for part in token.split("_"):
        lobe = _LOBE_TOKENS.get(part)
        if lobe and lobe not in lobes:
            lobes.append(lobe)
    if not lobes:
        return "unknown", []
    if len(lobes) == 1:
        return lobes[0], lobes
    return "multilobar", lobes


_HEMI = {"left": "left", "right": "right", "bi": "bilateral", "un": "unknown"}
_VIGIL = {"awake": "awake", "asleep": "asleep", "un": "unknown"}


class SeizeIT2Adapter(DatasetAdapter):
    """Reads a (possibly partial) local BIDS tree of ds005873."""

    name = "seizeit2"
    is_wearable = True

    def __init__(self, root: str | Path):
        super().__init__(root)
        self._events_cache: list[SeizureEvent] | None = None
        self._rec_cache: list[Recording] | None = None

    # ------------------------------------------------------------ discovery
    def _entities(self, path: Path) -> dict[str, str]:
        out = {}
        for part in path.name.split("_"):
            if "-" in part:
                k, _, v = part.partition("-")
                out[k] = v
        return out

    def sidecars(self) -> list[Path]:
        """Every ``*_eeg.json`` sidecar present locally."""
        return sorted(self.root.glob("sub-*/ses-*/eeg/*_eeg.json"))

    def recordings(self) -> list[Recording]:
        if self._rec_cache is not None:
            return self._rec_cache
        recs = []
        for js in self.sidecars():
            meta = json.loads(js.read_text())
            ent = self._entities(js)
            edf = js.with_name(js.name.replace("_eeg.json", "_eeg.edf"))
            montage = str(meta.get("EEGMontage", "")).strip()
            recs.append(Recording(
                dataset=self.name,
                patient_id=f"sub-{ent.get('sub', '?')}",
                session_id=f"ses-{ent.get('ses', '?')}",
                recording_id=js.name.replace("_eeg.json", ""),
                path=str(edf),
                sfreq=float(meta.get("SamplingFrequency", float("nan"))),
                duration_s=float(meta.get("RecordingDuration", float("nan"))),
                n_eeg_channels=int(meta.get("EEGChannelCount", 0)),
                channels=[],
                montage_name=f"behind-the-ear {montage}".strip(),
                placement_scheme=str(meta.get("EEGPlacementScheme", "")),
                is_wearable=True,
                notes=str(meta.get("RecordingType", "")),
            ))
        self._rec_cache = recs
        return recs

    # --------------------------------------------------------------- events
    def _events_tsv(self, recording_id: str) -> Path | None:
        hits = list(self.root.glob(f"sub-*/ses-*/eeg/{recording_id}_events.tsv"))
        return hits[0] if hits else None

    def events(self, recording: Recording | None = None) -> list[SeizureEvent]:
        if recording is not None:
            tsv = self._events_tsv(recording.recording_id)
            return [] if tsv is None else self._parse_events(tsv)
        if self._events_cache is None:
            out: list[SeizureEvent] = []
            for tsv in sorted(self.root.glob("sub-*/ses-*/eeg/*_events.tsv")):
                out.extend(self._parse_events(tsv))
            self._events_cache = out
        return self._events_cache

    def _parse_events(self, tsv: Path) -> list[SeizureEvent]:
        df = pd.read_csv(tsv, sep="\t", dtype=str).fillna("n/a")
        ent = self._entities(tsv)
        rec_id = tsv.name.replace("_events.tsv", "")
        out: list[SeizureEvent] = []
        idx = 0
        for r in df.itertuples():
            etype = str(getattr(r, "eventType", "n/a")).strip()
            if etype in NON_SEIZURE_EVENTS or etype in MISSING:
                continue
            try:
                onset = float(r.onset)
                dur = float(r.duration)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(onset) or not np.isfinite(dur) or dur <= 0:
                continue

            loc_raw = str(getattr(r, "localization", "n/a")).strip()
            lobe, lobes = harmonise_localization(loc_raw)
            lat_raw = str(getattr(r, "lateralization", "n/a")).strip()
            vig_raw = str(getattr(r, "vigilance", "n/a")).strip()
            conf = str(getattr(r, "confidence", "n/a")).strip()

            out.append(SeizureEvent(
                dataset=self.name,
                patient_id=f"sub-{ent.get('sub', '?')}",
                session_id=f"ses-{ent.get('ses', '?')}",
                recording_id=rec_id,
                event_index=idx,
                onset_s=onset,
                offset_s=onset + dur,
                seizure_type=harmonise_seizure_type(etype),
                seizure_type_raw=etype,
                seizure_type_provenance="annotated",
                onset_lobe=lobe,
                onset_lobe_raw=loc_raw,
                onset_lobe_provenance=("unknown" if lobe == "unknown"
                                       else "annotated"),
                onset_lobes_all=";".join(lobes),
                involves_temporal="temporal" in lobes,
                involves_frontal="frontal" in lobes,
                onset_hemisphere=_HEMI.get(lat_raw, "unknown"),
                onset_hemisphere_raw=lat_raw,
                onset_hemisphere_provenance=(
                    "annotated" if lat_raw in {"left", "right", "bi"}
                    else "unknown"),
                vigilance=_VIGIL.get(vig_raw, "unknown"),
                vigilance_raw=vig_raw,
                vigilance_provenance=("annotated"
                                      if vig_raw in {"awake", "asleep"}
                                      else "unknown"),
                channels_raw=str(getattr(r, "channels", "n/a")).strip(),
                confidence=float(conf) if conf not in MISSING else float("nan"),
            ))
            idx += 1
        return out

    # -------------------------------------------------------------- signals
    def load_signals(self, recording: Recording, channels: list[str] | None = None):
        import mne
        mne.set_log_level("ERROR")
        raw = mne.io.read_raw_edf(recording.path, preload=True, verbose="ERROR")
        names = list(raw.ch_names)
        if channels:
            picks = [names.index(c) for c in channels if c in names]
            names = [names[i] for i in picks]
            data = raw.get_data(picks=picks)
        else:
            data = raw.get_data()
        sfreq = float(raw.info["sfreq"])
        raw.close()
        return data, sfreq, names

    # ----------------------------------------------------------- dataset info
    def montages(self) -> dict[str, list[str]]:
        """SeizeIT2 ships the wearable montage itself; there is nothing to reduce.

        Each recording is a two-channel behind-the-ear derivation on one side,
        so the "montage" is a property of the file, not something the analysis
        selects. Keys are the montage names found in the local sidecars.
        """
        found = sorted({r.montage_name for r in self.recordings()})
        return {m: [] for m in found}

    def label_availability(self) -> dict[str, str]:
        return {
            "seizure_type": "annotated",
            "onset_lobe": "annotated",
            "onset_hemisphere": "annotated",
            "vigilance": "annotated",
            "channel_level_annotation": "not_annotated",
            "paired_full_scalp_eeg": "not_annotated",
        }


def placement_confound_note() -> str:
    return (
        "SeizeIT2 behind-the-ear sensors were positioned with reference to each "
        "patient's suspected seizure-onset zone. Differences in detection "
        "between anatomical subgroups are therefore confounded with electrode "
        "placement and must not be read as a pure effect of onset location."
    )
