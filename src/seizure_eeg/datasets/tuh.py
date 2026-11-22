"""TUH EEG Seizure Corpus (TUSZ) adapter — SKELETON, NOT YET VALIDATED.

Status
------
This file defines the interface and the label-mapping *plan* for TUSZ. It
deliberately produces **no results**: TUSZ requires a signed data-use agreement
and is not present in this repository. Every method raises or returns empty
until a real corpus is mounted and the mapping below has been checked against
the shipped ``_DOCS`` for the exact version in use.

Two rules that must survive integration
---------------------------------------
1. **A TUSZ seizure-type code is not an anatomical onset label.** ``cpsz``
   says "complex partial", i.e. focal with impaired awareness. It says nothing
   about which lobe. ``onset_lobe`` therefore stays ``not_annotated`` unless a
   clinical localisation is supplied from the EEG report or a separate
   validated table.
2. **Channel-level annotations are not localisation either.** TUSZ term-based
   (``.csv_bi`` / ``.csv``) files mark which channels an event was seen on.
   That is an electrographic field, shaped by montage and propagation. Mapping
   "seen on T3-T5" to "temporal lobe onset" is exactly the inference this
   project forbids elsewhere, and it is forbidden here.

Seizure-code mapping plan
-------------------------
See :data:`TUSZ_TYPE_MAP`. Codes are the TUSZ v2 label set. The mapping targets
:data:`seizure_eeg.schema.SEIZURE_TYPES` so that subgroup analyses written for
SeizeIT2 run unchanged.

Open questions to settle against the corpus documentation before use:

* ``tcsz`` (tonic-clonic) is not split in TUSZ into "generalized onset" versus
  "focal to bilateral". Mapping it to ``generalized`` would silently merge it
  with true generalized onset, so it maps to ``unclassified`` and a note is
  recorded; promote it only if the version in hand distinguishes them.
* ``nesz`` marks non-epileptic events. These must be **excluded** from seizure
  counts, not treated as a seizure subtype, and must also be excluded from the
  interictal background used for false-alarm rates.
* TUSZ ships ``bckg`` explicitly; background must come from the term files, not
  be assumed to be "anything not marked".
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..schema import Recording, SeizureEvent
from .base import DatasetAdapter

#: TUSZ label code -> harmonised seizure type (see module docstring).
TUSZ_TYPE_MAP: dict[str, str] = {
    "fnsz": "focal_unspecified",             # focal non-specific
    "spsz": "focal_aware",                   # simple partial (ILAE 2017: focal aware)
    "cpsz": "focal_impaired_awareness",      # complex partial
    "gnsz": "generalized",                   # generalized non-specific
    "absz": "generalized",                   # absence
    "tnsz": "generalized",                   # tonic
    "cnsz": "generalized",                   # clonic
    "atsz": "generalized",                   # atonic
    "mysz": "generalized",                   # myoclonic
    "tcsz": "unclassified",                  # see docstring: onset not distinguished
    "seiz": "unclassified",                  # seizure, type unspecified
}

#: Codes that are NOT epileptic seizures and must be excluded from both the
#: seizure counts and the interictal background.
TUSZ_EXCLUDE_CODES = {"nesz"}

#: Code that marks annotated background.
TUSZ_BACKGROUND_CODE = "bckg"

#: Standard TUSZ montages. TUSZ distributes referential recordings plus montage
#: definition files; the bipolar derivations below are the ones the CHB-MIT
#: montages map onto, so the wearable configurations transfer directly.
TUSZ_MONTAGES: dict[str, list[str]] = {
    "ar_tcp_18_bipolar": [
        "FP1-F7", "F7-T3", "T3-T5", "T5-O1",
        "FP2-F8", "F8-T4", "T4-T6", "T6-O2",
        "A1-T3", "T3-C3", "C3-CZ", "CZ-C4", "C4-T4", "T4-A2",
        "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    ],
    # TUSZ uses the older 10-20 labels T3/T4/T5/T6 where CHB-MIT uses T7/T8/P7/P8.
    "glasses_4_contacts": ["FP1-F7", "FP2-F8"],
    "temporal_4_contacts": ["F7-T3", "F8-T4"],
    "posterior_temporal_4_contacts": ["T3-T5", "T4-T6"],
    "left_temporal_2_contacts": ["F7-T3"],
    "right_temporal_2_contacts": ["F8-T4"],
}

#: Old-style 10-20 label -> modern label, for comparing TUSZ with CHB-MIT.
TUSZ_ELECTRODE_ALIASES = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}


class TUSZAdapter(DatasetAdapter):
    """Skeleton adapter. Raises until a real TUSZ tree is provided."""

    name = "tusz"
    is_wearable = False

    #: Flipped to True only once the loader has been validated against a real
    #: corpus. Analyses should refuse to report TUSZ numbers while it is False.
    IMPLEMENTED = False

    def __init__(self, root: str | Path):
        super().__init__(root)

    def _require(self) -> None:
        raise NotImplementedError(
            "The TUSZ adapter is a skeleton. TUSZ requires a signed data-use "
            "agreement (https://isip.piconepress.com/projects/nedc/) and is not "
            "bundled here. Implement parsing of the term-based annotation files "
            "against the _DOCS of the exact corpus version, re-check "
            "TUSZ_TYPE_MAP, and set IMPLEMENTED = True. Until then no TUSZ "
            "result may be reported.")

    def recordings(self) -> list[Recording]:
        if not self.IMPLEMENTED:
            self._require()
        return []                                     # pragma: no cover

    def events(self, recording: Recording | None = None) -> list[SeizureEvent]:
        if not self.IMPLEMENTED:
            self._require()
        return []                                     # pragma: no cover

    def load_signals(self, recording: Recording, channels: list[str] | None = None):
        if not self.IMPLEMENTED:
            self._require()
        return np.empty((0, 0)), float("nan"), []     # pragma: no cover

    def montages(self) -> dict[str, list[str]]:
        return dict(TUSZ_MONTAGES)

    def label_availability(self) -> dict[str, str]:
        return {
            "seizure_type": "annotated",          # TUSZ codes, via TUSZ_TYPE_MAP
            "onset_lobe": "not_annotated",        # NOT derivable from type or channels
            "onset_hemisphere": "not_annotated",  # likewise
            "vigilance": "not_annotated",
            "channel_level_annotation": "annotated",
            "paired_full_scalp_eeg": "annotated",
        }


def mapping_plan_frame() -> pd.DataFrame:
    """The seizure-code mapping plan as a table, for the report."""
    rows = [{"tusz_code": c, "harmonised_type": t,
             "included_as_seizure": True} for c, t in TUSZ_TYPE_MAP.items()]
    rows += [{"tusz_code": c, "harmonised_type": "EXCLUDED (non-epileptic)",
              "included_as_seizure": False} for c in sorted(TUSZ_EXCLUDE_CODES)]
    rows += [{"tusz_code": TUSZ_BACKGROUND_CODE,
              "harmonised_type": "background", "included_as_seizure": False}]
    return pd.DataFrame(rows)
