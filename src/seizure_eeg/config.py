"""Shared configuration: montages, feature bands, windowing, paths."""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------- paths
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data" / "chbmit"
RESULTS_ROOT = REPO_ROOT / "results"
TABLES_DIR = RESULTS_ROOT / "tables"
FIGURES_DIR = RESULTS_ROOT / "figures"
FEATURES_DIR = RESULTS_ROOT / "features"

PHYSIONET_BASE = "https://physionet.org/files/chbmit/1.0.0/"

# ---------------------------------------------------------------- windowing
WINDOW_S = 10.0          # non-overlapping analysis window
STEP_S = 10.0            # hop; equal to WINDOW_S => non-overlapping
BANDPASS = (0.5, 40.0)   # Hz
LINE_FREQ = 60.0         # CHB-MIT recorded in the US

BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "low_gamma": (30, 40),
}

# Per-channel feature names, in the order produced by features.window_features
FEATURES_PER_CHANNEL = (
    [f"logpow_{b}" for b in BANDS]
    + [f"relpow_{b}" for b in BANDS]
    + ["log_rms", "log_line_length", "spectral_entropy", "log_bandpow_ratio_hi_lo"]
)

# ---------------------------------------------------------------- montages
# NOTE: every entry below is a list of BIPOLAR DERIVATIONS (EDF channel names),
# not a list of electrodes. A "4-contact" wearable mock-up yields 2 bipolar
# channels; a "2-contact" one yields a single bipolar channel.
MONTAGES: dict[str, list[str]] = {
    "full_18_bipolar": [
        "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
        "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
        "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
        "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
        "FZ-CZ", "CZ-PZ",
    ],
    "glasses_4_contacts": ["FP1-F7", "FP2-F8"],
    "temporal_4_contacts": ["F7-T7", "F8-T8"],
    "posterior_temporal_4_contacts": ["T7-P7", "T8-P8"],
    "lower_temporal_4_contacts": ["T7-FT9", "FT10-T8"],
    "left_temporal_2_contacts": ["F7-T7"],
    "right_temporal_2_contacts": ["F8-T8"],
}

# Electrode contacts implied by each montage (what the hardware would need).
MONTAGE_CONTACTS: dict[str, list[str]] = {
    name: sorted({e for ch in chans for e in ch.split("-")})
    for name, chans in MONTAGES.items()
}

# Human-readable grouping used in figures/tables.
MONTAGE_ORDER = [
    "full_18_bipolar",
    "glasses_4_contacts",
    "temporal_4_contacts",
    "posterior_temporal_4_contacts",
    "lower_temporal_4_contacts",
    "left_temporal_2_contacts",
    "right_temporal_2_contacts",
]

MONTAGE_LABELS = {
    "full_18_bipolar": "Full scalp (18 bipolar ch)",
    "glasses_4_contacts": "Glasses-like (4 contacts, 2 ch)",
    "temporal_4_contacts": "Temporal (4 contacts, 2 ch)",
    "posterior_temporal_4_contacts": "Posterior-temporal (4 contacts, 2 ch)",
    "lower_temporal_4_contacts": "Lower-temporal (4 contacts, 2 ch)",
    "left_temporal_2_contacts": "Left temporal (2 contacts, 1 ch)",
    "right_temporal_2_contacts": "Right temporal (2 contacts, 1 ch)",
}

# Same physical person recorded twice in CHB-MIT: chb01 and chb21.
# Keep them in one CV group so patient-disjoint folds really are disjoint.
PATIENT_GROUPS = {"chb01": "chb01_chb21", "chb21": "chb01_chb21"}


def patient_id(case_id: str) -> str:
    """Map a CHB-MIT case folder name to the underlying person."""
    return PATIENT_GROUPS.get(case_id, case_id)
