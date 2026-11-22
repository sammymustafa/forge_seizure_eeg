"""Unit tests for the parsing, feature and scoring logic.

Run with ``python -m pytest tests -q`` (no downloaded data required).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

from seizure_eeg.config import BANDS, FEATURES_PER_CHANNEL, MONTAGES, patient_id
from seizure_eeg.dataset import canonical_channel_map, referential_map
from seizure_eeg.evaluation import PRE_TOL, EventScorer
from seizure_eeg.features import bandpass, window_features, window_quality
from seizure_eeg.summary import parse_summary

SUMMARY = """Data Sampling Rate: 256 Hz

Channels in EDF Files:
**********************
Channel 1: FP1-F7
Channel 2: F7-T7

File Name: chb99_01.edf
File Start Time: 11:42:54
Number of Seizures in File: 0

File Name: chb99_02.edf
Number of Seizures in File: 2
Seizure 1 Start Time: 100 seconds
Seizure 1 End Time: 140 seconds
Seizure 2 Start Time: 300 seconds
Seizure 2 End Time: 330 seconds
"""


# ----------------------------------------------------------------- summaries
def test_parse_summary_keeps_seizure_free_files(tmp_path):
    p = tmp_path / "s.txt"
    p.write_text(SUMMARY)
    anns = parse_summary(p)
    assert anns["chb99_01.edf"] == []
    assert anns["chb99_02.edf"] == [(100, 140), (300, 330)]


def test_parse_summary_rejects_declared_count_mismatch(tmp_path):
    p = tmp_path / "s.txt"
    p.write_text(SUMMARY.replace("Number of Seizures in File: 2",
                                 "Number of Seizures in File: 3"))
    with pytest.raises(ValueError, match="declares"):
        parse_summary(p)


def test_parse_summary_rejects_non_positive_duration(tmp_path):
    p = tmp_path / "s.txt"
    p.write_text(SUMMARY.replace("Seizure 1 End Time: 140", "Seizure 1 End Time: 90"))
    with pytest.raises(ValueError, match="non-positive"):
        parse_summary(p)


# ----------------------------------------------------------------- channels
def test_canonical_map_strips_duplicate_suffix_and_placeholders():
    m = canonical_channel_map(["FP1-F7", "T8-P8-0", "--0", "-", "T8-P8-1"])
    assert m["FP1-F7"] == "FP1-F7"
    assert m["T8-P8"] == "T8-P8-0"       # first occurrence wins
    assert "-" not in m and "" not in m


def test_referential_map_detects_common_reference():
    m = referential_map(["FP1-CS2", "F7-CS2", "--0", "T7-CS2"])
    assert m == {"FP1": "FP1-CS2", "F7": "F7-CS2", "T7": "T7-CS2"}
    assert referential_map(["FP1-F7", "F7-T7"]) == {}


def test_referential_map_handles_bare_labels_and_known_typos():
    m = referential_map(["F7", "T7", "01", "--0", "EKG1-CHIN", "C2"])
    assert m["F7"] == "F7" and m["T7"] == "T7"
    assert m["O1"] == "01"            # CHB-MIT header typo for O1
    assert "EKG1-CHIN" not in m and "C2" not in m


def test_patient_grouping_merges_chb01_and_chb21():
    assert patient_id("chb01") == patient_id("chb21")
    assert patient_id("chb05") == "chb05"


def test_every_montage_is_a_subset_of_the_full_contact_space():
    for name, chans in MONTAGES.items():
        assert chans, name
        for ch in chans:
            assert ch.count("-") == 1, (name, ch)


# ----------------------------------------------------------------- features
def test_feature_vector_length_and_leading_dims():
    rng = np.random.default_rng(0)
    x = bandpass(rng.normal(0, 50e-6, (3, 5, 2560)), 256.0)
    f = window_features(x, 256.0)
    assert f.shape == (3, 5, len(FEATURES_PER_CHANNEL))
    assert np.isfinite(f).all()


def test_band_power_tracks_a_known_sinusoid():
    sf, n = 256.0, 2560
    t = np.arange(n) / sf
    # 6 Hz (theta) at 10x the amplitude of 20 Hz (beta)
    sig = (100e-6 * np.sin(2 * np.pi * 6 * t) + 10e-6 * np.sin(2 * np.pi * 20 * t))
    f = window_features(bandpass(sig[None, :], sf), sf)[0]
    names = list(FEATURES_PER_CHANNEL)
    theta = f[names.index("logpow_theta")]
    beta = f[names.index("logpow_beta")]
    assert theta > beta
    # power ratio of 100 => 2 decades in log10
    assert 1.5 < (theta - beta) < 2.5
    assert f[names.index("relpow_theta")] > f[names.index("relpow_delta")]


def test_spectral_entropy_orders_noise_above_a_pure_tone():
    sf, n = 256.0, 2560
    t = np.arange(n) / sf
    rng = np.random.default_rng(1)
    tone = window_features(bandpass(
        (50e-6 * np.sin(2 * np.pi * 10 * t))[None, :], sf), sf)[0]
    noise = window_features(bandpass(
        rng.normal(0, 50e-6, n)[None, :], sf), sf)[0]
    i = FEATURES_PER_CHANNEL.index("spectral_entropy")
    assert noise[i] > tone[i]


def test_window_quality_flags_flatlines_and_nans():
    x = np.zeros((3, 100))
    x[0] = np.random.default_rng(0).normal(0, 1, 100)
    x[2, 5] = np.nan
    q = window_quality(x)
    assert q.tolist() == [True, False, False]


# ----------------------------------------------------------------- scoring
def _meta(n_windows=360, fname="a.edf"):
    st = np.arange(n_windows) * 10.0
    return pd.DataFrame({"edf_file": fname, "start_s": st, "stop_s": st + 10})


def _seiz(onset=300.0, offset=340.0, fname="a.edf"):
    return pd.DataFrame({"edf_file": [fname], "onset_s": [onset],
                         "offset_s": [offset]})


def test_alarm_requires_consecutive_windows():
    meta, sz = _meta(), _seiz()
    s = np.zeros(len(meta))
    s[30] = 1.0                       # a single window at 300-310 s
    strict = EventScorer(meta, sz, min_consecutive=2).score(s, 0.5)
    loose = EventScorer(meta, sz, min_consecutive=1).score(s, 0.5)
    assert strict["n_detected"] == 0
    assert loose["n_detected"] == 1


def test_latency_is_measured_from_the_end_of_the_confirming_window():
    meta, sz = _meta(), _seiz()
    s = np.zeros(len(meta))
    s[30:34] = 1.0                    # 300-340 s, confirmed at 320 s
    r = EventScorer(meta, sz, min_consecutive=2).score(s, 0.5)
    assert r["median_latency_s"] == 20.0
    assert r["n_false_alarms"] == 0


def test_an_always_on_detector_is_penalised_not_rewarded():
    meta, sz = _meta(), _seiz()
    r = EventScorer(meta, sz, min_consecutive=2).score(np.ones(len(meta)), 0.5)
    assert r["event_sensitivity"] == 1.0
    assert r["false_alarms_per_hour"] > 50      # re-issued every 60 s
    assert r["median_latency_s"] == -PRE_TOL    # credit is clipped
    assert r["alarm_duty_cycle"] == 1.0


def test_pre_onset_alarm_counts_as_detection_within_tolerance():
    meta, sz = _meta(), _seiz()
    s = np.zeros(len(meta))
    s[27:29] = 1.0                    # alarm confirmed at 290 s, onset at 300 s
    r = EventScorer(meta, sz, min_consecutive=2).score(s, 0.5)
    assert r["n_detected"] == 1
    assert r["median_latency_s"] == -10.0


def test_false_alarms_are_counted_per_alarm_event():
    meta, sz = _meta(), _seiz()
    s = np.zeros(len(meta))
    s[50:52] = 1.0
    s[100:102] = 1.0
    r = EventScorer(meta, sz, min_consecutive=2).score(s, 0.5)
    assert r["n_false_alarms"] == 2
    assert r["n_detected"] == 0


def test_threshold_search_respects_the_false_alarm_budget():
    meta, sz = _meta(3600), _seiz()
    rng = np.random.default_rng(0)
    s = rng.random(len(meta))
    sc = EventScorer(meta, sz, min_consecutive=2)
    for target in (1.0, 6.0, 24.0):
        thr = sc.threshold_for_target_fa(s, target)
        assert sc.score(s, thr)["false_alarms_per_hour"] <= target + 1e-9


def test_false_alarm_rate_falls_with_threshold_overall():
    """FA/h trends down with the threshold, but is not strictly monotone."""
    meta, sz = _meta(3600), _seiz()
    rng = np.random.default_rng(2)
    s = rng.random(len(meta))
    sc = EventScorer(meta, sz, min_consecutive=2)
    fa = np.array([sc.score(s, t)["false_alarms_per_hour"]
                   for t in np.linspace(0.1, 0.99, 12)])
    assert fa[0] > fa[-1]
    assert np.corrcoef(np.arange(len(fa)), fa)[0, 1] < -0.8


def test_raising_the_threshold_can_split_one_alarm_into_two():
    """The reason FA/h is not strictly monotone - documented, not accidental."""
    meta, sz = _meta(), _seiz(onset=3000.0, offset=3010.0)
    s = np.zeros(len(meta))
    s[10:15] = 0.9                    # one 50 s run, under the 60 s re-issue
    s[12] = 0.6                       # a dip inside it
    sc = EventScorer(meta, sz, min_consecutive=2)
    assert sc.score(s, 0.5)["n_false_alarms"] == 1
    assert sc.score(s, 0.7)["n_false_alarms"] == 2


def test_threshold_search_is_robust_to_that_non_monotonicity():
    meta, sz = _meta(3600), _seiz()
    rng = np.random.default_rng(3)
    s = rng.random(len(meta))
    sc = EventScorer(meta, sz, min_consecutive=2)
    for target in (0.5, 2.0, 10.0):
        thr = sc.threshold_for_target_fa(s, target)
        assert sc.score(s, thr)["false_alarms_per_hour"] <= target + 1e-9
