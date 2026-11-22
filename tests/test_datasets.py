"""Tests for the harmonised schema, adapters and subgroup statistics."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

from seizure_eeg.datasets.chbmit import (EXTERNAL_ANNOTATION_COLUMNS,
                                         load_external_annotations)
from seizure_eeg.datasets.seizeit2 import (SeizeIT2Adapter,
                                           harmonise_localization,
                                           harmonise_seizure_type)
from seizure_eeg.datasets.tuh import (TUSZ_EXCLUDE_CODES, TUSZ_TYPE_MAP,
                                      TUSZAdapter, mapping_plan_frame)
from seizure_eeg.schema import (ONSET_LOBES, SEIZURE_TYPES, SeizureEvent,
                                events_to_frame, validate_events)
from seizure_eeg.subgroups import paired_subgroup_delta, subgroup_sensitivity


# ------------------------------------------------------- label harmonisation
@pytest.mark.parametrize("raw,expected", [
    ("sz_foc_a", "focal_aware"),
    ("sz_foc_a_m_automatisms", "focal_aware"),
    ("sz_foc_ia", "focal_impaired_awareness"),
    ("sz_foc_ia_nm_behavior", "focal_impaired_awareness"),
    ("sz_foc_ua_m_tonic", "focal_unknown_awareness"),
    ("sz_foc_f2b", "focal_to_bilateral_tonic_clonic"),
    ("sz_foc", "focal_unspecified"),
    ("sz_gen_m_tonicClonic", "generalized"),
    ("sz_uo_nm", "unknown_onset"),
    ("sz", "unclassified"),
])
def test_seizure_type_harmonisation(raw, expected):
    assert harmonise_seizure_type(raw) == expected
    assert expected in SEIZURE_TYPES


def test_focal_aware_and_impaired_awareness_are_not_confused():
    """`sz_foc_a` is a string prefix of `sz_foc_a...` but NOT of `sz_foc_ia`."""
    assert harmonise_seizure_type("sz_foc_ia") != harmonise_seizure_type("sz_foc_a")
    assert harmonise_seizure_type("sz_foc_ia_m") == "focal_impaired_awareness"


def test_multilobar_onsets_are_not_collapsed_to_one_lobe():
    lobe, lobes = harmonise_localization("front_cen_temp")
    assert lobe == "multilobar"
    assert lobes == ["frontal", "central", "temporal"]
    assert harmonise_localization("temp") == ("temporal", ["temporal"])
    assert harmonise_localization("un") == ("unknown", [])
    assert harmonise_localization("n/a") == ("unknown", [])
    for code in ["temp", "front", "cen", "par", "occ", "ins", "un"]:
        assert harmonise_localization(code)[0] in ONSET_LOBES


# ------------------------------------------------------------------- schema
def _event(**kw):
    base = dict(dataset="d", patient_id="p1", session_id="s1",
                recording_id="r1", event_index=0, onset_s=10.0, offset_s=50.0)
    base.update(kw)
    return SeizureEvent(**base)


def test_event_frame_round_trip_and_duration():
    df = events_to_frame([_event(), _event(event_index=1, onset_s=100.0,
                                           offset_s=130.0)])
    validate_events(df)
    assert list(df.duration_s) == [40.0, 30.0]


def test_validate_rejects_unknown_category():
    df = events_to_frame([_event(seizure_type="not_a_type")])
    with pytest.raises(ValueError, match="seizure_type"):
        validate_events(df)


def test_validate_rejects_non_positive_duration():
    df = events_to_frame([_event(onset_s=50.0, offset_s=50.0)])
    with pytest.raises(ValueError, match="duration"):
        validate_events(df)


def test_unknown_and_not_annotated_are_distinguishable():
    a = _event(onset_lobe="unknown", onset_lobe_provenance="unknown")
    b = _event(onset_lobe="unknown", onset_lobe_provenance="not_annotated")
    df = events_to_frame([a, b])
    assert set(df.onset_lobe_provenance) == {"unknown", "not_annotated"}


# ------------------------------------------------------- external annotations
def test_external_annotations_require_provenance(tmp_path):
    p = tmp_path / "ext.csv"
    rows = {c: ["x"] for c in EXTERNAL_ANNOTATION_COLUMNS}
    rows["source_citation"] = [""]
    pd.DataFrame(rows).to_csv(p, index=False)
    with pytest.raises(ValueError, match="provenance"):
        load_external_annotations(p)


def test_external_annotations_accept_cited_rows(tmp_path):
    p = tmp_path / "ext.csv"
    rows = {c: ["x"] for c in EXTERNAL_ANNOTATION_COLUMNS}
    rows["source_citation"] = ["doi:10.0000/example"]
    rows["source_table"] = ["Table 2"]
    pd.DataFrame(rows).to_csv(p, index=False)
    assert len(load_external_annotations(p)) == 1


def test_missing_external_file_is_empty_not_an_error(tmp_path):
    assert load_external_annotations(tmp_path / "nope.csv").empty


# ----------------------------------------------------------------- SeizeIT2
def test_seizeit2_parses_events_tsv(tmp_path):
    eeg = tmp_path / "sub-001" / "ses-01" / "eeg"
    eeg.mkdir(parents=True)
    rec = "sub-001_ses-01_task-szMonitoring_run-01"
    (eeg / f"{rec}_events.tsv").write_text(
        "onset\tduration\teventType\tlateralization\tlocalization\tvigilance\t"
        "confidence\tchannels\tdateTime\trecordingDuration\n"
        "100.00\t20.00\timpd\tn/a\tn/a\tn/a\tn/a\tn/a\t2000-01-01\t3600.00\n"
        "500.00\t40.00\tsz_foc_ia_m_automatisms\tleft\ttemp\tasleep\t1.0\t"
        "n/a\t2000-01-01\t3600.00\n"
        "900.00\t60.00\tsz_foc_f2b\tbi\tfront_temp\tawake\tn/a\tn/a\t"
        "2000-01-01\t3600.00\n")
    (eeg / f"{rec}_eeg.json").write_text(
        '{"SamplingFrequency": 256, "EEGChannelCount": 2, '
        '"EEGMontage": "left", "EEGPlacementScheme": "behind-the-ear", '
        '"RecordingDuration": 3600.0, "RecordingType": "discontinuous"}')

    ad = SeizeIT2Adapter(tmp_path)
    recs = ad.recordings()
    assert len(recs) == 1 and recs[0].patient_id == "sub-001"
    assert recs[0].montage_name == "behind-the-ear left" and recs[0].is_wearable

    evs = ad.events()
    assert len(evs) == 2                       # the impedance check is not a seizure
    assert evs[0].seizure_type == "focal_impaired_awareness"
    assert evs[0].onset_lobe == "temporal" and evs[0].involves_temporal
    assert evs[0].onset_hemisphere == "left" and evs[0].vigilance == "asleep"
    assert evs[1].seizure_type == "focal_to_bilateral_tonic_clonic"
    assert evs[1].onset_lobe == "multilobar"
    assert evs[1].involves_temporal and evs[1].involves_frontal
    assert evs[1].onset_hemisphere == "bilateral"
    validate_events(events_to_frame(evs))


def test_seizeit2_label_availability_denies_paired_full_scalp():
    ad = SeizeIT2Adapter("nonexistent")
    assert ad.label_availability()["paired_full_scalp_eeg"] == "not_annotated"
    assert ad.is_wearable is True


# --------------------------------------------------------------------- TUSZ
def test_tusz_adapter_refuses_to_produce_results():
    ad = TUSZAdapter("nowhere")
    assert TUSZAdapter.IMPLEMENTED is False
    for call in (ad.recordings, ad.events):
        with pytest.raises(NotImplementedError, match="skeleton"):
            call()


def test_tusz_does_not_claim_anatomical_localisation():
    avail = TUSZAdapter("nowhere").label_availability()
    assert avail["onset_lobe"] == "not_annotated"
    assert avail["onset_hemisphere"] == "not_annotated"
    assert avail["channel_level_annotation"] == "annotated"


def test_tusz_type_map_targets_known_categories_and_excludes_non_epileptic():
    for code, label in TUSZ_TYPE_MAP.items():
        assert label in SEIZURE_TYPES, code
    assert "nesz" in TUSZ_EXCLUDE_CODES
    assert "nesz" not in TUSZ_TYPE_MAP
    plan = mapping_plan_frame()
    assert not plan.loc[plan.tusz_code == "nesz", "included_as_seizure"].iloc[0]
    # tonic-clonic must not be silently merged into generalized onset
    assert TUSZ_TYPE_MAP["tcsz"] == "unclassified"


# ----------------------------------------------------------------- subgroups
def _outcomes(n_per_patient=8, n_patients=12, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_patients):
        for k in range(n_per_patient):
            rows.append({"patient_id": f"p{p}",
                         "group": "a" if p % 2 == 0 else "b",
                         "detected": bool(rng.random() < (0.8 if p % 2 == 0 else 0.4)),
                         "latency_s": float(rng.normal(20, 4)),
                         "duration_s": 40.0})
    return pd.DataFrame(rows)


def test_subgroup_table_reports_counts_and_intervals():
    tab = subgroup_sensitivity(_outcomes(), "group", n_boot=400)
    assert set(tab.group) == {"a", "b"}
    for r in tab.itertuples():
        assert r.n_patients >= 1 and r.n_seizures >= 1
        assert r.sensitivity_lo <= r.sensitivity <= r.sensitivity_hi


def test_bootstrap_resamples_patients_not_events():
    """One patient's 200 seizures must not shrink the interval to nothing."""
    heavy = pd.DataFrame({"patient_id": ["p0"] * 200,
                          "group": ["a"] * 200,
                          "detected": [True] * 100 + [False] * 100,
                          "latency_s": [20.0] * 200, "duration_s": [40.0] * 200})
    others = pd.DataFrame({"patient_id": [f"p{i}" for i in range(1, 6)],
                           "group": ["a"] * 5, "detected": [True] * 5,
                           "latency_s": [20.0] * 5, "duration_s": [40.0] * 5})
    tab = subgroup_sensitivity(pd.concat([heavy, others]), "group", n_boot=800)
    r = tab.iloc[0]
    assert r.n_seizures == 205 and r.n_patients == 6
    # resampling patients keeps real uncertainty; resampling events would not
    assert r.sensitivity_hi - r.sensitivity_lo > 0.1


def test_paired_delta_detects_a_real_difference():
    d = paired_subgroup_delta(_outcomes(seed=1), "group", "a", "b", n_boot=800)
    assert d["delta"] > 0
    assert d["ci_lo"] < d["delta"] < d["ci_hi"]
    assert d["n_patients_a"] + d["n_patients_b"] == 12


def test_single_patient_subgroup_gets_no_interval():
    df = pd.DataFrame({"patient_id": ["p0"] * 4, "group": ["a"] * 4,
                       "detected": [True, False, True, True],
                       "latency_s": [20.0] * 4, "duration_s": [40.0] * 4})
    tab = subgroup_sensitivity(df, "group", n_boot=200)
    assert np.isnan(tab.iloc[0].sensitivity_lo)
