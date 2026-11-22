"""Montage-comparison experiments on CHB-MIT.

Sub-commands
------------
main      Experiments 1-2+4: electrode reduction and placement, patient-disjoint
          5-fold CV, paired windows, thresholds set on training patients only.
raw       Same but without per-recording feature normalisation (ablation).
lower     Adds the FT9/FT10 lower-temporal montage on the subjects that have it.
random    Experiment 3: every valid 2-channel bipolar subset as a placement
          control, to put the named placements on a percentile.
models    Shrinkage LDA vs gradient boosting on the full and temporal montages.
"""
from __future__ import annotations

import argparse
import sys
import time
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from seizure_eeg.benchmark import (load_store, montage_design, normalise_per_file,
                                   paired_mask, run_montage_cv)
from seizure_eeg.config import (FEATURES_PER_CHANNEL, MONTAGES, MONTAGE_ORDER,
                                TABLES_DIR)
from seizure_eeg.dataset import CH_INDEX

PRIMARY = [m for m in MONTAGE_ORDER if m != "lower_temporal_4_contacts"]
TARGETS = (1.0, 6.0, 24.0)


def _describe(store, mask, label):
    m = store.meta.loc[mask]
    print(f"[{label}] windows={mask.sum()} ({mask.sum()*10/3600:.1f} h) "
          f"ictal={int(m.label.sum())} patients={m.patient_id.nunique()} "
          f"cases={m.case_id.nunique()}", flush=True)


def run_set(store, montages, mask, *, normalise, model="lda", tag=""):
    folds, patients = [], []
    for mont in montages:
        t0 = time.time()
        f, p = run_montage_cv(store, mont, mask, model_kind=model,
                              normalise=normalise, target_fa_per_h=TARGETS,
                              min_consecutive=2)
        folds.append(f)
        patients.append(p)
        best = f[f.target_fa_per_h == 6.0]
        print(f"  {mont:32s} n_feat={f.n_features.iloc[0]:3d} "
              f"AUPRC={f.auprc.mean():.3f} "
              f"sens@6FA/h={best.event_sensitivity.mean():.3f} "
              f"FA/h={best.false_alarms_per_hour.mean():.2f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    fd = pd.concat(folds, ignore_index=True)
    pd_ = pd.concat(patients, ignore_index=True)
    if tag:
        fd.to_csv(TABLES_DIR / f"cv_folds_{tag}.csv", index=False)
        pd_.to_csv(TABLES_DIR / f"cv_patients_{tag}.csv", index=False)
    return fd, pd_


def cmd_main(store):
    mask = paired_mask(store, PRIMARY)
    _describe(store, mask, "main")
    run_set(store, PRIMARY, mask, normalise="perfile", tag="main")


def cmd_raw(store):
    mask = paired_mask(store, PRIMARY)
    _describe(store, mask, "raw-features")
    run_set(store, PRIMARY, mask, normalise="none", tag="rawfeat")


def cmd_lower(store):
    allm = MONTAGE_ORDER
    _, usable, _ = montage_design(store, "lower_temporal_4_contacts")
    cases_ok = sorted(store.meta.loc[usable, "case_id"].unique())
    sel = store.meta.case_id.isin(cases_ok).to_numpy()
    mask = paired_mask(store, allm) & sel
    print(f"[lower] cases with FT9/FT10: {len(cases_ok)}", flush=True)
    _describe(store, mask, "lower")
    run_set(store, allm, mask, normalise="perfile", tag="lower")


def cmd_random(store):
    """Every 2-channel subset of the 18 standard derivations as a control."""
    mask = paired_mask(store, PRIMARY)
    _describe(store, mask, "random-control")
    chans = MONTAGES["full_18_bipolar"]
    meta = store.meta.loc[mask]
    rows = []
    pairs = list(combinations(chans, 2))
    print(f"evaluating {len(pairs)} channel pairs", flush=True)
    for n, pair in enumerate(pairs, 1):
        cols = [CH_INDEX[c] for c in pair]
        sub = store.X[mask][:, cols, :].reshape(mask.sum(), -1).astype(np.float64)
        sub = normalise_per_file(sub, meta)
        f, _ = run_montage_cv(store, "pair", mask, feature_matrix=sub,
                              normalise="perfile", target_fa_per_h=(6.0,),
                              min_consecutive=2, per_patient=False)
        rows.append({
            "channels": "+".join(pair),
            "contacts": "+".join(sorted({e for c in pair for e in c.split("-")})),
            "auprc": f.auprc.mean(), "auroc": f.auroc.mean(),
            "event_sensitivity": f.event_sensitivity.mean(),
            "false_alarms_per_hour": f.false_alarms_per_hour.mean(),
        })
        if n % 20 == 0:
            print(f"  {n}/{len(pairs)}", flush=True)
    df = pd.DataFrame(rows).sort_values("auprc", ascending=False)
    df.to_csv(TABLES_DIR / "random_pair_control.csv", index=False)
    print(df.head(12).to_string(index=False), flush=True)


def cmd_models(store):
    mask = paired_mask(store, PRIMARY)
    _describe(store, mask, "models")
    out = []
    for model in ("lda", "hgb"):
        f, _ = run_set(store, ["full_18_bipolar", "temporal_4_contacts"], mask,
                       normalise="perfile", model=model)
        out.append(f)
    pd.concat(out, ignore_index=True).to_csv(
        TABLES_DIR / "cv_folds_models.csv", index=False)


COMMANDS = {"main": cmd_main, "raw": cmd_raw, "lower": cmd_lower,
            "random": cmd_random, "models": cmd_models}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=sorted(COMMANDS))
    ap.add_argument("--cases", default="")
    args = ap.parse_args()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    cases = args.cases.split(",") if args.cases else None
    t0 = time.time()
    store = load_store(cases)
    print(f"loaded {len(store)} windows ({len(store)*10/3600:.1f} h), "
          f"{store.meta.patient_id.nunique()} patients, "
          f"{len(store.seizures)} annotated seizures ({time.time()-t0:.0f}s)",
          flush=True)
    COMMANDS[args.command](store)


if __name__ == "__main__":
    main()
