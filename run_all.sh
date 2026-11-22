#!/usr/bin/env bash
# End-to-end reproduction. Expect ~1 h wall clock: most of it is the ~43 GB
# download; feature extraction is ~15 min on 4 cores and the benchmark ~25 min.
set -euo pipefail
cd "$(dirname "$0")"

python scripts/01_inventory.py                 # annotations + dataset inventory
python scripts/02_download.py --workers 6      # EDFs listed in the summaries
python scripts/04_extract_and_prune.py --keep chb01   # features; frees EDFs

python scripts/06_experiments.py main          # electrode reduction + placement
python scripts/06_experiments.py lower         # adds the FT9/FT10 montage
python scripts/06_experiments.py raw           # ablation: no per-file scaling
python scripts/06_experiments.py models        # LDA vs gradient boosting
python scripts/06_experiments.py random        # placement control (~15 min)

python scripts/07_report.py                    # CHB-MIT tables + RESULTS.md
python scripts/05_figures.py explore
python scripts/05_figures.py results

# --- SeizeIT2: real behind-the-ear wearable, seizure type and onset location
# Metadata first (~5700 tiny files), then only the seizure-containing
# recordings (12.5 GB of the 126 GB corpus).
python scripts/10_seizeit2_fetch.py metadata
python scripts/11_seizeit2_inventory.py
python scripts/10_seizeit2_fetch.py eeg --with-seizures
python scripts/12_seizeit2_features.py --workers 4
python scripts/13_seizeit2_benchmark.py
python scripts/14_seizeit2_figures.py

# --- cross-dataset deliverables
python scripts/16_harmonised_events.py         # one event schema, all datasets
python scripts/15_research_report.py           # REPORT.md
python scripts/08_build_notebook.py --execute  # expanded notebook

echo "Done. See REPORT.md, results/tables and results/figures."
