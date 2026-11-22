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

python scripts/07_report.py                    # summary tables
python scripts/05_figures.py explore
python scripts/05_figures.py results
python scripts/08_build_notebook.py --execute  # results walkthrough notebook

echo "Done. See results/tables and results/figures."
