#!/usr/bin/env bash
# Reproduce the entire analysis from the raw workbook.
# Scripts are order-dependent: 03 consumes the completed dataset from 02.
set -euo pipefail
cd "$(dirname "$0")"

echo "== Task 1: missingness diagnostics =="
python src/01_diagnostics.py

echo
echo "== Task 2: imputation =="
python src/02_impute.py

echo
echo "== Task 3: validation (this is the slow one, ~12 min) =="
python src/03_validate.py

echo
echo "Done. Deliverables in outputs/ ; report in WRITEUP.md"
