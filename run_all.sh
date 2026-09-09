#!/usr/bin/env bash
# Reproduce every table and figure in the paper.
set -euo pipefail

DATA="${1:-data/cybersecurity_dataset.csv}"
SEED="${2:-42}"
export IDS_DATA="$DATA" IDS_SEED="$SEED"

echo "== 1/4  binary detection, tuning, McNemar, alert volume =="
python src/ids_pipeline.py --data "$DATA" --out results --seed "$SEED"

echo "== 2/4  cumulative feature ablation and per-fold CV scores =="
python src/feature_ablation.py --data "$DATA" --seed "$SEED"

echo "== 3/4  multi-class and SHAP =="
python src/multiclass_shap.py --data "$DATA" --seed "$SEED"

echo "== 4/4  figures =="
python src/make_figures.py --data "$DATA" --seed "$SEED"

echo
echo "Done. Tables in results/tables, figures in results/figures."
