#!/usr/bin/env bash
# Reproduce every table and figure in the paper.
set -euo pipefail

DATA="${1:-data/cybersecurity_dataset.csv}"
SEED="${2:-42}"
export IDS_DATA="$DATA" IDS_SEED="$SEED"

echo "== 1/8  binary detection, tuning, McNemar, alert volume =="
python src/ids_pipeline.py --data "$DATA" --out results --seed "$SEED"

echo "== 2/8  cumulative feature ablation and per-fold CV scores =="
python src/feature_ablation.py --data "$DATA" --seed "$SEED"

echo "== 3/8  multi-class and SHAP =="
python src/multiclass_shap.py --data "$DATA" --seed "$SEED"

echo "== 4/8  prevalence sensitivity (Table 6, Fig. 5) =="
python src/prevalence_sensitivity.py

echo "== 5/8  bootstrap intervals for Table 4, paired differences for Table 8, FFNN variants for Table 5 =="
python src/reviewer_response.py --data "$DATA" --seed "$SEED" --boot 2000

echo "== 6/8  imbalance strategies on a fixed split (Table 10) =="
python src/imbalance_baselines.py --data "$DATA" --seed "$SEED"

echo "== 7/8  repeated splits (Table 9) =="
python src/repeated_splits.py --data "$DATA"

echo "== 8/8  figures =="
python src/make_figures.py --data "$DATA" --seed "$SEED"

echo
echo "Done. Tables in results/tables, figures in results/figures."
