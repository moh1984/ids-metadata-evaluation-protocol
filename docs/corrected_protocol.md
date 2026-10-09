# Corrected experimental protocol — how to run it

## What this fixes

| # | Defect (Section 6 of the manuscript) | What the script does instead |
|---|---|---|
| 1 | Balancing applied **before** the split → test set was 50% attack | Stratified split runs **first** on all 10,000 records; the test partition keeps the 24:1 source-data ratio. Balancing is applied to the **training** partition only. |
| 2 | Hyperparameters fixed a priori | `GridSearchCV` on the training partition only. The binary search is scored by **average precision (PR-AUC)**, the correct criterion under heavy imbalance; the multi-class search is run independently and scored by **macro F1**. |
| 3 | `StandardScaler` fitted on the full dataset; CV run over data containing the test set | Scaler sits **inside** a `Pipeline`, so it is re-fitted on every CV training fold. The test partition is never touched during tuning. |

Also added: false-positive rate, ROC-AUC, PR-AUC, balanced accuracy, an operational
a prevalence-sensitivity and alert-volume analysis, and paired McNemar tests computed from **real** predictions
(no more "best-case p ≥ 0.25" bounds).

## Install

```bash
pip install scikit-learn pandas numpy scipy xgboost shap statsmodels
```

The full reproduction pipeline requires every dependency pinned in
`requirements.txt`: `feature_ablation.py` imports XGBoost directly and
`multiclass_shap.py` needs both XGBoost and SHAP. `ids_pipeline.py` alone will
fall back to scikit-learn gradient boosting if XGBoost is missing, but the
manuscript must then say "Gradient Boosting (scikit-learn)", not "XGBoost".

## Run

```bash
# quick check first (a few minutes; runtime depends on hardware):
# small grids, binary stage only
python src/ids_pipeline.py --data data/cybersecurity_dataset.csv --out results --seed 42 --fast

# full reproduction of every table and figure (about an hour on two CPU cores)
bash run_all.sh data/cybersecurity_dataset.csv 42
```

If your column names differ from the manuscript's Table 2, edit `COLUMN_MAP`
at the top of the script:

```python
COLUMN_MAP = {"source_port": "src_port", "ts": "timestamp"}
```

The script exits with a clear message listing any missing columns.

## Output → manuscript mapping

| File | Paper item | Produced by |
|---|---|---|
| `table01_best_hyperparameters.csv` | Table 1 | `ids_pipeline.py` |
| `table04_binary.csv` | Table 4 | `ids_pipeline.py` |
| `table05_alert_volume.csv` | legacy: superseded by Table 6 | `ids_pipeline.py` |
| `table06_crossval_prauc.csv` | Table 7 | `ids_pipeline.py` |
| `table07_feature_ablation.csv` | Table 11 (cumulative ablation) | `feature_ablation.py` |
| `table07b_ablation_isolated.csv` | supplementary (isolated groups) | `feature_ablation.py` |
| `table08_multiclass.csv` | Table 13 | `multiclass_shap.py` |
| `table09_perclass.csv` | Table 14 | `multiclass_shap.py` |
| `table10_shap.csv` | Table 15 | `multiclass_shap.py` |
| `table_mcnemar_significance.csv` | Section 5.4 | `ids_pipeline.py` |
| `cv_fold_scores.json` | Figure 6 | `feature_ablation.py` |
| `confusion_matrix_multiclass.csv` | Figure 7 | `multiclass_shap.py` |
| `feature_importances.csv` | Figure 8 | `feature_ablation.py` |

This table mirrors `docs/RESULTS.md`, which is the authoritative index.

## Expect the headline numbers to fall

This is the point of the exercise, not a problem. At the 4% source-data prevalence:

- **Accuracy will rise** (toward ~96%) and becomes almost meaningless — a
  classifier predicting "benign" always scores 96%. Lead with **PR-AUC** and
  **attack-class recall at a stated FPR** instead.
- **Attack precision will drop sharply.** This is the honest number, and the
  one a SOC cares about.
- **Rare-class support does not improve.** Although the total test partition
  grows from 240 to 3,000 flows, the number of attack samples stays at 120 under
  the 4% source-data prevalence, so support for the rare attack categories remains
  extremely limited: C2 still has only 3 test instances. Consider collapsing the
  rarest categories into an "other" class for the multi-class task, and say so
  explicitly.

Report the new figures as the study's results, and keep the balanced-test
numbers only if you want a "for comparison with prior protocol" row.

## Reproducibility

`--seed` controls the split, the CV folds and every model. Record the value you
used in the paper, and add a Data/Code Availability statement pointing at
this script.
