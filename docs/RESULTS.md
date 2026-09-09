# Results index

Every number in the paper, and where it comes from.

| Paper item | File | Produced by |
|---|---|---|
| Table 1 — hyperparameters | `results/tables/table01_best_hyperparameters.csv` | `ids_pipeline.py` |
| Table 4 — binary results | `results/tables/table04_binary.csv` | `ids_pipeline.py` |
| Table 5 — alert volume | `results/tables/table05_alert_volume.csv` | `ids_pipeline.py` |
| Table 6 — cross-validation | `results/tables/table06_crossval_prauc.csv` | `ids_pipeline.py` |
| Table 7 — cumulative feature ablation | `results/tables/table07_feature_ablation.csv` | `feature_ablation.py` |
| Isolated-group ablation (supplementary) | `results/tables/table07b_ablation_isolated.csv` | `feature_ablation.py` |
| Table 8 — multi-class | `results/tables/table08_multiclass.csv` | `multiclass_shap.py` |
| Table 9 — per-class | `results/tables/table09_perclass.csv` | `multiclass_shap.py` |
| Table 10 — SHAP | `results/tables/table10_shap.csv` | `multiclass_shap.py` |
| McNemar tests (§5.3) | `results/tables/table_mcnemar_significance.csv` | `ids_pipeline.py` |
| Per-fold CV scores | `results/tables/cv_fold_scores.json` | `feature_ablation.py` |
| Confusion matrix | `results/tables/confusion_matrix_multiclass.csv` | `multiclass_shap.py` |
| Figure 1 | `results/figures/fig01_methodology_pipeline.png` | `make_figures.py` |
| Figures 2–9 | `results/figures/fig0*.png` | `make_figures.py` |
| Raw SHAP arrays | `results/shap_values.npy`, `results/shap_X.npy` | `multiclass_shap.py` |

Figure numbering follows the manuscript, which carries nine figures. Six earlier
figures that only restated a table were removed during revision, and the rest were
renumbered. Every figure is drawn at its final print size (6.5 in full width, 3.1 in
one column) so that all lettering lands at 10 pt on the page; re-running
`make_figures.py` reproduces them, though tight bounding boxes may differ by a few
pixels of whitespace from the copies embedded in the manuscript.

Every feature subset in the ablation preserves the canonical `FEATURE_ORDER`
sequence. This matters: XGBoost histogram split search breaks ties differently
when identical features arrive in a different column order, which shifts results
in the third or fourth decimal. With canonical ordering the final ablation row is
numerically identical to the full-feature model in Table 4.

## Reproducibility

Seed 42 throughout: the stratified split, the CV folds and every estimator.
Re-running `run_all.sh` on the committed dataset reproduces these files exactly.

Dependency versions are pinned in `requirements.txt` to those below, which produced
the committed results. Loosening the pins produces small drift, most visibly in
XGBoost (histogram binning): an independent re-run on a different XGBoost build gave
TP = 79 / FP = 35 / PR-AUC 0.6648 against the committed TP = 80 / FP = 37 /
PR-AUC 0.6700. Install from the pinned file to reproduce exactly.

```
scikit-learn 1.8.0 · xgboost 3.4.0 · shap 0.52.0
numpy 2.4.4 · pandas 3.0.2 · scipy 1.17.1 · statsmodels 0.14.6
```
