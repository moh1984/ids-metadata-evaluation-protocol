# Results index

Every number in the paper, and where it comes from.

| Manuscript table | File in `results/tables/` | Produced by |
|---|---|---|
| Table 1 — Machine learning models, search spaces and selected hyperparameter configurations | `table01_best_hyperparameters.csv` | `ids_pipeline.py` |
| Table 2 — Raw features of the cybersecurity threat detection dataset | `(described in the manuscript)` | `—` |
| Table 3 — Distribution of attack categories in the dataset | `(described in the manuscript)` | `—` |
| Table 4 — Binary classification results at the source-data prevalence of 4% (2,880 benign vs. 120 attack) | `table04_binary.csv + table12_bootstrap_intervals.csv` | `ids_pipeline.py, reviewer_response.py` |
| Table 5 — Feed-forward network under three imbalance treatments | `table13_ffnn_balanced.csv` | `reviewer_response.py` |
| Table 6 — Attack precision and alert volume at three class priors | `table11_prevalence_sensitivity.csv` | `prevalence_sensitivity.py` |
| Table 7 — Five-Fold cross-Validation on the training partition | `table06_crossval_prauc.csv` | `ids_pipeline.py` |
| Table 8 — Paired bootstrap differences in PR-AUC, 2,000 resamples | `table12b_paired_differences.csv` | `reviewer_response.py` |
| Table 9 — Stability of the ordering across five stratified splits | `table15_repeated_splits.csv, table15b_per_seed.csv` | `repeated_splits.py` |
| Table 10 — Imbalance strategies compared on the same data, split and configurations | `table14_imbalance_strategies.csv` | `imbalance_baselines.py` |
| Table 11 — Cumulative contribution of each engineered feature group (Gradient boosting) | `table07_feature_ablation.csv` | `feature_ablation.py` |
| Table 12 — Multi-class tuning protocol and selected configurations | `mc_best_params.json` | `multiclass_shap.py` |
| Table 13 — Multi-Class classification performance | `table08_multiclass.csv` | `multiclass_shap.py` |
| Table 14 — Per-Class performance of XGBoost multi-Class classifier | `table09_perclass.csv` | `multiclass_shap.py` |
| Table 15 — Top-10 features by mean absolute SHAP value (XGBoost binary classifier) | `table10_shap.csv` | `multiclass_shap.py` |
| Figure 5 | `results/figures/fig10_prevalence_sensitivity.png` | `prevalence_sensitivity.py` |
| Figures 1–4 and 6–10 | `results/figures/fig0*.png` | `make_figures.py` |
| Section 5.4 — McNemar tests | `table_mcnemar_significance.csv` | `ids_pipeline.py` |
| Figure 7 — confusion matrix | `confusion_matrix_multiclass.csv` | `multiclass_shap.py` |
| Figure 8 — feature importance | `feature_importances.csv` | `feature_ablation.py` |
| Figure 6 — per-fold CV scores | `cv_fold_scores.json` | `feature_ablation.py` |

`table05_alert_volume.csv` is a legacy output: the single-prior alert volume it reports was superseded by Table 6, which gives the same quantity at three priors.

Table and figure numbering follows the manuscript, which carries fifteen tables and ten figures: nine produced by `make_figures.py` and Fig. 5 by `prevalence_sensitivity.py`. Six earlier
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

Seed 42 fixes the stratified split, the CV folds and every estimator, except in
`repeated_splits.py`, which deliberately varies the split across seeds 42, 1, 7,
13 and 2024, and `prevalence_sensitivity.py`, which is deterministic.
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

### Provenance of the bootstrap outputs

`table12_bootstrap_intervals.csv` and `table12b_paired_differences.csv` were produced on Google Colab with XGBoost installed, which is required for the gradient-boosting rows. `table12b` was written from the paired differences that the same run reported, in the layout the current script emits. Re-running
`python src/reviewer_response.py --boot 2000` regenerates both, and also adds the sample-weighted
feed-forward variant that the manuscript reports in Table 5.
