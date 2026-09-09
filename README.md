# ids-metadata-evaluation-protocol

**Evaluating ML intrusion detection at a realistic base rate.**

Code, data and results for the paper:

> **Evaluation Protocol Matters More Than Model Choice: Intrusion Detection from Raw Network Metadata at an Operational Base Rate**
> Musab Alzghoul¹, Mahmoud Baklizi², Mohammad Alkhazaleh¹*, Osama Qtaish³
> ¹ Isra University, Dept. of Computer Science · ² University of Petra, Dept. of Computer Science · ³ Isra University, Dept. of Software Engineering · * corresponding author

---

## What this study does

Five classifiers — logistic regression, SVM (RBF), random forest, gradient boosting (XGBoost) and a feed-forward neural network — are evaluated on 10,000 network flow records for both binary attack detection and ten-class attack attribution, using 26 features engineered from 13 raw metadata attributes, with SHAP attributions for interpretability.

**The evaluation protocol is the point.** Many published IDS pipelines balance the dataset *before* splitting, which produces a test partition whose class prior does not exist in deployment. This repository evaluates at the true 24:1 base rate and shows that the choice changes the conclusions, not just the decimals.

## Headline results

Held-out partition: 3,000 flows (2,880 benign, 120 attack), preserving the source 24:1 ratio.

### Binary detection

| Model | Accuracy % | Balanced Acc % | Attack Recall % | Attack Precision % | FPR % | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 79.40 | 79.69 | 80.00 | 13.91 | 20.63 | 0.310 | 0.863 |
| SVM (RBF) | 85.97 | 70.73 | 54.17 | 15.08 | 12.71 | 0.271 | 0.837 |
| Random Forest | 97.37 | 74.27 | 49.17 | **76.62** | **0.63** | 0.659 | **0.911** |
| **Gradient Boosting** | 97.43 | **82.69** | **66.67** | 68.38 | 1.29 | **0.670** | 0.909 |
| Neural Network | 96.03 | 56.41 | 13.33 | 51.61 | 0.52 | 0.339 | 0.859 |

McNemar exact tests on paired predictions: nine of ten pairwise comparisons significant at *p* < 0.001. The exception is **gradient boosting vs. random forest (p = 0.885)** — statistically indistinguishable in overall correctness, differing instead in operating point.

### Why the protocol matters

Under a *balanced* test partition (the earlier protocol) logistic regression reached 87.92% accuracy and looked competitive. At the true base rate its attack-class precision is **13.91%** — 594 false alerts against 96 true ones on the test partition.

### Feature ablation

Cumulative: each row is a strict superset of the row above, so the delta is attributable to that group alone.

| Feature set (cumulative) | n | CV PR-AUC | Test PR-AUC | Δ | Test ROC-AUC |
|---|---|---|---|---|---|
| Raw attributes only | 7 | 0.4040 ± 0.0749 | 0.4254 | — | 0.7641 |
| + traffic-volume derivations | 11 | 0.3780 ± 0.0864 | 0.4509 | +0.0256 | 0.7876 |
| + port-level indicators | 17 | 0.4706 ± 0.0600 | 0.5100 | +0.0591 | 0.7916 |
| + user-agent and URL features | 25 | 0.6662 ± 0.0380 | 0.6631 | **+0.1531** | 0.9102 |
| **+ temporal (full set)** | 26 | **0.6703 ± 0.0457** | **0.6700** | +0.0068 | 0.9087 |

Feature engineering raises held-out PR-AUC by 57.5% relative. User-agent and URL features alone account for almost two-thirds of that gain. The derived traffic-volume features contribute 0.0256 while *lowering* cross-validated PR-AUC — they are deterministic functions of byte counts already in the raw set — and we report that negative result rather than omit it.

Every subset preserves the canonical feature ordering, so the final row is numerically identical to the full-feature model in Table 4 above. `results/tables/table07b_ablation_isolated.csv` additionally reports each group added to the raw set in isolation.


### Multi-class attribution does not work at these sample sizes

Gradient boosting reaches 97.17% accuracy but only **33.41% macro F1**. A classifier predicting "benign" for every flow scores 96.00%. Command injection, exploit attempt and C2 are never predicted at all. Binary detection from this metadata is viable; fine-grained attribution is not.

## Reproducing the results

```bash
git clone https://github.com/moh1984/ids-metadata-evaluation-protocol.git
cd ids-metadata-evaluation-protocol
pip install -r requirements.txt

# 1. binary detection, tuning, McNemar, alert volume  (30-60 min on two cores)
python src/ids_pipeline.py --data data/cybersecurity_dataset.csv --out results/ --seed 42

# 2. feature ablation + per-fold CV scores            (about 10 min)
python src/feature_ablation.py

# 3. multi-class + SHAP on the tuned model            (about 20 min)
python src/multiclass_shap.py

# 4. all nine figures, drawn at final print size      (about 1 min)
python src/make_figures.py
```

Add `--fast` to step 1 for a smaller grid when checking that the data loads correctly; it
takes a few minutes, though the exact time depends on the hardware. All timings quoted here
were measured on two CPU cores and scale with core count.

**Google Colab:** open `notebooks/IDS_pipeline_colab.ipynb`. Colab ships everything except `shap`, which the first cell installs. No GPU required — set Runtime → CPU.

All randomness is controlled by `--seed` (default 42), which every script accepts and `run_all.sh` propagates to all four: it fixes the split, the CV folds and every estimator.

Dependency versions are **pinned** in `requirements.txt`. XGBoost in particular is version-sensitive (histogram binning), and loosening the pin produces small numerical drift in the third decimal.

### Column names

The scripts expect the 13 attributes below. If yours differ, edit `COLUMN_MAP` at the top of `src/ids_pipeline.py`; the script exits with a message naming any missing column.

```
timestamp, src_ip, dst_ip, src_port, dst_port, protocol, bytes_sent,
bytes_received, user_agent, url, is_internal_traffic, label, attack_type
```

## Repository layout

```
data/            the 10,000-record dataset (see Data provenance below)
src/             three experiment scripts plus the figure-generation script
notebooks/       Colab runner
results/tables/  every table in the paper, as CSV
results/figures/ all nine figures, exactly as published
paper/           the manuscript
docs/            what the corrected protocol changes and why
```

## Method summary

1. **Split first.** Stratified 70/30 on the ten-class label over all 10,000 records. The test partition keeps the 24:1 ratio and is untouched by balancing, scaling and tuning.
2. **Balance the training partition only**, by class weighting (`class_weight='balanced'`, `scale_pos_weight` for XGBoost) rather than under-sampling, which would have discarded 95.8% of the benign records.
3. **No leakage.** `StandardScaler` sits inside a `Pipeline`, so it is refitted on each CV training fold.
4. **Tune on training data only.** Binary task: five-fold stratified `GridSearchCV` scored by average precision (PR-AUC), since accuracy is uninformative at a 4% positive rate. Multi-class task: an independent five-fold stratified `GridSearchCV` scored by macro F1, which weights all ten classes equally instead of following the benign majority.
5. **Report metrics that survive imbalance.** Attack-class precision/recall/F1, balanced accuracy, FPR, ROC-AUC and PR-AUC, plus raw TP/FP/FN counts.

## Data provenance

`data/cybersecurity_dataset.csv` contains 10,000 flow records generated from a **simulated** enterprise network environment: 9,600 benign and 400 attack, across nine attack categories (brute-force 108, port-scan 81, SQL injection 64, XSS 34, credential stuffing 28, DDoS 27, command injection 26, exploit attempt 22, C2 10).

Simulation gives clean labels and controlled attack proportions, but the regularities the classifiers exploit are those the generator produced. Results here should not be read as an estimate of accuracy on production traffic. Section 6 of the paper states this and the other limitations in full.

## Citation

See `CITATION.cff`, or cite the paper directly once published.

## Repository

`ids-metadata-evaluation-protocol` — https://github.com/moh1984/ids-metadata-evaluation-protocol

## License

Code released under the MIT License (`LICENSE`). If your institution requires different terms, replace the file before making the repository public.
