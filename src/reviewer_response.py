#!/usr/bin/env python3
"""
Reviewer-requested additions to the binary experiment.

1. ATTACK-FOCUSED UNCERTAINTY (both reviewers)
   The submitted paper established the ensemble advantage with McNemar tests on
   sample-wise correctness. At 96% benign, overall correctness is dominated by
   majority-class decisions, so that test does not speak to PR-AUC, attack
   precision, attack recall or false-positive rate -- the quantities the paper
   actually argues about. This script re-does the comparison by paired bootstrap
   resampling of the held-out predictions: the same resampled index is applied to
   every model, so the differences are paired and the intervals are directly
   comparable.

2. IMBALANCE-AWARE FFNN (first reviewer)
   The submitted pipeline gave class weights to logistic regression, SVM and
   random forest, and scale_pos_weight to gradient boosting, but nothing to the
   feed-forward network, which then recorded the worst attack recall (13.33%).
   That asymmetry makes the cross-model conclusion unsafe.

   MLPClassifier has no class_weight argument, but its fit() does accept
   sample_weight, which is the correct algorithm-level remedy: it reweights the
   loss without discarding any data. Two balanced variants are therefore
   reported next to the original -- sample weighting, and the resampling to 1:1
   that a missing class_weight might otherwise tempt one into -- so the effect
   of the omission can be separated from the effect of the remedy chosen.

    python src/reviewer_response.py [--data ...] [--seed 42] [--boot 2000]

Writes results/tables/table12_bootstrap_intervals.csv,
       results/tables/table13_ffnn_balanced.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils import resample

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ids_pipeline import engineer_features

try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except ImportError:
    HAVE_XGB = False
    print("[warn] xgboost missing: the gradient boosting rows will be skipped. "
          "Install the pinned version to reproduce the published numbers.")

TAB = "results/tables"


# --------------------------------------------------------------- metrics
# Threshold metrics use each estimator's own predict(); PR-AUC uses its scores.
# This matters for SVC, whose predict() follows the decision function while
# predict_proba() is a separately fitted Platt calibration, so the two can
# disagree on a large fraction of samples. Table 4 of the paper reports
# predict(), and these comparisons follow it.
def metrics(y, score, pred=None, thr=0.5):
    if pred is None:
        pred = (score >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "PR-AUC": average_precision_score(y, score),
        "Attack precision": tp / (tp + fp) if tp + fp else 0.0,
        "Attack recall": tp / (tp + fn) if tp + fn else 0.0,
        "FPR": fp / (fp + tn) if fp + tn else 0.0,
    }


def boot_intervals(y, scores, preds, n_boot, rng):
    """Paired bootstrap: one resampled index, applied to every model."""
    names = list(scores)
    draws = {m: {k: [] for k in ("PR-AUC", "Attack precision", "Attack recall", "FPR")}
             for m in names}
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].sum() == 0:                 # a draw with no attacks says nothing
            continue
        for m in names:
            for k, v in metrics(y[idx], scores[m][idx], preds[m][idx]).items():
                draws[m][k].append(v)
    return draws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("IDS_DATA", "data/cybersecurity_dataset.csv"))
    ap.add_argument("--seed", type=int, default=int(os.environ.get("IDS_SEED", 42)))
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    SEED = args.seed
    os.makedirs(TAB, exist_ok=True)

    df = pd.read_csv(args.data)
    X = engineer_features(df)
    y = df["label"].astype(int).values
    m = df["attack_type"].astype(str).values
    Xtr, Xte, ytr, yte, _, _ = train_test_split(
        X, y, m, test_size=.30, stratify=pd.Series(m), random_state=SEED)
    pw = (len(ytr) - ytr.sum()) / ytr.sum()

    def pipe(clf):
        return Pipeline([("sc", StandardScaler()), ("clf", clf)])

    # the configurations selected by the grid search reported in Table 1
    models = {
        "Logistic Regression": pipe(LogisticRegression(
            C=0.01, max_iter=5000, class_weight="balanced", random_state=SEED)),
        "SVM (RBF Kernel)": pipe(SVC(
            C=0.1, gamma=0.001, probability=True, class_weight="balanced", random_state=SEED)),
        "Random Forest": pipe(RandomForestClassifier(
            n_estimators=500, max_depth=None, max_features=0.5, min_samples_split=3,
            class_weight="balanced_subsample", random_state=SEED, n_jobs=-1)),
        "Neural Network (FFNN)": pipe(MLPClassifier(
            hidden_layer_sizes=(64, 32), alpha=0.01, learning_rate_init=0.01,
            max_iter=1500, early_stopping=True, random_state=SEED)),
    }
    if HAVE_XGB:
        models["Gradient Boosting"] = pipe(XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=6, subsample=1.0,
            scale_pos_weight=pw, eval_metric="logloss", tree_method="hist",
            random_state=SEED, n_jobs=-1))

    scores, preds = {}, {}
    for name, est in models.items():
        print(f"[fit] {name}", flush=True)
        est.fit(Xtr, ytr)
        scores[name] = est.predict_proba(Xte)[:, 1]
        preds[name] = est.predict(Xte)

    # ---------------------------------------------------- 2. balanced FFNN
    def mlp():
        return pipe(MLPClassifier(hidden_layer_sizes=(64, 32), alpha=0.01,
                                  learning_rate_init=0.01, max_iter=1500,
                                  early_stopping=True, random_state=SEED))

    # (a) sample weighting: the loss is reweighted, no data is discarded
    w = np.where(ytr == 1, pw, 1.0)
    ffnn_sw = mlp()
    print("[fit] Neural Network (sample-weighted)", flush=True)
    ffnn_sw.fit(Xtr, ytr, clf__sample_weight=w)
    scores["Neural Network (sample-weighted)"] = ffnn_sw.predict_proba(Xte)[:, 1]
    preds["Neural Network (sample-weighted)"] = ffnn_sw.predict(Xte)

    # (b) resampling to 1:1, which throws away 95% of the benign training data
    pos = np.where(ytr == 1)[0]
    neg = np.where(ytr == 0)[0]
    bal = np.concatenate([pos, resample(neg, n_samples=len(pos), replace=False,
                                        random_state=SEED)])
    ffnn_us = mlp()
    print("[fit] Neural Network (under-sampled 1:1)", flush=True)
    ffnn_us.fit(Xtr.iloc[bal] if hasattr(Xtr, "iloc") else Xtr[bal], ytr[bal])
    scores["Neural Network (under-sampled)"] = ffnn_us.predict_proba(Xte)[:, 1]
    preds["Neural Network (under-sampled)"] = ffnn_us.predict(Xte)

    def row(label, key):
        return {"Variant": label,
                **{k: round(v * 100, 2) if k != "PR-AUC" else round(v, 4)
                   for k, v in metrics(yte, scores[key], preds[key]).items()}}

    ffnn = pd.DataFrame([
        row("As submitted (no imbalance handling)", "Neural Network (FFNN)"),
        row("Sample weighting in fit()", "Neural Network (sample-weighted)"),
        row("Training partition resampled to 1:1", "Neural Network (under-sampled)"),
    ])
    ffnn.to_csv(f"{TAB}/table13_ffnn_balanced.csv", index=False)
    print("\n== FFNN with and without imbalance handling ==")
    print(ffnn.to_string(index=False))

    # ------------------------------------------- 1. paired bootstrap intervals
    print(f"\n[bootstrap] {args.boot} paired resamples ...", flush=True)
    rng = np.random.default_rng(SEED)
    draws = boot_intervals(yte, scores, preds, args.boot, rng)

    rows = []
    for name in scores:
        point = metrics(yte, scores[name], preds[name])
        row = {"Model": name}
        for k, v in point.items():
            lo, hi = np.percentile(draws[name][k], [2.5, 97.5])
            scale = 1 if k == "PR-AUC" else 100
            row[k] = round(v * scale, 4 if scale == 1 else 2)
            row[f"{k} 95% CI"] = f"[{lo*scale:.4g}, {hi*scale:.4g}]"
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(f"{TAB}/table12_bootstrap_intervals.csv", index=False)
    print("\n== attack-focused metrics with paired bootstrap intervals ==")
    print(out.to_string(index=False))

    # ---------------------------------- paired differences, the correct test
    # Whether two independent intervals overlap is not a test of their
    # difference: intervals can overlap while the paired difference excludes
    # zero, and the reverse. Since every model was scored on the same resampled
    # index, the difference can be formed inside each resample and its interval
    # read directly. That is what decides significance below.
    ref = "Gradient Boosting" if "Gradient Boosting" in scores else "Random Forest"
    diffs = []
    for name in scores:
        if name == ref or name.startswith("Neural Network ("):
            if name != "Neural Network (FFNN)":
                continue
        if name == ref:
            continue
        for metric in ("PR-AUC", "Attack precision", "Attack recall", "FPR"):
            d = np.array(draws[ref][metric]) - np.array(draws[name][metric])
            lo, hi = np.percentile(d, [2.5, 97.5])
            scale = 1 if metric == "PR-AUC" else 100
            diffs.append({
                "Comparison": f"{ref} minus {name}",
                "Metric": metric,
                "Difference": round(float(d.mean()) * scale, 4 if scale == 1 else 2),
                "95% CI": f"[{lo*scale:.4g}, {hi*scale:.4g}]",
                "Excludes zero": "yes" if not (lo <= 0 <= hi) else "no",
            })
    dd = pd.DataFrame(diffs)
    dd.to_csv(f"{TAB}/table12b_paired_differences.csv", index=False)
    print(f"\n== paired differences against {ref} (the test the conclusions rest on) ==")
    print(dd.to_string(index=False))


if __name__ == "__main__":
    main()
