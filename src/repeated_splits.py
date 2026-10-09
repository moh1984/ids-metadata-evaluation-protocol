#!/usr/bin/env python3
"""
Repeated-split stability check.

Reviewer 2 asked whether the model ordering survives resampling of the split, or
whether it is an artefact of the one stratified partition used throughout the
paper. This script repeats the whole binary experiment over several seeds and
reports the mean and standard deviation of each metric, plus how often each model
takes first place on PR-AUC.

The hyperparameters are fixed at the values selected by the grid search reported
in Table 1 rather than re-searched on every seed. That is deliberate: the
question is whether the ranking is stable under resampling, not whether the
tuning is. Re-searching would confound the two and multiply the runtime by the
size of the grids.

Only the split changes between seeds. The 70/30 ratio, the stratification by
attack category and the 4% prevalence of the test partition are identical every
time, so nothing about the evaluation protocol varies.

    python src/repeated_splits.py [--data ...] [--seeds 42 1 7 13 2024]

Writes results/tables/table15_repeated_splits.csv
       results/tables/table15b_per_seed.csv
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ids_pipeline import engineer_features

try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except ImportError:
    HAVE_XGB = False
    print("[warn] xgboost missing: gradient boosting will be skipped, which leaves the "
          "comparison that matters most out of the table. Install it before using these results.")

TAB = "results/tables"


def build(seed, pos_weight):
    """The Table 1 configurations, rebuilt for a given seed."""
    def pipe(clf):
        return Pipeline([("sc", StandardScaler()), ("clf", clf)])
    m = {
        "Logistic Regression": pipe(LogisticRegression(
            C=0.01, max_iter=5000, class_weight="balanced", random_state=seed)),
        "SVM (RBF Kernel)": pipe(SVC(
            C=0.1, gamma=0.001, probability=True, class_weight="balanced", random_state=seed)),
        "Random Forest": pipe(RandomForestClassifier(
            n_estimators=500, max_depth=None, max_features=0.5, min_samples_split=3,
            class_weight="balanced_subsample", random_state=seed, n_jobs=-1)),
        "Neural Network (FFNN)": pipe(MLPClassifier(
            hidden_layer_sizes=(64, 32), alpha=0.01, learning_rate_init=0.01,
            max_iter=1500, early_stopping=True, random_state=seed)),
    }
    if HAVE_XGB:
        m["Gradient Boosting"] = pipe(XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=6, subsample=1.0,
            scale_pos_weight=pos_weight, eval_metric="logloss", tree_method="hist",
            random_state=seed, n_jobs=-1))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("IDS_DATA", "data/cybersecurity_dataset.csv"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 7, 13, 2024])
    args = ap.parse_args()
    os.makedirs(TAB, exist_ok=True)

    df = pd.read_csv(args.data)
    X = engineer_features(df)
    y = df["label"].astype(int).values
    cat = df["attack_type"].astype(str).values

    per_seed = []
    for seed in args.seeds:
        Xtr, Xte, ytr, yte, _, _ = train_test_split(
            X, y, cat, test_size=.30, stratify=pd.Series(cat), random_state=seed)
        pw = (len(ytr) - ytr.sum()) / ytr.sum()
        print(f"\n== seed {seed} == train {len(ytr):,} ({ytr.sum()} attack) | "
              f"test {len(yte):,} ({yte.sum()} attack)", flush=True)

        for name, est in build(seed, pw).items():
            est.fit(Xtr, ytr)
            score = est.predict_proba(Xte)[:, 1]
            pred = est.predict(Xte)
            tn, fp, fn, tp = confusion_matrix(yte, pred, labels=[0, 1]).ravel()
            per_seed.append({
                "Seed": seed, "Model": name,
                "PR-AUC": average_precision_score(yte, score),
                "Attack precision (%)": 100 * tp / (tp + fp) if tp + fp else 0.0,
                "Attack recall (%)": 100 * tp / (tp + fn) if tp + fn else 0.0,
                "FPR (%)": 100 * fp / (fp + tn) if fp + tn else 0.0,
            })
            print(f"   {name:24s} PR-AUC {per_seed[-1]['PR-AUC']:.4f}", flush=True)

    raw = pd.DataFrame(per_seed)
    raw.round(4).to_csv(f"{TAB}/table15b_per_seed.csv", index=False)

    # how often does each model take first place on PR-AUC?
    wins = raw.loc[raw.groupby("Seed")["PR-AUC"].idxmax(), "Model"].value_counts()

    agg = raw.groupby("Model").agg(["mean", "std"])
    out = pd.DataFrame({
        "Model": agg.index,
        "PR-AUC mean": agg[("PR-AUC", "mean")].round(4).values,
        "PR-AUC sd": agg[("PR-AUC", "std")].round(4).values,
        "Attack precision mean (%)": agg[("Attack precision (%)", "mean")].round(2).values,
        "Attack recall mean (%)": agg[("Attack recall (%)", "mean")].round(2).values,
        "FPR mean (%)": agg[("FPR (%)", "mean")].round(4).values,
        "Best on PR-AUC": [int(wins.get(m, 0)) for m in agg.index],
    }).sort_values("PR-AUC mean", ascending=False)
    out["Rank"] = range(1, len(out) + 1)
    out.to_csv(f"{TAB}/table15_repeated_splits.csv", index=False)

    print(f"\n\n== across {len(args.seeds)} stratified splits ==")
    print(out.to_string(index=False))
    print(f"\nFirst place on PR-AUC, out of {len(args.seeds)} splits: "
          + ", ".join(f"{m} {n}" for m, n in wins.items()))


if __name__ == "__main__":
    main()
