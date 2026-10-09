#!/usr/bin/env python3
"""
Matched comparison of imbalance-handling strategies on the same data and split.

Both reviewers asked the same thing in different words: partitioning before
resampling is now recognised good practice, so the paper must say what is new
beyond it, and must compare against a contemporary imbalance-aware approach on
the same data and the same split rather than against literature numbers obtained
on other datasets and other priors.

This script holds everything fixed except the imbalance strategy:

    A  class weighting                 (as submitted)
    B  SMOTE-style oversampling of the training partition (simplified; see below)
    C  random under-sampling           (the pre-correction protocol)
    D  class weighting + threshold tuned on training folds

All four are fitted on the same 7,000-record training partition, from the same
stratified split, and scored on the same untouched 3,000-record test partition
at the 4% prevalence of the source data. Strategy C is included because it is
what the earlier version of this work did, so its row quantifies the cost of the
protocol error rather than merely asserting it.

The oversampling baseline is implemented here rather than imported, so the
repository carries no dependency beyond the pinned list. It follows the idea of
Chawla et al. [16] -- interpolate each minority sample towards one of its k
nearest minority neighbours -- but simplifies it in two ways: the interpolation
coefficient is drawn independently per feature rather than once along the line
to the neighbour, and the binary indicators among the 26 features are treated as
continuous. It is therefore a SMOTE-style baseline, not a faithful SMOTE or
SMOTENC implementation, and the manuscript says so under Table 10.

    python src/imbalance_baselines.py [--data ...] [--seed 42]

Writes results/tables/table14_imbalance_strategies.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ids_pipeline import engineer_features

try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except ImportError:
    HAVE_XGB = False

TAB = "results/tables"


def smote(X, y, seed, k=5):
    """SMOTE-style oversampling: interpolate each minority sample towards a
    neighbour. Simplified as described in the module docstring."""
    rng = np.random.default_rng(seed)
    Xv = X.values if hasattr(X, "values") else X
    minority = Xv[y == 1]
    n_needed = int((y == 0).sum() - (y == 1).sum())
    if n_needed <= 0 or len(minority) < 2:
        return X, y
    k = min(k, len(minority) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(minority)
    _, idx = nn.kneighbors(minority)
    base = rng.integers(0, len(minority), n_needed)
    pick = rng.integers(1, k + 1, n_needed)          # skip the point itself
    gap = rng.random((n_needed, Xv.shape[1]))
    synth = minority[base] + gap * (minority[idx[base, pick]] - minority[base])
    Xn = np.vstack([Xv, synth])
    yn = np.concatenate([y, np.ones(n_needed, dtype=int)])
    return pd.DataFrame(Xn, columns=X.columns) if hasattr(X, "columns") else Xn, yn


def report(y, pred, score):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "PR-AUC": round(average_precision_score(y, score), 4),
        "Attack precision (%)": round(100 * tp / (tp + fp), 2) if tp + fp else 0.0,
        "Attack recall (%)": round(100 * tp / (tp + fn), 2) if tp + fn else 0.0,
        "FPR (%)": round(100 * fp / (fp + tn), 4) if fp + tn else 0.0,
        "Attack F1 (%)": round(100 * f1_score(y, pred, zero_division=0), 2),
        "TP": int(tp), "FP": int(fp),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("IDS_DATA", "data/cybersecurity_dataset.csv"))
    ap.add_argument("--seed", type=int, default=int(os.environ.get("IDS_SEED", 42)))
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
    print(f"train {len(ytr):,} ({ytr.sum()} attack) | test {len(yte):,} ({yte.sum()} attack)\n")

    def make(name, weighted):
        """The Table 1 configurations, with class weighting switched on or off."""
        if name == "Logistic Regression":
            return LogisticRegression(C=0.01, max_iter=5000, random_state=SEED,
                                      class_weight="balanced" if weighted else None)
        if name == "Random Forest":
            return RandomForestClassifier(
                n_estimators=500, max_depth=None, max_features=0.5, min_samples_split=3,
                random_state=SEED, n_jobs=-1,
                class_weight="balanced_subsample" if weighted else None)
        if name == "Gradient Boosting" and HAVE_XGB:
            return XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6,
                                 subsample=1.0, scale_pos_weight=pw if weighted else 1,
                                 eval_metric="logloss", tree_method="hist",
                                 random_state=SEED, n_jobs=-1)
        return None

    names = ["Logistic Regression", "Random Forest"] + (["Gradient Boosting"] if HAVE_XGB else [])
    if not HAVE_XGB:
        print("[warn] xgboost missing: the gradient boosting rows are skipped.\n")

    pos, neg = np.where(ytr == 1)[0], np.where(ytr == 0)[0]
    under = np.concatenate([pos, resample(neg, n_samples=len(pos), replace=False,
                                          random_state=SEED)])

    rows = []
    for name in names:
        # ---- A. class weighting, as submitted ----
        pipeA = Pipeline([("sc", StandardScaler()), ("clf", make(name, True))]).fit(Xtr, ytr)
        sA = pipeA.predict_proba(Xte)[:, 1]
        rows.append({"Model": name, "Imbalance strategy": "A. Class weighting (as submitted)",
                     **report(yte, pipeA.predict(Xte), sA)})

        # ---- B. SMOTE on the training partition only ----
        Xs, ys = smote(Xtr, ytr, SEED)
        pipeB = Pipeline([("sc", StandardScaler()), ("clf", make(name, False))]).fit(Xs, ys)
        rows.append({"Model": name, "Imbalance strategy": "B. SMOTE on training partition",
                     **report(yte, pipeB.predict(Xte), pipeB.predict_proba(Xte)[:, 1])})

        # ---- C. random under-sampling, the pre-correction protocol ----
        pipeC = Pipeline([("sc", StandardScaler()), ("clf", make(name, False))]).fit(
            Xtr.iloc[under], ytr[under])
        rows.append({"Model": name, "Imbalance strategy": "C. Random under-sampling",
                     **report(yte, pipeC.predict(Xte), pipeC.predict_proba(Xte)[:, 1])})

        # ---- D. class weighting with the threshold tuned on training folds ----
        cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
        oof = np.zeros(len(ytr))
        for tr, va in cv.split(Xtr, ytr):
            p = Pipeline([("sc", StandardScaler()), ("clf", make(name, True))]).fit(
                Xtr.iloc[tr], ytr[tr])
            oof[va] = p.predict_proba(Xtr.iloc[va])[:, 1]
        grid = np.linspace(.05, .95, 91)
        best = max(grid, key=lambda t: f1_score(ytr, (oof >= t).astype(int), zero_division=0))
        rows.append({"Model": name,
                     "Imbalance strategy": f"D. Class weighting + tuned threshold ({best:.2f})",
                     **report(yte, (sA >= best).astype(int), sA)})
        print(f"[done] {name}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(f"{TAB}/table14_imbalance_strategies.csv", index=False)
    print()
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
