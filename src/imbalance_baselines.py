#!/usr/bin/env python3
"""
Matched comparison of imbalance-handling strategies on the same data and split.

All three reviewers converged on the same requirement: partitioning before
resampling is now recognised good practice, so the paper must say what is new
beyond it, and must compare against a contemporary imbalance-aware approach on
the same data and the same split rather than against literature numbers obtained
on other datasets and other priors. The third reviewer added that the comparison
must use an established, categorical-aware implementation with a clearly stated
tuning budget, because an ad-hoc oversampler is not evidence about contemporary
practice.

This script holds everything fixed except the imbalance strategy:

    A  class weighting                        (as submitted)
    B  SMOTENC on the training partition      (imbalanced-learn reference impl.)
    C  random under-sampling                  (the pre-correction protocol)
    D  class weighting + threshold tuned on training folds

All four are fitted on the same 7,000-record training partition, from the same
stratified split, and scored on the same untouched 3,000-record test partition
at the 4% prevalence of the source data. Strategy C is included because it is
what the earlier version of this work did, so its row quantifies the cost of the
protocol error rather than merely asserting it.

Why SMOTENC rather than SMOTE
-----------------------------
Fourteen of the 26 engineered features are not continuous: thirteen are 0/1
indicators and one, protocol_encoded, is a three-level nominal code. Plain SMOTE
interpolates every column as a real number, which invents fractional values for
columns that have no fractional meaning. SMOTENC (Chawla et al. [16], Sec. 6.1)
is the variant defined for mixed continuous and categorical data: it interpolates
the continuous columns and assigns each synthetic point the majority category of
its k nearest neighbours for the categorical ones. CATEGORICAL below lists the
fourteen columns passed to it.

Earlier versions of this script carried a hand-written oversampler so that the
repository had no dependency beyond the pinned list. That trade was the wrong
way round -- a dependency is cheaper than a non-standard baseline -- and it has
been replaced by the reference implementation.

Tuning budget
-------------
No hyperparameter search is run here. Every strategy reuses the configuration
selected for its model in Table 1, so the four rows of a model differ only in
how imbalance is handled. The only quantity fitted inside this script is the
decision threshold of strategy D, chosen on a 91-point grid over [0.05, 0.95]
by maximising attack F1 on out-of-fold training predictions (five-fold
stratified). The test partition is not consulted by any strategy.

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

try:
    from imblearn.over_sampling import SMOTENC
    HAVE_IMBLEARN = True
except ImportError:
    HAVE_IMBLEARN = False

TAB = "results/tables"

# The thirteen 0/1 indicators plus the three-level protocol code. Ports and the
# hour of day are left continuous: they are ordered integers with large or
# cyclic ranges, and the classifiers already consume them as numbers.
CATEGORICAL = [
    "is_wellknown_dst", "is_wellknown_src", "same_port",
    "dst_is_service", "src_is_service",
    "ua_is_bot", "ua_is_mobile",
    "has_url", "url_has_login", "url_has_admin", "url_has_param",
    "is_night", "is_internal",
    "protocol_encoded",
]


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

    if not HAVE_IMBLEARN:
        sys.exit("imbalanced-learn is required for strategy B. "
                 "Install it with: pip install imbalanced-learn")

    df = pd.read_csv(args.data)
    X = engineer_features(df)
    y = df["label"].astype(int).values
    m = df["attack_type"].astype(str).values
    Xtr, Xte, ytr, yte, _, _ = train_test_split(
        X, y, m, test_size=.30, stratify=pd.Series(m), random_state=SEED)
    pw = (len(ytr) - ytr.sum()) / ytr.sum()
    cat_idx = [X.columns.get_loc(c) for c in CATEGORICAL]
    print(f"train {len(ytr):,} ({ytr.sum()} attack) | test {len(yte):,} ({yte.sum()} attack)")
    print(f"SMOTENC categorical columns: {len(cat_idx)} of {X.shape[1]}\n")

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

    # SMOTENC is fitted once on the training partition and reused for every
    # model, so the three B rows see exactly the same resampled training set.
    sm = SMOTENC(categorical_features=cat_idx, k_neighbors=5, random_state=SEED)
    Xs, ys = sm.fit_resample(Xtr, ytr)
    Xs = pd.DataFrame(np.asarray(Xs), columns=Xtr.columns)
    print(f"SMOTENC training set: {len(ys):,} records "
          f"({int(ys.sum())} attack, {int((ys == 0).sum())} benign)\n")

    rows = []
    for name in names:
        # ---- A. class weighting, as submitted ----
        pipeA = Pipeline([("sc", StandardScaler()), ("clf", make(name, True))]).fit(Xtr, ytr)
        sA = pipeA.predict_proba(Xte)[:, 1]
        rows.append({"Model": name, "Imbalance strategy": "A. Class weighting (as submitted)",
                     **report(yte, pipeA.predict(Xte), sA)})

        # ---- B. SMOTENC on the training partition only ----
        pipeB = Pipeline([("sc", StandardScaler()), ("clf", make(name, False))]).fit(Xs, ys)
        rows.append({"Model": name, "Imbalance strategy": "B. SMOTENC on training partition",
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
