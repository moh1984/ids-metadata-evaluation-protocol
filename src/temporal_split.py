#!/usr/bin/env python3
"""
Chronological split: the same models, evaluated without a random partition.

The third reviewer asked for "an episode-, time-, host-, or independently
generated test split" to test whether the reported performance depends on
related records appearing on both sides of a random flow-level split. The
records carry no episode, session or host identifier -- every source and
destination address in the file is distinct -- so a grouped split cannot be
built. They do carry timestamps spanning three months, so a chronological
split can.

Records are sorted by timestamp, the first 70% become the training partition
and the last 30% the test partition. No record in the test partition precedes
any record in the training partition, so anything the generator produced as a
time-localised episode falls on one side of the cut rather than being split
across it. Everything else -- the feature set, the model configurations, the
class weighting, the seed -- is identical to the main experiment, so the only
difference from Table 4 is how the partition was formed.

Prevalence is not forced: the chronological test partition carries whatever
attack rate that period of the simulation happens to contain, which is itself
worth reporting.

    python src/temporal_split.py [--data ...] [--seed 42] [--frac 0.70]

Writes results/tables/table17_temporal_split.csv
"""

import argparse
import os
import sys

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ids_pipeline import engineer_features

try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except ImportError:
    HAVE_XGB = False

TAB = "results/tables"

# PR-AUC from the random stratified split, Table 4, for side-by-side reading.
RANDOM_SPLIT = {
    "Logistic Regression": 0.3104,
    "Random Forest": 0.6594,
    "Gradient Boosting": 0.6700,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("IDS_DATA", "data/cybersecurity_dataset.csv"))
    ap.add_argument("--seed", type=int, default=int(os.environ.get("IDS_SEED", 42)))
    ap.add_argument("--frac", type=float, default=0.70)
    args = ap.parse_args()
    SEED = args.seed
    os.makedirs(TAB, exist_ok=True)

    df = pd.read_csv(args.data)
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    assert ts.notna().all(), "every record needs a parseable timestamp for a chronological split"
    df = df.assign(_ts=ts).sort_values("_ts").reset_index(drop=True)

    X = engineer_features(df)
    y = df["label"].astype(int).values
    cut = int(len(df) * args.frac)
    Xtr, Xte, ytr, yte = X.iloc[:cut], X.iloc[cut:], y[:cut], y[cut:]
    pw = (len(ytr) - ytr.sum()) / ytr.sum()

    print(f"train {len(ytr):,} ({ytr.sum()} attack, {100 * ytr.mean():.2f}%)  "
          f"{df['_ts'].iloc[0]} -> {df['_ts'].iloc[cut - 1]}")
    print(f"test  {len(yte):,} ({yte.sum()} attack, {100 * yte.mean():.2f}%)  "
          f"{df['_ts'].iloc[cut]} -> {df['_ts'].iloc[-1]}\n")

    models = {
        "Logistic Regression": LogisticRegression(
            C=0.01, max_iter=5000, random_state=SEED, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=500, max_depth=None, max_features=0.5, min_samples_split=3,
            random_state=SEED, n_jobs=-1, class_weight="balanced_subsample"),
    }
    if HAVE_XGB:
        models["Gradient Boosting"] = XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=6, subsample=1.0,
            scale_pos_weight=pw, eval_metric="logloss", tree_method="hist",
            random_state=SEED, n_jobs=-1)
    else:
        print("[warn] xgboost missing: the gradient boosting row is skipped.\n")

    rows = []
    for name, clf in models.items():
        pipe = Pipeline([("sc", StandardScaler()), ("clf", clf)]).fit(Xtr, ytr)
        s = pipe.predict_proba(Xte)[:, 1]
        tn, fp, fn, tp = confusion_matrix(yte, pipe.predict(Xte), labels=[0, 1]).ravel()
        rows.append({
            "Model": name,
            "PR-AUC (chronological)": round(average_precision_score(yte, s), 4),
            "PR-AUC (random, Table 4)": RANDOM_SPLIT.get(name, float("nan")),
            "Attack precision (%)": round(100 * tp / (tp + fp), 2) if tp + fp else 0.0,
            "Attack recall (%)": round(100 * tp / (tp + fn), 2) if tp + fn else 0.0,
            "FPR (%)": round(100 * fp / (fp + tn), 4) if fp + tn else 0.0,
            "ROC-AUC": round(roc_auc_score(yte, s), 4),
            "TP": int(tp), "FP": int(fp),
        })
        print(f"[done] {name}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(f"{TAB}/table17_temporal_split.csv", index=False)
    print()
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
