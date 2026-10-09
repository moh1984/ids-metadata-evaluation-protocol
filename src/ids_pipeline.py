#!/usr/bin/env python3
"""
=============================================================================
 Corrected experimental pipeline for:
   "Evaluation Protocol Matters More Than Model Choice: Intrusion Detection
    from Raw Network Metadata Under a Preserved Class Prior"
=============================================================================

This script re-runs the study under a protocol that fixes the three
methodological defects documented in Section 6 of the manuscript:

  FIX 1  Evaluation at the source-data prevalence
         The original code balanced the data BEFORE splitting, so the test
         set was 50% attack while the source data is 4% attack. Here the
         stratified split happens FIRST on all 10,000 records, so the test
         partition preserves the 24:1 ratio of the source data. Balancing is applied to
         the TRAINING partition only. Adds FPR, ROC-AUC and PR-AUC.

  FIX 2  Hyperparameter search
         The original hyperparameters were fixed a priori. Here every model
         is tuned by GridSearchCV on the training partition only. The binary
         search is scored by average precision (PR-AUC), the appropriate
         criterion under heavy imbalance; the multi-class search, in
         src/multiclass_shap.py, is scored independently by macro F1.

  FIX 3  No leakage
         StandardScaler is fitted INSIDE a Pipeline, so it is re-fitted on
         each CV training fold rather than on the full dataset. The test
         partition is never seen during tuning or model selection.

Also produced: paired McNemar tests on real predictions, Wilson confidence
intervals and an alert-volume estimate at the prevalence of the source data.

-----------------------------------------------------------------------------
USAGE
    python src/ids_pipeline.py --data data/cybersecurity_dataset.csv --out results/

    Optional:
      --fast          smaller grids for a quick smoke test; runtime depends
                      on hardware
      --seed 42       random seed (default 42)

EXPECTED INPUT COLUMNS (rename via COLUMN_MAP below if yours differ)
    timestamp, src_ip, dst_ip, src_port, dst_port, protocol,
    bytes_sent, bytes_received, user_agent, url,
    is_internal_traffic, label, attack_type

OUTPUT (written to <--out>/tables/)
    table01_best_hyperparameters.csv    Table 1  - selected hyperparameters
    table04_binary.csv                  Table 4  - binary results at the source-data prevalence
    table05_alert_volume.csv            legacy   - single-prior alert volume, superseded by Table 6
    table06_crossval_prauc.csv          Table 7  - cross-validation on the training partition
    table_mcnemar_significance.csv      Section 5.4 - paired McNemar tests
    summary.md                          everything above, paste-ready

Multi-class results, per-class results and SHAP are produced by
src/multiclass_shap.py; the feature ablation by src/feature_ablation.py.
=============================================================================
"""

import argparse
import json
import os
import sys
import warnings
from itertools import combinations

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

# Optional dependencies -------------------------------------------------------
try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    HAVE_XGB = False
    print("[warn] xgboost not installed - falling back to sklearn "
          "GradientBoostingClassifier. NOTE: the manuscript must then say "
          "'Gradient Boosting (sklearn)', not 'XGBoost'.")

try:
    from statsmodels.stats.contingency_tables import mcnemar
    HAVE_SM = True
except ImportError:
    HAVE_SM = False


# =============================================================================
# 0.  CONFIGURATION
# =============================================================================

COLUMN_MAP = {
    # "your_column_name": "expected_name"
}

SERVICE_PORTS = {80, 443, 22, 53, 21, 445, 1433, 3389}
BOT_PATTERNS = ("bot", "curl", "python", "wget", "scrapy", "spider",
                "sqlmap", "nmap", "nikto", "masscan")
MOBILE_PATTERNS = ("mobile", "android", "iphone", "ipad", "ipod")
ADMIN_PATTERNS = ("/admin", "/phpmyadmin", "/wp-login", "/.env",
                  "/manager", "/config", "/cpanel")
LOGIN_PATTERNS = ("login", "signin", "auth", "session", "logon")

FEATURE_ORDER = [
    # traffic volume (6)
    "bytes_sent", "bytes_received", "bytes_ratio", "total_bytes",
    "log_bytes_sent", "log_bytes_received",
    # port-level (8)
    "src_port", "dst_port", "port_diff", "is_wellknown_dst",
    "is_wellknown_src", "same_port", "dst_is_service", "src_is_service",
    # user-agent (3)
    "ua_is_bot", "ua_is_mobile", "ua_len",
    # url (5)
    "has_url", "url_has_login", "url_has_admin", "url_has_param", "url_len",
    # temporal (2)
    "hour", "is_night",
    # encoded categoricals (2)
    "protocol_encoded", "is_internal",
]  # 26 features


# =============================================================================
# 1.  FEATURE ENGINEERING  (unchanged from the manuscript, Section 3.1)
# =============================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transform 13 raw attributes into the 26 features of Section 3.1."""
    f = pd.DataFrame(index=df.index)

    # ---- traffic volume (6) -------------------------------------------------
    bs = pd.to_numeric(df["bytes_sent"], errors="coerce").fillna(0)
    br = pd.to_numeric(df["bytes_received"], errors="coerce").fillna(0)
    f["bytes_sent"] = bs
    f["bytes_received"] = br
    f["bytes_ratio"] = bs / (br + 1.0)
    f["total_bytes"] = bs + br
    f["log_bytes_sent"] = np.log1p(bs.clip(lower=0))
    f["log_bytes_received"] = np.log1p(br.clip(lower=0))

    # ---- port-level (8) -----------------------------------------------------
    sp = pd.to_numeric(df["src_port"], errors="coerce").fillna(0).astype(int)
    dp = pd.to_numeric(df["dst_port"], errors="coerce").fillna(0).astype(int)
    f["src_port"] = sp
    f["dst_port"] = dp
    f["port_diff"] = (sp - dp).abs()
    f["is_wellknown_dst"] = (dp < 1024).astype(int)
    f["is_wellknown_src"] = (sp < 1024).astype(int)
    f["same_port"] = (sp == dp).astype(int)
    f["dst_is_service"] = dp.isin(SERVICE_PORTS).astype(int)
    f["src_is_service"] = sp.isin(SERVICE_PORTS).astype(int)

    # ---- user-agent (3) -----------------------------------------------------
    ua = df["user_agent"].fillna("").astype(str)
    ua_l = ua.str.lower()
    f["ua_is_bot"] = ua_l.apply(lambda s: int(any(p in s for p in BOT_PATTERNS)))
    f["ua_is_mobile"] = ua_l.apply(lambda s: int(any(p in s for p in MOBILE_PATTERNS)))
    f["ua_len"] = ua.str.len()

    # ---- url (5) ------------------------------------------------------------
    url = df["url"].fillna("").astype(str)
    url_l = url.str.lower()
    f["has_url"] = (url.str.len() > 0).astype(int)
    f["url_has_login"] = url_l.apply(lambda s: int(any(p in s for p in LOGIN_PATTERNS)))
    f["url_has_admin"] = url_l.apply(lambda s: int(any(p in s for p in ADMIN_PATTERNS)))
    f["url_has_param"] = url_l.str.contains(r"\?|=|&", regex=True).astype(int)
    f["url_len"] = url.str.len()

    # ---- temporal (2) -------------------------------------------------------
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    hour = ts.dt.hour.fillna(0).astype(int)
    f["hour"] = hour
    f["is_night"] = ((hour >= 22) | (hour <= 5)).astype(int)

    # ---- encoded categoricals (2) -------------------------------------------
    proto = df["protocol"].fillna("TCP").astype(str).str.upper().str.strip()
    f["protocol_encoded"] = proto.map({"ICMP": 0, "TCP": 1, "UDP": 2}).fillna(1).astype(int)

    internal = df["is_internal_traffic"]
    if internal.dtype == object:
        internal = internal.astype(str).str.lower().isin(["true", "1", "yes", "y"])
    f["is_internal"] = internal.astype(int)

    f = f[FEATURE_ORDER].replace([np.inf, -np.inf], 0).fillna(0)
    assert f.shape[1] == 26, f"expected 26 features, built {f.shape[1]}"
    return f


# =============================================================================
# 2.  STATISTICS HELPERS
# =============================================================================

def wilson_ci(k, n, z=1.96):
    """Wilson score interval for a proportion, in percent."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (100 * (c - h) / d, 100 * (c + h) / d)


def mcnemar_exact(y_true, pred_a, pred_b):
    """Exact two-sided McNemar test on genuinely paired predictions."""
    a_ok = (pred_a == y_true)
    b_ok = (pred_b == y_true)
    n01 = int(np.sum(a_ok & ~b_ok))   # A right, B wrong
    n10 = int(np.sum(~a_ok & b_ok))   # A wrong, B right
    if HAVE_SM:
        table = [[int(np.sum(a_ok & b_ok)), n01],
                 [n10, int(np.sum(~a_ok & ~b_ok))]]
        p = float(mcnemar(table, exact=True).pvalue)
    else:
        from scipy.stats import binomtest
        n = n01 + n10
        p = 1.0 if n == 0 else float(
            binomtest(min(n01, n10), n, 0.5, alternative="two-sided").pvalue)
    return n01, n10, p


def binary_metrics(y_true, y_pred, y_score=None):
    """Attack class = 1. Reports the metrics an IDS paper actually needs."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    n = len(y_true)
    lo, hi = wilson_ci(int(tp + tn), n)
    out = {
        "Accuracy (%)": 100 * (tp + tn) / n,
        "Acc 95% CI": f"[{lo:.2f}-{hi:.2f}]",
        "Balanced Acc (%)": 100 * balanced_accuracy_score(y_true, y_pred),
        "Attack Precision (%)": 100 * precision_score(y_true, y_pred, zero_division=0),
        "Attack Recall (%)": 100 * recall_score(y_true, y_pred, zero_division=0),
        "Attack F1 (%)": 100 * f1_score(y_true, y_pred, zero_division=0),
        "FPR (%)": 100 * fp / (fp + tn) if (fp + tn) else 0.0,
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
    }
    if y_score is not None:
        out["ROC-AUC"] = roc_auc_score(y_true, y_score)
        out["PR-AUC"] = average_precision_score(y_true, y_score)
    return out


# =============================================================================
# 3.  MODEL ZOO  -  every estimator wrapped in a leakage-free Pipeline
# =============================================================================

def build_models(seed, fast, pos_weight):
    """Returns {name: (Pipeline, param_grid)}. Scaler is INSIDE the pipeline."""

    def pipe(clf):
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    if fast:
        grids = {
            "Logistic Regression": {"clf__C": [0.1, 1.0, 10.0]},
            "SVM (RBF Kernel)":    {"clf__C": [1.0, 10.0], "clf__gamma": ["scale", 0.01]},
            "Random Forest":       {"clf__n_estimators": [300],
                                    "clf__max_depth": [None, 15],
                                    "clf__min_samples_split": [2, 5]},
            "Gradient Boosting":   {"clf__n_estimators": [300],
                                    "clf__learning_rate": [0.05, 0.1],
                                    "clf__max_depth": [4, 6]},
            "Neural Network (FFNN)": {"clf__hidden_layer_sizes": [(128, 64, 32), (64, 32)],
                                      "clf__alpha": [1e-4, 1e-2]},
        }
    else:
        grids = {
            "Logistic Regression": {"clf__C": [0.01, 0.1, 0.5, 1.0, 10.0, 100.0],
                                    "clf__penalty": ["l2"]},
            "SVM (RBF Kernel)":    {"clf__C": [0.1, 1.0, 10.0, 100.0],
                                    "clf__gamma": ["scale", 0.1, 0.01, 0.001]},
            "Random Forest":       {"clf__n_estimators": [200, 300, 500],
                                    "clf__max_depth": [None, 10, 15, 25],
                                    "clf__min_samples_split": [2, 3, 5],
                                    "clf__max_features": ["sqrt", 0.5]},
            "Gradient Boosting":   {"clf__n_estimators": [200, 300, 500],
                                    "clf__learning_rate": [0.01, 0.05, 0.1],
                                    "clf__max_depth": [3, 4, 6, 8],
                                    "clf__subsample": [0.8, 1.0]},
            "Neural Network (FFNN)": {"clf__hidden_layer_sizes": [(128, 64, 32), (64, 32), (128,)],
                                      "clf__alpha": [1e-5, 1e-4, 1e-3, 1e-2],
                                      "clf__learning_rate_init": [1e-3, 1e-2]},
        }

    if HAVE_XGB:
        gb = XGBClassifier(random_state=seed, eval_metric="logloss",
                           tree_method="hist", scale_pos_weight=pos_weight,
                           n_jobs=-1)
    else:
        from sklearn.ensemble import GradientBoostingClassifier
        gb = GradientBoostingClassifier(random_state=seed)
        grids["Gradient Boosting"].pop("clf__max_features", None)

    models = {
        "Logistic Regression": pipe(LogisticRegression(
            max_iter=5000, class_weight="balanced", random_state=seed)),
        "SVM (RBF Kernel)": pipe(SVC(
            probability=True, class_weight="balanced", random_state=seed)),
        "Random Forest": pipe(RandomForestClassifier(
            class_weight="balanced_subsample", random_state=seed, n_jobs=-1)),
        "Gradient Boosting": pipe(gb),
        "Neural Network (FFNN)": pipe(MLPClassifier(
            max_iter=1500, early_stopping=True, random_state=seed)),
    }
    return {k: (models[k], grids[k]) for k in models}


# =============================================================================
# 4.  MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to the dataset CSV")
    ap.add_argument("--out", default="results", help="output directory")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fast", action="store_true", help="smaller grids")
    ap.add_argument("--test-size", type=float, default=0.30)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    TAB = os.path.join(args.out, 'tables')
    os.makedirs(TAB, exist_ok=True)
    rng = args.seed
    log = []

    def say(s=""):
        print(s)
        log.append(s)

    # ---- load ---------------------------------------------------------------
    df = pd.read_csv(args.data)
    if COLUMN_MAP:
        df = df.rename(columns=COLUMN_MAP)
    required = ["timestamp", "src_port", "dst_port", "protocol", "bytes_sent",
                "bytes_received", "user_agent", "url", "is_internal_traffic",
                "label", "attack_type"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        sys.exit(f"[error] missing columns: {missing}\n"
                 f"        found: {list(df.columns)}\n"
                 f"        edit COLUMN_MAP at the top of this script.")

    say("# Corrected pipeline results\n")
    say(f"Records: {len(df):,}")

    X = engineer_features(df)
    y = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int).values
    y_multi = df["attack_type"].fillna("benign").astype(str).str.lower().values

    n_att, n_ben = int(y.sum()), int((y == 0).sum())
    say(f"Benign: {n_ben:,} ({100*n_ben/len(y):.2f}%)  |  "
        f"Attack: {n_att:,} ({100*n_att/len(y):.2f}%)  |  ratio {n_ben/max(n_att,1):.1f}:1")
    say(f"Features engineered: {X.shape[1]}\n")

    # =========================================================================
    # FIX 1a - SPLIT FIRST, on the ORIGINAL distribution.
    #          The test set keeps the 24:1 ratio of the source data.
    # =========================================================================
    strat = pd.Series(y_multi)
    vc = strat.value_counts()
    if (vc < 2).any():                      # classes too rare to stratify
        strat = pd.Series(y)
    Xtr, Xte, ytr, yte, mtr, mte = train_test_split(
        X, y, y_multi, test_size=args.test_size,
        stratify=strat, random_state=rng)

    say(f"Train: {len(ytr):,} ({int(ytr.sum())} attack, "
        f"{100*ytr.mean():.2f}%)")
    say(f"Test : {len(yte):,} ({int(yte.sum())} attack, "
        f"{100*yte.mean():.2f}%)   <- source-data class prior preserved\n")

    pos_weight = (len(ytr) - ytr.sum()) / max(ytr.sum(), 1)

    # =========================================================================
    # FIX 2 + FIX 3 - GridSearchCV over leakage-free Pipelines,
    #                 on the TRAINING partition only, scored by PR-AUC.
    # =========================================================================
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=rng)
    # multi-class CV must not request more folds than the rarest class allows
    min_cls = int(pd.Series(mtr).value_counts().min())
    cv_mc = StratifiedKFold(n_splits=max(2, min(5, min_cls)), shuffle=True,
                            random_state=rng)
    models = build_models(rng, args.fast, pos_weight)

    fitted, cv_rows, best_rows = {}, [], []
    for name, (pipe, grid) in models.items():
        print(f"[tuning] {name} ...", flush=True)
        gs = GridSearchCV(pipe, grid, scoring="average_precision",
                          cv=cv, n_jobs=-1, refit=True)
        gs.fit(Xtr, ytr)
        fitted[name] = gs.best_estimator_
        i = gs.best_index_
        cv_rows.append({
            "Model": name,
            "CV PR-AUC (mean)": gs.cv_results_["mean_test_score"][i],
            "CV PR-AUC (std)": gs.cv_results_["std_test_score"][i],
        })
        best_rows.append({"Model": name,
                          "Best hyperparameters": json.dumps(gs.best_params_)})
        print(f"           best PR-AUC={gs.best_score_:.4f}  {gs.best_params_}")

    pd.DataFrame(cv_rows).to_csv(f"{TAB}/table06_crossval_prauc.csv", index=False)
    pd.DataFrame(best_rows).to_csv(f"{TAB}/table01_best_hyperparameters.csv", index=False)

    # =========================================================================
    # BINARY EVALUATION on the imbalanced test set
    # =========================================================================
    say("## Table 4 (replacement) - binary, evaluated at the source-data prevalence\n")
    rows, preds, scores = [], {}, {}
    for name, est in fitted.items():
        yp = est.predict(Xte)
        ys = est.predict_proba(Xte)[:, 1]
        preds[name], scores[name] = yp, ys
        r = {"Model": name}
        r.update(binary_metrics(yte, yp, ys))
        rows.append(r)

    t4 = pd.DataFrame(rows)
    t4.to_csv(f"{TAB}/table04_binary.csv", index=False)
    say(t4.round(3).to_markdown(index=False))
    say()

    # ---- operational alert volume ------------------------------------------
    say("## Operational alert volume implied by each FPR\n")
    ben_test = int((yte == 0).sum())
    op = []
    for r in rows:
        fpr = r["FPR (%)"] / 100
        op.append({
            "Model": r["Model"],
            "FPR (%)": r["FPR (%)"],
            "False alerts per 1,000 benign flows": 1000 * fpr,
            "False alerts on this test set": r["FP"],
            "True alerts on this test set": r["TP"],
            "Alerts that are real (%)": r["Attack Precision (%)"],
        })
    t_op = pd.DataFrame(op)
    t_op.to_csv(f"{TAB}/table05_alert_volume.csv", index=False)
    say(t_op.round(3).to_markdown(index=False))
    say(f"\n(Test partition contains {ben_test:,} benign flows.)\n")

    # =========================================================================
    # PAIRED SIGNIFICANCE TESTS on real predictions
    # =========================================================================
    say("## Section 5.4 - McNemar exact tests (paired, real predictions)\n")
    sig = []
    for a, b in combinations(fitted.keys(), 2):
        n01, n10, p = mcnemar_exact(yte, preds[a], preds[b])
        sig.append({
            "Model A": a, "Model B": b,
            "A right / B wrong": n01, "A wrong / B right": n10,
            "p-value": p,
            "Significant at 0.05": "yes" if p < 0.05 else "no",
        })
    t9 = pd.DataFrame(sig).sort_values("p-value")
    t9.to_csv(f"{TAB}/table_mcnemar_significance.csv", index=False)
    say(t9.round(5).to_markdown(index=False))
    say()

    with open(f"{args.out}/summary.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(log))
    print(f"\nDone. All tables written to {args.out}/")
    print(f"Paste-ready summary: {args.out}/summary.md")


if __name__ == "__main__":
    main()
