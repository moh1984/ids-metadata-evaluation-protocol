"""Corrected ablation: genuinely NESTED feature sets, each a superset of the previous."""
import argparse, os, sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ids_pipeline import engineer_features, FEATURE_ORDER
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

ap=argparse.ArgumentParser()
ap.add_argument('--data', default=os.environ.get('IDS_DATA','data/cybersecurity_dataset.csv'))
ap.add_argument('--seed', type=int, default=int(os.environ.get('IDS_SEED', 42)))
args=ap.parse_args()
SEED=args.seed
os.makedirs('results/tables', exist_ok=True)
df=pd.read_csv(args.data)
X=engineer_features(df); y=df['label'].astype(int).values; m=df['attack_type'].astype(str).values
Xtr,Xte,ytr,yte,_,_=train_test_split(X,y,m,test_size=.30,stratify=pd.Series(m),random_state=SEED)
pw=(len(ytr)-ytr.sum())/ytr.sum()
cv=StratifiedKFold(5,shuffle=True,random_state=SEED)

RAW=['bytes_sent','bytes_received','src_port','dst_port','protocol_encoded','is_internal','hour']
G_TRAF=['bytes_ratio','total_bytes','log_bytes_sent','log_bytes_received']
G_PORT=['port_diff','is_wellknown_dst','is_wellknown_src','same_port','dst_is_service','src_is_service']
G_UAURL=['ua_is_bot','ua_is_mobile','ua_len','has_url','url_has_login','url_has_admin','url_has_param','url_len']
G_TEMP=['is_night']

NESTED=[('Raw attributes only', RAW),
        ('+ traffic-volume derivations', RAW+G_TRAF),
        ('+ port-level indicators', RAW+G_TRAF+G_PORT),
        ('+ user-agent and URL features', RAW+G_TRAF+G_PORT+G_UAURL),
        ('+ temporal attribute (full set)', RAW+G_TRAF+G_PORT+G_UAURL+G_TEMP)]
ISOLATED=[('Raw + traffic-volume only', RAW+G_TRAF),
          ('Raw + port-level only', RAW+G_PORT),
          ('Raw + user-agent/URL only', RAW+G_UAURL),
          ('Raw + temporal only', RAW+G_TEMP)]

def canon(cols):
    """Return cols in canonical FEATURE_ORDER sequence.

    XGBoost's histogram split search can break ties differently when identical
    features are presented in a different column order, which shifts results in
    the third or fourth decimal. Ordering every subset canonically makes the
    final row of this table numerically identical to the full-feature model
    reported in Table 4.
    """
    return [f for f in FEATURE_ORDER if f in set(cols)]


def evaluate(cols):
    cols = canon(cols)
    p=Pipeline([('sc',StandardScaler()),('clf',XGBClassifier(
        n_estimators=500,learning_rate=0.05,max_depth=6,subsample=1.0,scale_pos_weight=pw,
        eval_metric='logloss',tree_method='hist',random_state=SEED,n_jobs=-1))])
    cvs=cross_val_score(p,Xtr[cols],ytr,cv=cv,scoring='average_precision',n_jobs=-1)
    p.fit(Xtr[cols],ytr); s=p.predict_proba(Xte[cols])[:,1]
    return cvs.mean(),cvs.std(),average_precision_score(yte,s),roc_auc_score(yte,s)

print("=== NESTED (cumulative) ===",flush=True)
rows=[];prev=None
for lbl,cols in NESTED:
    assert len(set(cols))==len(cols)
    cm,cs,tp,tr=evaluate(cols)
    d='' if prev is None else f'{tp-prev:+.4f}'
    rows.append({'Feature set':lbl,'n':len(cols),'CV PR-AUC':round(cm,4),'CV std':round(cs,4),
                 'Test PR-AUC':round(tp,4),'Delta PR-AUC':d,'Test ROC-AUC':round(tr,4)})
    prev=tp; print(rows[-1],flush=True)
pd.DataFrame(rows).to_csv('results/tables/table07_feature_ablation.csv',index=False)

print("\n=== ISOLATED (raw + one group) ===",flush=True)
r2=[]
base=evaluate(RAW)
r2.append({'Feature set':'Raw attributes only','n':7,'CV PR-AUC':round(base[0],4),
           'Test PR-AUC':round(base[2],4),'Test ROC-AUC':round(base[3],4)})
print(r2[-1],flush=True)
for lbl,cols in ISOLATED:
    cm,cs,tp,tr=evaluate(cols)
    r2.append({'Feature set':lbl,'n':len(cols),'CV PR-AUC':round(cm,4),
               'Test PR-AUC':round(tp,4),'Test ROC-AUC':round(tr,4)})
    print(r2[-1],flush=True)
pd.DataFrame(r2).to_csv('results/tables/table07b_ablation_isolated.csv',index=False)

# ---- per-fold CV scores for every model (Table 6 / Figure 5) ----
import json
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

_models = {
 'Logistic Regression': LogisticRegression(C=0.01, max_iter=5000, class_weight='balanced', random_state=SEED),
 'SVM (RBF Kernel)': SVC(C=0.1, gamma=0.001, probability=True, class_weight='balanced', random_state=SEED),
 'Random Forest': RandomForestClassifier(n_estimators=500, max_depth=None, max_features=0.5,
                                         min_samples_split=3, class_weight='balanced_subsample',
                                         random_state=SEED, n_jobs=-1),
 'Gradient Boosting': XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=1.0,
                                    scale_pos_weight=pw, eval_metric='logloss', tree_method='hist',
                                    random_state=SEED, n_jobs=-1),
 'Neural Network (FFNN)': MLPClassifier(hidden_layer_sizes=(64, 32), alpha=0.01, learning_rate_init=0.01,
                                        max_iter=1500, early_stopping=True, random_state=SEED),
}
print("\n=== per-fold CV PR-AUC ===", flush=True)
_folds = {}
for _n, _c in _models.items():
    _s = cross_val_score(Pipeline([('sc', StandardScaler()), ('clf', _c)]),
                         Xtr, ytr, cv=cv, scoring='average_precision', n_jobs=-1)
    _folds[_n] = [round(float(v), 4) for v in _s]
    print(_n, _folds[_n], f"mean={_s.mean():.4f}", flush=True)
json.dump(_folds, open('results/tables/cv_fold_scores.json', 'w'), indent=1)

# ---- feature importances for Figure 7 ----
_rf = RandomForestClassifier(n_estimators=500, max_depth=None, max_features=0.5,
                             min_samples_split=3, class_weight='balanced_subsample',
                             random_state=SEED, n_jobs=-1)
_pl = Pipeline([('sc', StandardScaler()), ('clf', _rf)]).fit(Xtr[FEATURE_ORDER], ytr)
_rf_imp = _pl.named_steps['clf'].feature_importances_
_xg = XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=1.0,
                    scale_pos_weight=pw, eval_metric='logloss', tree_method='hist',
                    random_state=SEED, n_jobs=-1)
_pl2 = Pipeline([('sc', StandardScaler()), ('clf', _xg)]).fit(Xtr[FEATURE_ORDER], ytr)
_xg_imp = _pl2.named_steps['clf'].feature_importances_
pd.DataFrame({'Feature': FEATURE_ORDER,
              'RandomForest (Gini)': _rf_imp,
              'GradientBoosting (gain)': _xg_imp}).to_csv(
    'results/tables/feature_importances.csv', index=False)
print('feature importances written', flush=True)

print('DONE', flush=True)
