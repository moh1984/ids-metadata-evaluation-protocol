import json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ids_pipeline import engineer_features, FEATURE_ORDER, wilson_ci, mcnemar_exact
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix, balanced_accuracy_score)

ap=argparse.ArgumentParser()
ap.add_argument('--data', default=os.environ.get('IDS_DATA','data/cybersecurity_dataset.csv'))
ap.add_argument('--seed', type=int, default=int(os.environ.get('IDS_SEED', 42)))
args=ap.parse_args()
SEED=args.seed
os.makedirs('results/tables', exist_ok=True)
df=pd.read_csv(args.data)
X=engineer_features(df); y=df['label'].astype(int).values
m=df['attack_type'].astype(str).str.lower().values
Xtr,Xte,ytr,yte,mtr,mte=train_test_split(X,y,m,test_size=.30,stratify=pd.Series(m),random_state=SEED)
pw=(len(ytr)-ytr.sum())/ytr.sum()
le=LabelEncoder().fit(np.concatenate([mtr,mte])); etr,ete=le.transform(mtr),le.transform(mte)
minc=int(pd.Series(mtr).value_counts().min())
cv=StratifiedKFold(min(5,max(2,minc)),shuffle=True,random_state=SEED)
print('classes',list(le.classes_),'| min train class',minc,'| folds',cv.n_splits,flush=True)

def P(c): return Pipeline([('sc',StandardScaler()),('clf',c)])
grids={
 'Logistic Regression':(P(LogisticRegression(max_iter=5000,class_weight='balanced',random_state=SEED)),
                        {'clf__C':[0.01,0.1,1.0,10.0]}),
 'SVM (RBF Kernel)':(P(SVC(class_weight='balanced',probability=False,random_state=SEED)),
                     {'clf__C':[1.0,10.0],'clf__gamma':['scale',0.001]}),
 'Random Forest':(P(RandomForestClassifier(class_weight='balanced_subsample',random_state=SEED,n_jobs=-1)),
                  {'clf__n_estimators':[300,500],'clf__max_depth':[None,15],'clf__max_features':['sqrt',0.5]}),
 'Gradient Boosting':(P(XGBClassifier(eval_metric='mlogloss',tree_method='hist',random_state=SEED,n_jobs=-1)),
                      {'clf__n_estimators':[300,500],'clf__learning_rate':[0.05,0.1],'clf__max_depth':[4,6]}),
 'Neural Network (FFNN)':(P(MLPClassifier(max_iter=1500,early_stopping=True,random_state=SEED)),
                          {'clf__hidden_layer_sizes':[(64,32),(128,64,32)],'clf__alpha':[1e-3,1e-2]}),
}
rows=[];preds={};best={}
for n,(pipe,g) in grids.items():
    print('[mc]',n,flush=True)
    gs=GridSearchCV(pipe,g,scoring='f1_macro',cv=cv,n_jobs=-1); gs.fit(Xtr,etr)
    yp=le.inverse_transform(gs.predict(Xte)); preds[n]=yp; best[n]=gs.best_params_
    k=int((yp==mte).sum()); lo,hi=wilson_ci(k,len(mte))
    rows.append({'Model':n,'Accuracy (%)':100*accuracy_score(mte,yp),'CI':f'[{lo:.2f}-{hi:.2f}]',
        'Balanced Acc (%)':100*balanced_accuracy_score(mte,yp),
        'Weighted F1 (%)':100*f1_score(mte,yp,average='weighted',zero_division=0),
        'Macro F1 (%)':100*f1_score(mte,yp,average='macro',zero_division=0)})
    print('   ',rows[-1],gs.best_params_,flush=True)
t8=pd.DataFrame(rows); t8.to_csv('results/tables/table08_multiclass.csv',index=False)
json.dump({k:str(v) for k,v in best.items()},open('results/tables/mc_best_params.json','w'),indent=1)

bestm=t8.sort_values('Macro F1 (%)',ascending=False).iloc[0]['Model']
print('best multiclass:',bestm,flush=True)
yp=preds[bestm]; rep=classification_report(mte,yp,output_dict=True,zero_division=0)
pc=[]
for c in sorted(set(mte)):
    d=rep.get(c,{}); sup=int((mte==c).sum()); tp=int(((mte==c)&(yp==c)).sum()); pr=int((yp==c).sum())
    pc.append({'Attack Category':c,'Precision (%)':100*d.get('precision',0),'TP / predicted':f'{tp}/{pr}',
               'Recall (%)':100*d.get('recall',0),'TP / support':f'{tp}/{sup}',
               'F1 (%)':100*d.get('f1-score',0),'Test Support':sup})
pd.DataFrame(pc).to_csv('results/tables/table09_perclass.csv',index=False)
labels=sorted(set(mte))
cm=confusion_matrix(mte,yp,labels=labels)
np.savetxt('results/tables/confusion_matrix_multiclass.csv',cm,delimiter=',',fmt='%d')
json.dump(labels,open('results/tables/cm_labels.json','w'))
print('multiclass done',flush=True)

# ---------------- SHAP on the NEW tuned binary XGB ----------------
import shap
bin_xgb=Pipeline([('sc',StandardScaler()),('clf',XGBClassifier(n_estimators=500,learning_rate=0.05,
    max_depth=6,subsample=1.0,scale_pos_weight=pw,eval_metric='logloss',tree_method='hist',
    random_state=SEED,n_jobs=-1))]).fit(Xtr,ytr)
Xs=bin_xgb.named_steps['sc'].transform(Xte)
sv=shap.TreeExplainer(bin_xgb.named_steps['clf']).shap_values(Xs)
if isinstance(sv,list): sv=sv[1]
if sv.ndim==3: sv=sv[:,:,1]
imp=np.abs(sv).mean(0)
t10=(pd.DataFrame({'Feature':FEATURE_ORDER,'Mean |SHAP|':imp})
     .sort_values('Mean |SHAP|',ascending=False).head(10).reset_index(drop=True))
t10.insert(0,'Rank',range(1,len(t10)+1))
t10.to_csv('results/tables/table10_shap.csv',index=False)
np.save('results/shap_values.npy',sv); np.save('results/shap_X.npy',Xs)
print(t10.round(4).to_string(index=False),flush=True)
print('ALL DONE',flush=True)
