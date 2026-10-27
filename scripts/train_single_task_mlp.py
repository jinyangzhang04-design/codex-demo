"""Train single-task ECFP MLP baselines using the task-specific splits."""
import argparse, json, warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from drugscreen.features.descriptors import featurize

ROOT=Path(__file__).resolve().parents[1]
TASK_TYPES={'esol':'regression','freesolv':'regression','bbbp':'classification','clintox_ct_tox':'classification'}
warnings.filterwarnings('ignore', category=ConvergenceWarning)

def select_threshold(y,p):
    if len(np.unique(y))<2:return .5
    return float(max(np.linspace(.05,.95,37),key=lambda t:f1_score(y,p>=t,zero_division=0)))

def score(task,y,p,t=.5):
    if TASK_TYPES[task]=='classification':return {'n':int(len(y)),'positive_rate':float(y.mean()),'threshold':float(t),'roc_auc':float(roc_auc_score(y,p)),'pr_auc':float(average_precision_score(y,p)),'f1':float(f1_score(y,p>=t,zero_division=0))}
    return {'n':int(len(y)),'rmse':float(mean_squared_error(y,p)**.5),'mae':float(mean_absolute_error(y,p)),'r2':float(r2_score(y,p))}

def run(data,splits,features,tasks,seed):
    reports,models={},{}
    for task in tasks:
        observed=data[task].notna().to_numpy(); split=splits[f'split_{task}'].to_numpy(); train=observed&(split=='train'); val=observed&(split=='validation'); test=observed&(split=='test'); x_train=features[train];x_val=features[val];x_test=features[test]; y_train=data.loc[train,task].to_numpy();y_val=data.loc[val,task].to_numpy();y_test=data.loc[test,task].to_numpy()
        if TASK_TYPES[task]=='classification':
            model=make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=(256,128),early_stopping=True,validation_fraction=.15,max_iter=200,batch_size=64,random_state=seed))
            model.fit(x_train,y_train.astype(int)); p_val=model.predict_proba(x_val)[:,1];p_test=model.predict_proba(x_test)[:,1];t=select_threshold(y_val.astype(int),p_val)
        else:
            y_mean=float(y_train.mean());y_scale=max(float(y_train.std()),1e-6);model=make_pipeline(StandardScaler(),MLPRegressor(hidden_layer_sizes=(256,128),early_stopping=True,validation_fraction=.15,max_iter=200,batch_size=64,random_state=seed));model.fit(x_train,(y_train-y_mean)/y_scale);p_test=model.predict(x_test)*y_scale+y_mean;t=.5
        reports[task]={'train_n':int(train.sum()),'validation_n':int(val.sum()),'test_n':int(test.sum()),'metrics':score(task,y_test,p_test,t)};models[task]=model
    return reports,models

def main():
    p=argparse.ArgumentParser();p.add_argument('--seeds',default='0,1,42');a=p.parse_args();d=pd.read_csv(ROOT/'data/processed/joint_multitask.csv');s=pd.read_csv(ROOT/'data/processed/scaffold_splits.csv');tasks=[c for c in d if c!='smiles'];x=np.vstack([featurize(v) for v in d.smiles]);runs=[];models=None
    for seed in [int(v) for v in a.seeds.split(',')]:
        reports,models=run(d,s,x,tasks,seed);runs.append({'seed':seed,'tasks':reports})
    aggregate={}
    for task in tasks:
        names=list(runs[0]['tasks'][task]['metrics']);aggregate[task]={name:{'mean':float(np.mean([r['tasks'][task]['metrics'][name] for r in runs])),'std':float(np.std([r['tasks'][task]['metrics'][name] for r in runs]))} for name in names}
    out=ROOT/'artifacts/baseline';out.mkdir(parents=True,exist_ok=True);joblib.dump({'models':models,'tasks':tasks},out/'ecfp_single_task_mlp.joblib');report={'model':'ECFP + single-task MLP','seeds':[int(v) for v in a.seeds.split(',')],'runs':runs,'aggregate':aggregate};(ROOT/'reports/single_task_mlp_metrics.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'aggregate':aggregate},indent=2))

if __name__=='__main__':main()
