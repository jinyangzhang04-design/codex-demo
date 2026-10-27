"""Train the PyTorch masked multi-task model on ECFP inputs."""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.metrics import average_precision_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from drugscreen.features.descriptors import featurize
from drugscreen.models.torch_multitask import TorchMaskedMultiTask

ROOT = Path(__file__).resolve().parents[1]
TASK_TYPES = {"esol":"regression", "freesolv":"regression", "bbbp":"classification", "clintox_ct_tox":"classification"}

def score(task, y, pred):
    if TASK_TYPES[task] == "classification": return {"n":int(len(y)), "positive_rate":float(y.mean()), "roc_auc":float(roc_auc_score(y,pred)), "pr_auc":float(average_precision_score(y,pred)), "f1":float(f1_score(y,pred>=.5))}
    return {"n":int(len(y)), "rmse":float(mean_squared_error(y,pred)**.5), "mae":float(mean_absolute_error(y,pred)), "r2":float(r2_score(y,pred))}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--epochs',type=int,default=300); p.add_argument('--seed',type=int,default=42); a=p.parse_args()
    d=pd.read_csv(ROOT/'data/processed/joint_multitask.csv'); s=pd.read_csv(ROOT/'data/processed/scaffold_splits.csv'); tasks=[c for c in d if c!='smiles']; x=np.vstack([featurize(v) for v in d.smiles]); y=d[tasks].to_numpy(float); observed=~np.isnan(y)
    train=np.zeros_like(observed); val=np.zeros_like(observed)
    for i,t in enumerate(tasks): train[:,i]=(s[f'split_{t}']=='train')&observed[:,i]; val[:,i]=(s[f'split_{t}']=='validation')&observed[:,i]
    model=TorchMaskedMultiTask(tasks,TASK_TYPES,epochs=a.epochs,seed=a.seed).fit(x,y,train,val); pred=model.predict(x); report={'model':'ECFP + PyTorch masked multi-task MLP','seed':a.seed,'tasks':{}}
    for i,t in enumerate(tasks):
        test=(s[f'split_{t}']=='test')&observed[:,i]; report['tasks'][t]={'train_n':int(train[:,i].sum()),'validation_n':int(val[:,i].sum()),'test_n':int(test.sum()),'metrics':score(t,y[test,i],pred[test,i])}
    out=ROOT/'artifacts/torch_multitask'; out.mkdir(parents=True,exist_ok=True); torch.save({'state_dict':model.net_.state_dict(),'tasks':tasks,'x_mean':model.x_mean_,'x_scale':model.x_scale_,'y_mean':model.y_mean_,'y_scale':model.y_scale_},out/'model.pt'); (ROOT/'reports/torch_multitask_metrics.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))

if __name__=='__main__': main()
