"""Fine-tune the pretrained SMILES encoder with masked multi-task losses."""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch import nn
from sklearn.metrics import average_precision_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from drugscreen.models.smiles_transformer import SmilesMLM, SmilesMultiTask, SmilesTokenizer

ROOT=Path(__file__).resolve().parents[1]
TASK_TYPES={'esol':'regression','freesolv':'regression','bbbp':'classification','clintox_ct_tox':'classification'}

def threshold(y,p):
    if len(np.unique(y))<2:return .5
    return float(max(np.linspace(.05,.95,37),key=lambda t:f1_score(y,p>=t,zero_division=0)))

def score(task,y,p,t=.5):
    if TASK_TYPES[task]=='classification':return {'n':int(len(y)),'positive_rate':float(y.mean()),'threshold':float(t),'roc_auc':float(roc_auc_score(y,p)),'pr_auc':float(average_precision_score(y,p)),'f1':float(f1_score(y,p>=t,zero_division=0))}
    return {'n':int(len(y)),'rmse':float(mean_squared_error(y,p)**.5),'mae':float(mean_absolute_error(y,p)),'r2':float(r2_score(y,p))}

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',default='artifacts/pretraining/smiles_mlm.pt');p.add_argument('--epochs',type=int,default=20);p.add_argument('--seed',type=int,default=42);p.add_argument('--split-mode',choices=['global','task'],default='global');a=p.parse_args();torch.manual_seed(a.seed);np.random.seed(a.seed)
    d=pd.read_csv(ROOT/'data/processed/joint_multitask.csv');s=pd.read_csv(ROOT/'data/processed/scaffold_splits.csv');tasks=[c for c in d if c!='smiles']; checkpoint=torch.load(ROOT/a.checkpoint,map_location='cpu'); tokenizer=SmilesTokenizer.from_state(checkpoint['tokenizer']); ids=torch.tensor([tokenizer.encode(v) for v in d.smiles],dtype=torch.long); y=d[tasks].to_numpy(float); observed=~np.isnan(y)
    config=checkpoint['config']; base=SmilesMLM(len(tokenizer.vocabulary),tokenizer.max_length,**config); base.load_state_dict(checkpoint['state_dict']); model=SmilesMultiTask(base.encoder,len(tasks)); optimizer=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-4)
    target=np.nan_to_num(y,nan=0).astype(np.float32); mean=np.zeros(len(tasks),np.float32); scale=np.ones(len(tasks),np.float32); train=np.zeros_like(observed);val=np.zeros_like(observed)
    for i,t in enumerate(tasks):
        split_column='split' if a.split_mode=='global' else f'split_{t}'
        train[:,i]=(s[split_column]=='train')&observed[:,i];val[:,i]=(s[split_column]=='validation')&observed[:,i]
        if TASK_TYPES[t]=='regression': values=y[train[:,i],i];mean[i]=values.mean();scale[i]=max(values.std(),1e-6);target[:,i]=(target[:,i]-mean[i])/scale[i]
    tx,ty=ids,torch.from_numpy(target); tm,vm=torch.from_numpy(train),torch.from_numpy(val);best=None;best_loss=float('inf')
    def loss(mask):
        output=model(tx);terms=[]
        for i,t in enumerate(tasks):
            observed_task=mask[:,i]
            if not torch.any(observed_task):continue
            terms.append(nn.functional.binary_cross_entropy_with_logits(output[observed_task,i],ty[observed_task,i]) if TASK_TYPES[t]=='classification' else nn.functional.mse_loss(output[observed_task,i],ty[observed_task,i]))
        return torch.stack(terms).mean()
    for epoch in range(a.epochs):
        model.train(); optimizer.zero_grad(); train_loss=loss(tm); train_loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1); optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss=float(loss(vm))
        if val_loss<best_loss:best_loss=val_loss;best={k:v.detach().clone() for k,v in model.state_dict().items()}
        print(f'epoch {epoch+1}/{a.epochs} train={train_loss.detach().item():.4f} val={val_loss:.4f}')
    model.load_state_dict(best);model.eval();
    with torch.no_grad():output=model(tx).numpy()
    pred=output.copy()
    for i,t in enumerate(tasks):pred[:,i]=1/(1+np.exp(-np.clip(output[:,i],-30,30))) if TASK_TYPES[t]=='classification' else output[:,i]*scale[i]+mean[i]
    report={'model':'pretrained SMILES Transformer + masked multi-task heads','split_mode':a.split_mode,'seed':a.seed,'tasks':{}}
    for i,t in enumerate(tasks):
        split_column='split' if a.split_mode=='global' else f'split_{t}'
        test=(s[split_column]=='test')&observed[:,i];th=threshold(y[val[:,i],i],pred[val[:,i],i]) if TASK_TYPES[t]=='classification' else .5;report['tasks'][t]={'train_n':int(train[:,i].sum()),'validation_n':int(val[:,i].sum()),'test_n':int(test.sum()),'metrics':score(t,y[test,i],pred[test,i],th)}
    out=ROOT/'artifacts/smiles_finetuned';out.mkdir(parents=True,exist_ok=True);torch.save({'state_dict':model.state_dict(),'tokenizer':tokenizer.state(),'tasks':tasks,'mean':mean,'scale':scale},out/f'{a.split_mode}_model.pt');(ROOT/f'reports/smiles_finetuned_metrics_{a.split_mode}.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
