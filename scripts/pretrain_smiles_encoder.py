"""Pretrain a compact SMILES encoder with masked-token prediction."""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.nn import functional as F
from drugscreen.models.smiles_transformer import SmilesMLM, SmilesTokenizer

ROOT = Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(); p.add_argument('--epochs',type=int,default=8); p.add_argument('--seed',type=int,default=42); p.add_argument('--max-length',type=int,default=96); a=p.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    data=pd.read_csv(ROOT/'data/processed/joint_multitask.csv'); splits=pd.read_csv(ROOT/'data/processed/scaffold_splits.csv'); smiles=data.loc[splits['split']=='train','smiles'].dropna().drop_duplicates().tolist(); tokenizer=SmilesTokenizer.build(smiles,a.max_length); ids=torch.tensor([tokenizer.encode(s) for s in smiles],dtype=torch.long)
    model=SmilesMLM(len(tokenizer.vocabulary),a.max_length,d_model=64,nhead=4,layers=2,feedforward=128); optimizer=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-4); batch=256; history=[]
    for epoch in range(a.epochs):
        model.train(); order=torch.randperm(len(ids)); losses=[]
        for start in range(0,len(ids),batch):
            original=ids[order[start:start+batch]].clone(); masked=original.clone(); labels=torch.full_like(original,-100); candidates=(original!=tokenizer.pad_id)&(original!=tokenizer.bos_id)&(original!=tokenizer.eos_id); selected=(torch.rand_like(original.float())<0.15)&candidates; labels[selected]=original[selected]; masked[selected]=tokenizer.mask_id
            optimizer.zero_grad(); logits=model(masked); loss=F.cross_entropy(logits.view(-1,len(tokenizer.vocabulary)),labels.view(-1),ignore_index=-100); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step(); losses.append(loss.detach().item())
        history.append(float(np.mean(losses))); print(f'epoch {epoch+1}/{a.epochs} loss={history[-1]:.4f}')
    out=ROOT/'artifacts/pretraining'; out.mkdir(parents=True,exist_ok=True); torch.save({'state_dict':model.state_dict(),'tokenizer':tokenizer.state(),'config':{'d_model':64,'nhead':4,'layers':2,'feedforward':128}},out/'smiles_mlm.pt'); (ROOT/'reports/smiles_pretraining.json').write_text(json.dumps({'molecules':len(smiles),'vocabulary_size':len(tokenizer.vocabulary),'max_length':a.max_length,'epochs':a.epochs,'loss':history},indent=2)+'\n')

if __name__=='__main__': main()
