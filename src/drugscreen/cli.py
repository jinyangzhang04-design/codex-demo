import argparse
import pandas as pd
from drugscreen.data.standardize import standardize_smiles
from drugscreen.features.descriptors import featurize
from drugscreen.models.baseline import BaselineModel
from drugscreen.screening.rank import rank_candidates

def main():
    p = argparse.ArgumentParser(prog="drugscreen")
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train"); t.add_argument("--input", required=True); t.add_argument("--model", required=True)
    s = sub.add_parser("screen"); s.add_argument("--input", required=True); s.add_argument("--model", required=True); s.add_argument("--output", required=True)
    a = p.parse_args()
    if a.cmd == "train":
        df = pd.read_csv(a.input); df["smiles"] = df.smiles.map(standardize_smiles); df = df.dropna(subset=["smiles"])
        tasks = [c for c in df.columns if c != "smiles"]; X = pd.DataFrame([featurize(x) for x in df.smiles]).values; y = df[tasks].values
        BaselineModel().fit(X, y).save(a.model); print(f"trained {len(df)} molecules on {len(tasks)} tasks")
    else:
        df = pd.read_csv(a.input); df["smiles"] = df.smiles.map(standardize_smiles); df = df.dropna(subset=["smiles"])
        X = pd.DataFrame([featurize(x) for x in df.smiles]).values; pred = BaselineModel.load(a.model).predict(X)
        out = pd.DataFrame(pred, columns=[f"task_{i}" for i in range(pred.shape[1])]); out.insert(0, "smiles", df.smiles.values); rank_candidates(out).to_csv(a.output, index=False); print(f"screened {len(out)} molecules")
