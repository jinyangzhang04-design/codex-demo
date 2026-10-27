"""Audit scaffold overlap and cross-task molecule leakage in saved splits."""
import json
from pathlib import Path
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]

def scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol) if mol is not None else ""

def main():
    data = pd.read_csv(ROOT / "data/processed/joint_multitask.csv")
    splits = pd.read_csv(ROOT / "data/processed/scaffold_splits.csv")
    tasks = [column for column in data.columns if column != "smiles"]
    scaffolds = data.smiles.map(scaffold)
    result = {"tasks": {}, "cross_task_leakage": {}}
    for task in tasks:
        observed = data[task].notna()
        task_split = splits[f"split_{task}"]
        train_scaffolds = set(scaffolds[observed & (task_split == "train")])
        test_scaffolds = set(scaffolds[observed & (task_split == "test")])
        result["tasks"][task] = {
            "train_n": int((observed & (task_split == "train")).sum()),
            "validation_n": int((observed & (task_split == "validation")).sum()),
            "test_n": int((observed & (task_split == "test")).sum()),
            "train_test_scaffold_overlap": len(train_scaffolds & test_scaffolds),
        }
    for target in tasks:
        target_test = data[target].notna() & (splits[f"split_{target}"] == "test")
        result["cross_task_leakage"][target] = {}
        for source in tasks:
            if source == target:
                continue
            source_train = data[source].notna() & (splits[f"split_{source}"] == "train")
            result["cross_task_leakage"][target][source] = int((target_test & source_train).sum())
    global_train = set(data.loc[splits["split"] == "train", "smiles"])
    global_test = set(data.loc[splits["split"] == "test", "smiles"])
    result["global_split"] = {"train_n": len(global_train), "test_n": len(global_test), "molecule_overlap": len(global_train & global_test)}
    output = ROOT / "reports/split_leakage_audit.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
