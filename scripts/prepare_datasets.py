"""Normalize the available public datasets into one sparse multi-task table."""
import argparse
import gzip
import json
import random
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

def canonical(smiles):
    if pd.isna(smiles):
        return None
    mol = Chem.MolFromSmiles(str(smiles).strip())
    return Chem.MolToSmiles(mol, canonical=True) if mol is not None else None

def aggregate(df, target, kind="regression"):
    df = df[["smiles", target]].copy()
    df["smiles"] = df["smiles"].map(canonical)
    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df.dropna(subset=["smiles"])
    if kind == "classification":
        # A repeated molecule can have conflicting assay labels. Keep a
        # deterministic majority label rather than creating an invalid 0.5
        # class, which would break ROC-AUC and PR-AUC.
        return df.groupby("smiles", as_index=False)[target].agg(lambda values: values.mode().iloc[0])
    return df.groupby("smiles", as_index=False)[target].mean()

def read_esol():
    path = RAW / "esol.csv"
    if not path.exists():
        return None
    raw = pd.read_csv(path)
    return aggregate(raw.rename(columns={"measured log solubility in mols per litre": "esol"}), "esol")

def read_clintox():
    path = RAW / "clintox.csv.gz"
    if not path.exists():
        return None
    raw = pd.read_csv(path, compression="gzip")
    return aggregate(raw.rename(columns={"CT_TOX": "clintox_ct_tox"}), "clintox_ct_tox", "classification")

def read_bbbp():
    path = RAW / "BBBP.csv"
    if not path.exists():
        return None
    raw = pd.read_csv(path)
    return aggregate(raw.rename(columns={"p_np": "bbbp"}), "bbbp", "classification")

def read_freesolv():
    path = RAW / "freesolv.txt"
    if not path.exists():
        return None
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        fields = [field.strip() for field in line.split(";", 9)]
        if len(fields) >= 4:
            rows.append({"smiles": fields[1], "freesolv": fields[3]})
    return aggregate(pd.DataFrame(rows), "freesolv")

def scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol) if mol is not None else ""

def make_splits(df, seed=42):
    groups = {}
    for index, smiles in enumerate(df["smiles"]):
        groups.setdefault(scaffold(smiles), []).append(index)
    keys = list(groups)
    random.Random(seed).shuffle(keys)
    target = {"train": len(df) * 0.8, "validation": len(df) * 0.1, "test": len(df) * 0.1}
    counts = {key: 0 for key in target}
    assignment = {}
    for key in sorted(keys, key=lambda x: len(groups[x]), reverse=True):
        # Fill the largest remaining quota first. This preserves the intended
        # 80/10/10 proportions even when scaffolds have very different sizes.
        split = max(target, key=lambda name: target[name] - counts[name])
        assignment.update({index: split for index in groups[key]})
        counts[split] += len(groups[key])
    return [assignment[i] for i in range(len(df))]

def make_task_split(df, task, seed=42):
    """Create an 80/10/10 scaffold split for one observed task."""
    observed = df[task].notna()
    subset = df.loc[observed, ["smiles"]].reset_index()
    local = make_splits(subset.rename(columns={"index": "original_index"}), seed)
    result = ["unused"] * len(df)
    for original, split in zip(subset["index"], local):
        result[int(original)] = split
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    tables = {"esol": read_esol(), "freesolv": read_freesolv(), "bbbp": read_bbbp(), "clintox": read_clintox()}
    available = {name: table for name, table in tables.items() if table is not None}
    if not available:
        raise SystemExit("No raw datasets found. Run scripts/download_datasets.py first.")
    joint = None
    report = {"datasets": {}, "available_datasets": list(available)}
    for name, table in available.items():
        table.to_csv(PROCESSED / f"{name}.csv", index=False)
        report["datasets"][name] = {"rows": len(table), "columns": list(table.columns), "unique_smiles": int(table.smiles.nunique())}
        joint = table if joint is None else joint.merge(table, on="smiles", how="outer")
    joint = joint.sort_values("smiles").reset_index(drop=True)
    joint.to_csv(PROCESSED / "joint_multitask.csv", index=False)
    task_names = [column for column in joint.columns if column != "smiles"]
    split = make_splits(joint, args.seed)
    split_table = pd.DataFrame({"smiles": joint.smiles, "split": split})
    for task in task_names:
        split_table[f"split_{task}"] = make_task_split(joint, task, args.seed)
    split_table.to_csv(PROCESSED / "scaffold_splits.csv", index=False)
    report["joint"] = {"rows": len(joint), "tasks": task_names, "missing_by_task": {column: int(joint[column].isna().sum()) for column in task_names}, "split_counts": pd.Series(split).value_counts().to_dict(), "task_split_counts": {task: pd.Series(split_table[f"split_{task}"]).value_counts().to_dict() for task in task_names}}
    (REPORTS / "data_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
