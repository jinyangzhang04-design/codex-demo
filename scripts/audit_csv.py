"""Audit a CSV containing a smiles column and one or more task columns."""
import argparse
import json
from pathlib import Path
import pandas as pd
from rdkit import Chem

def audit(path: str):
    df = pd.read_csv(path)
    smiles_col = next((c for c in df.columns if c.lower() in {"smiles", "canonical_smiles"}), None)
    if smiles_col is None:
        raise ValueError("CSV must contain a smiles column")
    canonical = df[smiles_col].map(lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(str(x))) if Chem.MolFromSmiles(str(x)) else None)
    tasks = [c for c in df.columns if c != smiles_col]
    return {"path": str(path), "rows": len(df), "columns": list(df.columns), "valid_smiles": int(canonical.notna().sum()), "invalid_smiles": int(canonical.isna().sum()), "unique_canonical_smiles": int(canonical.dropna().nunique()), "missing_by_task": {c: int(df[c].isna().sum()) for c in tasks}, "task_dtypes": {c: str(df[c].dtype) for c in tasks}}

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("csv"); p.add_argument("--output")
    args = p.parse_args(); result = audit(args.csv); print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output: Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
