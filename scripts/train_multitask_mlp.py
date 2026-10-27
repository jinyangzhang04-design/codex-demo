"""Train and evaluate the masked multi-task ECFP MLP."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score

from drugscreen.features.descriptors import featurize
from drugscreen.models.masked_multitask_mlp import MaskedMultiTaskMLP

ROOT = Path(__file__).resolve().parents[1]
TASK_TYPES = {"esol": "regression", "freesolv": "regression", "bbbp": "classification", "clintox_ct_tox": "classification"}

def score(task, y, pred):
    if TASK_TYPES[task] == "classification":
        return {"n": int(len(y)), "positive_rate": float(y.mean()), "roc_auc": float(roc_auc_score(y, pred)), "pr_auc": float(average_precision_score(y, pred)), "f1": float(f1_score(y, pred >= 0.5))}
    return {"n": int(len(y)), "rmse": float(mean_squared_error(y, pred) ** 0.5), "mae": float(mean_absolute_error(y, pred)), "r2": float(r2_score(y, pred))}

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--seed", type=int, default=42); parser.add_argument("--epochs", type=int, default=600); args = parser.parse_args()
    data = pd.read_csv(ROOT / "data/processed/joint_multitask.csv")
    splits = pd.read_csv(ROOT / "data/processed/scaffold_splits.csv")
    tasks = [column for column in data.columns if column != "smiles"]
    x = np.vstack([featurize(smiles) for smiles in data.smiles])
    y = data[tasks].to_numpy(dtype=float)
    mask = (~np.isnan(y)).astype(np.float32)
    train_mask = np.zeros_like(mask)
    for index, task in enumerate(tasks):
        train_mask[:, index] = (splits[f"split_{task}"].to_numpy() == "train") & (mask[:, index] > 0)
    model = MaskedMultiTaskMLP(tasks, TASK_TYPES, epochs=args.epochs, seed=args.seed).fit(x, y, train_mask)
    pred = model.predict(x)
    report = {"model": "ECFP + masked multi-task MLP", "seed": args.seed, "tasks": {}}
    for index, task in enumerate(tasks):
        test = (splits[f"split_{task}"].to_numpy() == "test") & (mask[:, index] > 0)
        report["tasks"][task] = {"train_n": int(train_mask[:, index].sum()), "test_n": int(test.sum()), "metrics": score(task, y[test, index], pred[test, index])}
    output = ROOT / "artifacts" / "multitask"; output.mkdir(parents=True, exist_ok=True)
    model.save(output / "ecfp_masked_mlp.joblib")
    (ROOT / "reports/multitask_mlp_metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
