"""Train per-task ECFP random-forest baselines on the saved scaffold split."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (average_precision_score, f1_score, mean_absolute_error,
                             mean_squared_error, r2_score, roc_auc_score)

from drugscreen.features.descriptors import featurize

ROOT = Path(__file__).resolve().parents[1]

def metric(task, y_true, pred):
    if task in {"bbbp", "clintox_ct_tox"}:
        labels = (pred >= 0.5).astype(int)
        result = {"n": int(len(y_true)), "positive_rate": float(np.mean(y_true))}
        for name, fn in (("roc_auc", roc_auc_score), ("pr_auc", average_precision_score), ("f1", f1_score)):
            try:
                result[name] = float(fn(y_true, pred if name != "f1" else labels))
            except ValueError:
                result[name] = None
        return result
    return {"n": int(len(y_true)), "rmse": float(mean_squared_error(y_true, pred) ** 0.5), "mae": float(mean_absolute_error(y_true, pred)), "r2": float(r2_score(y_true, pred))}

def run_seed(data, splits, features, tasks, seed):
    reports, models = {}, {}
    for task in tasks:
        observed = data[task].notna().to_numpy()
        task_split = splits[f"split_{task}"].to_numpy()
        train = observed & (task_split == "train")
        test = observed & (task_split == "test")
        x_train, y_train = features[train], data.loc[train, task].to_numpy()
        x_test, y_test = features[test], data.loc[test, task].to_numpy()
        if task in {"bbbp", "clintox_ct_tox"}:
            model = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1, class_weight="balanced")
            model.fit(x_train, y_train.astype(int))
            pred = model.predict_proba(x_test)[:, 1] if len(model.classes_) == 2 else np.full(len(y_test), float(model.classes_[0]))
        else:
            model = RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1)
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
        reports[task] = {"train_n": int(train.sum()), "test_n": int(test.sum()), "metrics": metric(task, y_test, pred)}
        models[task] = model
    return reports, models

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", default=None, help="comma-separated seeds for aggregate evaluation")
    args = parser.parse_args()
    data = pd.read_csv(ROOT / "data/processed/joint_multitask.csv")
    splits = pd.read_csv(ROOT / "data/processed/scaffold_splits.csv")
    tasks = [column for column in data.columns if column != "smiles"]
    features = np.vstack([featurize(smiles) for smiles in data.smiles])
    seeds = [int(value) for value in args.seeds.split(",")] if args.seeds else [args.seed]
    runs = []
    models = None
    for seed in seeds:
        reports, models = run_seed(data, splits, features, tasks, seed)
        runs.append({"seed": seed, "tasks": reports})
    aggregate = {}
    for task in tasks:
        metric_names = list(runs[0]["tasks"][task]["metrics"])
        aggregate[task] = {name: {"mean": float(np.mean([run["tasks"][task]["metrics"][name] for run in runs])), "std": float(np.std([run["tasks"][task]["metrics"][name] for run in runs]))} for name in metric_names}
    output = ROOT / "artifacts" / "baseline"
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump({"models": models, "tasks": tasks, "feature_size": int(features.shape[1])}, output / "ecfp_random_forest.joblib")
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/baseline_metrics.json").write_text(json.dumps({"model": "ECFP + Random Forest", "seeds": seeds, "runs": runs, "aggregate": aggregate}, indent=2) + "\n")
    print(json.dumps({"seeds": seeds, "aggregate": aggregate}, indent=2))

if __name__ == "__main__":
    main()
