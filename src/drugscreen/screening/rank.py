import pandas as pd

def rank_candidates(predictions: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.DataFrame:
    weights = weights or {c: 1.0 for c in predictions.columns if c != "smiles"}
    out = predictions.copy()
    out["score"] = sum(out[col] * weight for col, weight in weights.items() if col in out)
    return out.sort_values("score", ascending=False).reset_index(drop=True)
