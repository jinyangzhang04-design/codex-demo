from dataclasses import dataclass
import joblib
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor

@dataclass
class BaselineModel:
    model: object | None = None
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model = MultiOutputRegressor(RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))
        self.model.fit(X, y)
        return self
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not fitted")
        return self.model.predict(X)
    def save(self, path: str):
        joblib.dump(self, path)
    @staticmethod
    def load(path: str):
        return joblib.load(path)
