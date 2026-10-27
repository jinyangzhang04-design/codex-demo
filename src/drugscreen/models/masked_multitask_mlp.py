"""Small NumPy multi-task MLP with per-task missing-label masks.

This keeps the first research baseline lightweight and reproducible. A later
experiment can replace the input representation with a pretrained encoder
without changing the masked loss or prediction interface.
"""
from dataclasses import dataclass
import joblib
import numpy as np

@dataclass
class MaskedMultiTaskMLP:
    task_names: list[str]
    task_types: dict[str, str]
    hidden_size: int = 256
    epochs: int = 250
    learning_rate: float = 2e-3
    seed: int = 42

    def fit(self, x, y, mask):
        rng = np.random.default_rng(self.seed)
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        mask = np.asarray(mask, dtype=np.float32)
        self.x_mean_ = x.mean(axis=0)
        self.x_scale_ = x.std(axis=0)
        self.x_scale_[self.x_scale_ < 1e-6] = 1.0
        x = (x - self.x_mean_) / self.x_scale_
        n_features, n_tasks = x.shape[1], y.shape[1]
        self.y_mean_ = np.zeros(n_tasks, dtype=np.float32)
        self.y_scale_ = np.ones(n_tasks, dtype=np.float32)
        for index, task in enumerate(self.task_names):
            values = y[mask[:, index] > 0, index]
            if self.task_types[task] == "regression" and len(values):
                self.y_mean_[index] = values.mean()
                self.y_scale_[index] = max(values.std(), 1e-6)
        ys = y.copy()
        for index, task in enumerate(self.task_names):
            if self.task_types[task] == "regression":
                ys[:, index] = (ys[:, index] - self.y_mean_[index]) / self.y_scale_[index]
            ys[:, index] = np.nan_to_num(ys[:, index], nan=0.0)
        self.w1_ = rng.normal(0, np.sqrt(2 / n_features), (n_features, self.hidden_size)).astype(np.float32)
        self.b1_ = np.zeros(self.hidden_size, dtype=np.float32)
        self.w2_ = rng.normal(0, np.sqrt(2 / self.hidden_size), (self.hidden_size, n_tasks)).astype(np.float32)
        self.b2_ = np.zeros(n_tasks, dtype=np.float32)
        # Adam state.
        params = [self.w1_, self.b1_, self.w2_, self.b2_]
        moments = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
        for step in range(1, self.epochs + 1):
            hidden = np.maximum(x @ self.w1_ + self.b1_, 0)
            logits = hidden @ self.w2_ + self.b2_
            grad_out = np.zeros_like(logits)
            for index, task in enumerate(self.task_names):
                observed = mask[:, index] > 0
                count = max(int(observed.sum()), 1)
                if self.task_types[task] == "classification":
                    probs = 1.0 / (1.0 + np.exp(-np.clip(logits[:, index], -30, 30)))
                    grad_out[observed, index] = (probs[observed] - ys[observed, index]) / count
                else:
                    grad_out[observed, index] = (logits[observed, index] - ys[observed, index]) / count
            grad_w2 = hidden.T @ grad_out
            grad_b2 = grad_out.sum(axis=0)
            grad_hidden = (grad_out @ self.w2_.T) * (hidden > 0)
            grad_w1 = x.T @ grad_hidden
            grad_b1 = grad_hidden.sum(axis=0)
            for param, grad, (m, v) in zip(params, [grad_w1, grad_b1, grad_w2, grad_b2], moments):
                m[:] = 0.9 * m + 0.1 * grad
                v[:] = 0.999 * v + 0.001 * (grad * grad)
                m_hat = m / (1 - 0.9 ** step)
                v_hat = v / (1 - 0.999 ** step)
                param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)
        return self

    def predict(self, x):
        x = (np.asarray(x, dtype=np.float32) - self.x_mean_) / self.x_scale_
        hidden = np.maximum(x @ self.w1_ + self.b1_, 0)
        logits = hidden @ self.w2_ + self.b2_
        result = logits.copy()
        for index, task in enumerate(self.task_names):
            if self.task_types[task] == "classification":
                result[:, index] = 1.0 / (1.0 + np.exp(-np.clip(logits[:, index], -30, 30)))
            else:
                result[:, index] = logits[:, index] * self.y_scale_[index] + self.y_mean_[index]
        return result

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
